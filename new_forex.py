# new_forex.py
# Orchestrator: scrape (ForexFactory) -> prompts -> GPT-5-mini/Typhoon -> parse -> validate -> persist -> H1 check -> EOD summary -> Telegram.
# - Includes the original scraper (Playwright primary, requests+BS fallback, lazy import).
# - Uses H1 as reference for order status checks.
# - Deterministic JSON output (+ confidence).
# - Scheduler-friendly: near-news / EOD; plus daily-quota fallback = >=1 order per pair per day.
# - Day profile: Mon wide TP & fewer plans → Fri narrow TP & more plans.

import os, re, json, time, math, argparse
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from bs4 import BeautifulSoup
import requests
from openai import OpenAI

from get_data import IQDataFetcher
from strategy_engine import (
    StrategyConfig, OCOPlan, PipMath, WeeklyLedger,
    plan_to_deterministic_json, maybe_reprice_pending, OrderLeg
)
from tele_signals import TelegramNotifier, TyphoonForexAnalyzer, log_today_summary_only

load_dotenv()

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
TYPHOON_API_KEY    = os.getenv("TYPHOON_API_KEY")  # optional fallback
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
MODEL_ID           = os.getenv("MODEL_ID", "gpt-5-mini")
PAIRS              = [p.strip() for p in os.getenv("PAIRS", "EUR/USD,GBP/USD,USD/JPY,EUR/GBP,EUR/CHF").split(",")]
BKK_TZ = timezone(timedelta(hours=7))

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
if not TELEGRAM_CHAT_ID:
    raise RuntimeError("Missing TELEGRAM_CHAT_ID (or CHAT_ID)")


# ---------- Prompt loading ----------

def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

SYSTEM_PROMPT = _load_text("prompts/system_gpt5.txt")
USER_TEMPLATE = _load_text("prompts/user_template.txt")

def format_user_prompt(ctx: dict) -> str:
    """Format a deterministic, data-first user prompt."""
    return USER_TEMPLATE.format(**ctx)


# ---------- Model clients ----------

def _openai_client() -> OpenAI:
    assert OPENAI_API_KEY, "Missing OPENAI_API_KEY"
    return OpenAI(api_key=OPENAI_API_KEY)

def _typhoon_client() -> TyphoonForexAnalyzer:
    assert TYPHOON_API_KEY, "Missing TYPHOON_API_KEY"
    return TyphoonForexAnalyzer(TYPHOON_API_KEY)


# =====================================================================
#                           SCRAPER SECTION
# =====================================================================

def _wait_for_event_rows(page, min_rows: int = 5, timeout_ms: int = 60000):
    start = time.time()
    while True:
        try:
            cnt = page.locator("table.calendar__table tr.calendar__row[data-event-id]").count()
            if cnt >= min_rows: return
        except Exception:
            pass
        if (time.time() - start) * 1000 > timeout_ms:
            raise TimeoutError(f"Calendar rows not ready: found < {min_rows} within {timeout_ms}ms")
        time.sleep(0.25)

def scrape_ff_core_rows(day: str = "today", tz: str = "Asia/Bangkok", headless: bool = True) -> list:
    """High-fidelity DOM scraper for ForexFactory calendar (lazy Playwright import)."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        print(f"ℹ️ Playwright not available: {e}. Using requests fallback.")
        return []

    tz_cookie_val = tz.replace("/", "%2F")
    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless,
            args=["--no-sandbox","--disable-setuid-sandbox","--disable-dev-shm-usage"])
        context = browser.new_context(user_agent=ua, locale="en-US")
        context.add_cookies([{"name":"fftimezone","value":tz_cookie_val,"domain":".forexfactory.com","path":"/"}])
        page = context.new_page()
        page.goto(f"https://www.forexfactory.com/calendar?day={day}",
                  wait_until="domcontentloaded", timeout=90000)
        page.wait_for_selector("table.calendar__table", state="visible", timeout=60000)
        try: page.wait_for_load_state("load", timeout=30000)
        except Exception: pass
        _wait_for_event_rows(page, min_rows=3, timeout_ms=60000)

        rows = page.query_selector_all("table.calendar__table > tbody > tr")
        events, current_date, last_time = [], None, None

        def _impact_from_cell(cell):
            if not cell: return ""
            icon = cell.query_selector("span[title]")
            if icon:
                t = (icon.get_attribute("title") or "").strip().lower()
                if "high" in t: return "High"
                if "medium" in t: return "Medium"
                if "low" in t: return "Low"
                if "non-economic" in t: return "Non-Economic"
            cls = (cell.get_attribute("class") or "")
            if "ff-impact-red" in cls: return "High"
            if "ff-impact-ora" in cls: return "Medium"
            if "ff-impact-yel" in cls: return "Low"
            if "ff-impact-gra" in cls: return "Non-Economic"
            return ""

        for r in rows:
            rclass = (r.get_attribute("class") or "")
            if "calendar__row--day-breaker" in rclass:
                try: current_date = (r.query_selector("td.calendar__cell") or r).inner_text().strip()
                except Exception: current_date = current_date or ""
                continue
            if "calendar__details" in rclass or "calendar__row" not in rclass: continue

            event_id = r.get_attribute("data-event-id") or ""
            time_cell = r.query_selector("td.calendar__time span")
            time_txt = (time_cell.inner_text().strip() if time_cell else "") if time_cell else ""
            if not time_txt: time_txt = last_time or ""
            else: last_time = time_txt

            ccy_cell = r.query_selector("td.calendar__currency span")
            currency = (ccy_cell.inner_text() or "").strip() if ccy_cell else ""
            impact = _impact_from_cell(r.query_selector("td.calendar__impact"))
            title_cell = r.query_selector("td.calendar__event .calendar__event-title")
            event_title = (title_cell.inner_text().strip() if title_cell else "")

            def _txt(sel):
                cell = r.query_selector(sel)
                return (cell.inner_text().strip() if cell else "")
            actual   = _txt("td.calendar__actual")
            forecast = _txt("td.calendar__forecast")
            previous = _txt("td.calendar__previous span") or ""

            if not time_txt: time_txt = "Tentative"
            if currency and event_title:
                events.append({
                    "event_id": event_id, "date_label": current_date or "", "time": time_txt,
                    "currency": currency, "impact": impact, "event_title": event_title,
                    "actual": actual, "forecast": forecast, "previous": previous
                })

        browser.close()
        return events

def scrape_forex_factory_requests() -> list:
    """Fallback scraper using requests + BeautifulSoup (best-effort)."""
    print("🔄 Trying fallback scraper with requests...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    cookies = {'fftimezone': 'Asia%2FBangkok'}
    try:
        r = requests.get('https://www.forexfactory.com/', headers=headers, cookies=cookies, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'lxml')
        table = soup.select_one("table.calendar__table")
        if not table:
            print("❌ Calendar table not found in requests HTML")
            return []
        out = []
        for row in table.select("tr.calendar__row"):
            cells = [c.get_text(strip=True) for c in row.find_all('td')]
            if len(cells) < 6: continue
            out.append({
                "Time": cells[0] if cells else "", "Currency": (cells[1] if len(cells) > 1 else ""),
                "Impact": (cells[2] if len(cells) > 2 else ""), "Event": (cells[3] if len(cells) > 3 else ""),
                "Actual": (cells[4] if len(cells) > 4 else ""), "Forecast": (cells[5] if len(cells) > 5 else ""),
                "Previous": (cells[6] if len(cells) > 6 else ""),
            })
        print(f"✅ Requests method extracted {len(out)} events!")
        return out
    except Exception as e:
        print(f"❌ Requests method failed: {e}")
        return []

def core_rows_to_basic_rows(core_events: list) -> list:
    mapped = []
    for e in core_events or []:
        mapped.append({
            "Time":     (e.get("time") or "").strip(),
            "Currency": (e.get("currency") or "").strip(),
            "Impact":   (e.get("impact") or "").strip(),
            "Event":    (e.get("event_title") or "").strip(),
            "Actual":   (e.get("actual") or "").strip(),
            "Forecast": (e.get("forecast") or "").strip(),
            "Previous": (e.get("previous") or "").strip(),
            "event_id": (e.get("event_id") or "").strip(),
            "date_label": (e.get("date_label") or "").strip(),
        })
    return mapped

def scrape_news_preferred(day: str = "today", tz: str = "Asia/Bangkok") -> list:
    try:
        core = scrape_ff_core_rows(day=day, tz=tz, headless=True)
        print(f"🔄 Playwright core rows: {len(core)}")
        if core:
            print("Sample:", core[min(2, len(core)-1)])
        mapped = core_rows_to_basic_rows(core)
        if mapped: return mapped
    except Exception as e:
        print(f"❌ Playwright failed: {e}")
    return scrape_forex_factory_requests()

# --- Time parsing (Bangkok) ---

def _parse_ff_time_to_bkk_iso(date_label: str, time_txt: str) -> Optional[str]:
    if not time_txt or time_txt.lower() in ("all day", "tentative"):
        return None
    now_bkk = datetime.now(BKK_TZ)
    try:
        base = datetime.strptime(f"{date_label} {now_bkk.year}", "%a %b %d %Y").replace(tzinfo=BKK_TZ)
    except Exception:
        try:
            base = datetime.strptime(f"{date_label} {now_bkk.year}", "%b %d %Y").replace(tzinfo=BKK_TZ)
        except Exception:
            return None
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*([ap]m)\s*$", time_txt.strip().lower())
    if not m: return None
    hh, mm, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
    if hh == 12: hh = 0
    if ampm == "pm": hh += 12
    return base.replace(hour=hh, minute=mm, second=0, microsecond=0).isoformat()

def extract_high_impact_windows(events_7: List[Dict], window_min: int = 20) -> List[datetime]:
    times = []
    for ev in events_7 or []:
        if (ev.get("Impact") or "").strip().lower() != "high": continue
        iso = _parse_ff_time_to_bkk_iso((ev.get("date_label") or ""), (ev.get("Time") or ""))
        if iso:
            try: times.append(datetime.fromisoformat(iso))
            except Exception: pass
    uniq = {}
    for t in times:
        k = t.replace(second=0, microsecond=0); uniq[k] = k
    return sorted(uniq.values())

def _pair_ccys(pair: str):
    base, quote = pair.replace(" ", "").upper().split("/")
    return base, quote

def filter_news_times_for_pair(_high_times: List[datetime], events_7: List[Dict], pair: str) -> List[datetime]:
    base, quote = _pair_ccys(pair)
    allowed = {base, quote}
    out = []
    for ev in events_7 or []:
        cur = (ev.get("Currency") or "").strip().upper()
        if cur in allowed and (ev.get("Impact","").strip().lower() == "high"):
            iso = _parse_ff_time_to_bkk_iso(ev.get("date_label",""), ev.get("Time",""))
            if iso:
                try: out.append(datetime.fromisoformat(iso))
                except Exception: pass
    seen = set(); unique = []
    for t in sorted(out):
        k = t.replace(second=0, microsecond=0)
        if k not in seen: seen.add(k); unique.append(k)
    return unique


# =====================================================================
#                          MODEL PARSER
# =====================================================================

def _extract_json_block(text: str) -> dict:
    fence = re.search(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence: return json.loads(fence.group(1))
    obj = re.search(r"\{[\s\S]*\}", text)
    if not obj: raise ValueError("Model output missing JSON plan.")
    return json.loads(obj.group(0))

def parse_model_output_to_plan(text: str, pair: str, ctx: dict, cfg: StrategyConfig):
    obj = _extract_json_block(text)
    for k in ["pair","decision","oco_combo","orders","time_in_force","status","confidence","reason"]:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

    combo = tuple(obj["oco_combo"])
    legs: List[OrderLeg] = []
    for o in obj["orders"]:
        legs.append(OrderLeg(order_type=o["type"], entry=float(o["entry"]), sl=float(o["sl"]), tp=float(o["tp"])))

    now_utc = datetime.now(BKK_TZ).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    plan = OCOPlan(
        pair=obj["pair"], combo=combo, legs=tuple(legs),
        time_in_force=obj["time_in_force"], status=obj.get("status","pending"),
        confidence=float(obj["confidence"]), created_at=obj.get("created_at", now_utc),
        updated_at=now_utc, source=obj.get("source", ctx.get("source", {})),
        decision=obj["decision"], reason=obj.get("reason",""),
    )

    spot = ctx["indicators"]["spot"]
    if (not plan.validate_combo(cfg)
        or (not plan.validate_rr(cfg.rr_min, cfg.rr_max))
        or (spot is None)
        or (not plan.sanity_vs_spot(spot, cfg.buffer_pips(pair), pair))):
        plan.decision = "reject"
        plan.reason = "Policy validation failed (combo/RR/spot-sanity)"
    return plan


# =====================================================================
#                     NEWS WINDOW & EOD SUMMARY
# =====================================================================

def within_news_window(now_bkk: datetime, high_impact_times: list, window_min: int = 20) -> bool:
    for t in high_impact_times:
        if abs((now_bkk - t).total_seconds()) <= window_min * 60:
            return True
    return False

def is_eod_summary_time(now_bkk: datetime) -> bool:
    return (now_bkk.hour == 20 and now_bkk.minute >= 59)

def get_day_profile(now_bkk: datetime, cfg: StrategyConfig) -> dict:
    wd = now_bkk.weekday()  # Mon=0
    prof = cfg.weekday_profile.get(wd, cfg.weekday_profile[0]).copy()
    prof["weekday"] = wd
    return prof


# =====================================================================
#                          ORCHESTRATION
# =====================================================================

def run_once(mode: str, dry_run: bool = False):
    cfg = StrategyConfig()
    ledger = WeeklyLedger(cfg.ledger_path, cfg.tz)
    ledger.reset_if_new_week()

    # Scrape calendar (for timing/relevance)
    events_7 = scrape_news_preferred(day="today", tz="Asia/Bangkok")
    high_impact_bkk = extract_high_impact_windows(events_7, window_min=20)
    now_bkk = datetime.now(BKK_TZ)
    today_iso = now_bkk.date().isoformat()

    oa = _openai_client()
    notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    fetcher  = IQDataFetcher()

    # Prepare recent_news payload
    recent_news = []
    for ev in events_7[:60]:
        dt_iso = _parse_ff_time_to_bkk_iso(ev.get("date_label", ""), ev.get("Time", ""))
        recent_news.append({
            "event_id": ev.get("event_id", ""), "title": ev.get("Event", ""), "currency": ev.get("Currency", ""),
            "impact": ev.get("Impact", ""), "actual": ev.get("Actual", ""), "forecast": ev.get("Forecast", ""),
            "previous": ev.get("Previous", ""), "time_iso": dt_iso
        })

    profile = get_day_profile(now_bkk, cfg)

    # Decide pairs to process
    pairs_to_process = list(PAIRS)
    if mode == "auto":
        filtered = []
        for pair in PAIRS:
            pair_times = filter_news_times_for_pair(high_impact_bkk, events_7, pair)
            if within_news_window(now_bkk, pair_times, window_min=20) or is_eod_summary_time(now_bkk):
                filtered.append(pair)
        quota_pairs = [p for p in PAIRS if not ledger.has_order_for_pair_on_date(p, today_iso)]

        if filtered or is_eod_summary_time(now_bkk) or quota_pairs:
            pairs_to_process = list(dict.fromkeys(filtered + quota_pairs))
        else:
            print("⏭️ Not near high-impact news window; not EOD; all pairs already have daily order. Early exit.")
            return

    accepted_msgs, rejected_pairs = [], []

    for pair in pairs_to_process:
        # Indicators (H1 + spot)
        ind  = fetcher.get_indicators(pair, timeframe="H1", lookback=200)
        spot = ind["spot"]

        # ATR guidance (optional but useful)
        try:
            tech = fetcher.get_technical_data(pair)
            d1_atr14 = float(tech.get("d1_atr14", 0.0)) or 0.0
            d1_adr20 = float(tech.get("d1_adr20", 0.0)) or 0.0
        except Exception:
            d1_atr14, d1_adr20 = 0.0, 0.0

        plans_quota = max(1, int(profile.get("plans_per_pair", 1)))
        existing_cnt = ledger.count_orders_for_pair_on_date(pair, today_iso)
        remaining = max(0, plans_quota - existing_cnt)
        if remaining == 0:  # quota already satisfied
            continue

        for variant_idx in range(remaining):
            ctx = {
                "pair": pair,
                "tif_date": (now_bkk.date()).isoformat(),
                "recent_news": recent_news,
                "indicators": {"H1": ind["H1"], "spot": spot},
                "source": {"news_ids": [e.get("event_id") for e in recent_news if e.get("event_id")],
                           "indicators": {"H1": ind["H1"]}},
                "rr_min": cfg.rr_min, "rr_max": cfg.rr_max, "prefer_rr": cfg.prefer_rr,
                "weekday_name": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][profile["weekday"]],
                "tp_atr_low":  profile["tp_atr_mult"][0], "tp_atr_high": profile["tp_atr_mult"][1],
                "weekly_target_pips": cfg.weekly_target_pips, "pairs_per_week": cfg.pairs_per_week,
                "d1_atr14": d1_atr14, "d1_adr20": d1_adr20, "variant_idx": variant_idx,
            }
            user_prompt = format_user_prompt(ctx)
            user_prompt += (
                f"\n\n# Day Profile\n"
                f"Today: {ctx['weekday_name']}\n"
                f"- Target TP width ≈ ATR*{ctx['tp_atr_low']:.2f}–ATR*{ctx['tp_atr_high']:.2f} (guideline; keep RR within [{cfg.rr_min},{cfg.rr_max}])\n"
                f"- Weekly objective: ≥{cfg.weekly_target_pips} net pips across {cfg.pairs_per_week} pairs.\n"
                f"- Variant index: {variant_idx} (avoid overlapping with earlier variants today; small nudge ≈0.25*ATR allowed)\n"
            )

            # Primary model
            try:
                resp = oa.chat.completions.create(
                    model=MODEL_ID,
                    messages=[{"role":"system","content":SYSTEM_PROMPT},
                              {"role":"user","content":user_prompt}],
                    temperature=0.1, max_tokens=900,
                )
                text = resp.choices[0].message.content
            except Exception:
                text = None

            plan: Optional[OCOPlan] = None
            if text:
                try:
                    plan = parse_model_output_to_plan(text, pair, ctx, cfg)
                except Exception:
                    plan = None

            # Optional fallback
            if (plan is None or plan.decision == "reject") and TYPHOON_API_KEY:
                try:
                    typhoon = _typhoon_client()
                    alt_text = typhoon.chat(SYSTEM_PROMPT, user_prompt)
                    plan = parse_model_output_to_plan(alt_text, pair, ctx, cfg)
                except Exception:
                    pass

            if plan is None:
                rejected_pairs.append(pair)
                continue

            plan_obj = plan_to_deterministic_json(plan)
            plan_obj.setdefault("source", {}).setdefault("indicators", {})["spot"] = spot
            plan_obj["variant_idx"] = variant_idx
            plan_obj["day_profile"] = {"weekday": profile["weekday"], "tp_atr_mult": profile["tp_atr_mult"], "plans_quota": plans_quota}

            # Reprice pending (Limit legs only)
            plan_obj = maybe_reprice_pending(plan_obj, spot, cfg)

            # Persist
            ledger.append_or_update(plan_obj)

            # Mark statuses via H1 bars
            h1_bars = fetcher.get_bars(pair, timeframe="H1", lookback=cfg.h1_lookback)
            ledger.mark_statuses(pair, h1_bars, cfg)

            if plan_obj["decision"] == "accept":
                msg = (
                    f"✅ {pair} [{ctx['weekday_name']} v{variant_idx}]\n"
                    f"Combo: {plan_obj['oco_combo']}\n"
                    f"Conf: {plan_obj['confidence']}\n"
                    f"Reason: {plan_obj['reason']}\n"
                    "Orders:\n" +
                    "\n".join([f"- {o['type']} @ {o['entry']} | SL {o['sl']} | TP {o['tp']} | RR {o['rr']}" for o in plan_obj["orders"]])
                )
                accepted_msgs.append(msg)
            else:
                rejected_pairs.append(pair)

    # Notify
    if accepted_msgs:
        notifier.send_message("📌 Accepted OCO Plans\n" + "\n\n".join(accepted_msgs))
    if rejected_pairs:
        notifier.send_message("ℹ️ Rejected pairs this run: " + ", ".join(sorted(set(rejected_pairs))))

    # EOD summary
    if is_eod_summary_time(now_bkk):
        entries = ledger.list_entries()
        lines = [f"{e['pair']} | {e['status']} | {e['oco_combo']} | conf={e['confidence']} | created={e['created_at']}" for e in entries]
        log_today_summary_only("📒 End-of-day summary\n" + "\n".join(lines))


# ---------- CLI ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "force"], default="auto",
                        help="auto: near-news/EOD or daily-quota; force: run immediately")
    args = parser.parse_args()
    run_once(mode=args.mode, dry_run=False)
