# new_forex.py
# Orchestrator: scrape (ForexFactory) -> GPT-5-mini analysis -> parse/validate -> persist -> H1 status check
# -> near-news/EOD gating -> Telegram.
# - Typhoon is used ONLY to format/summarize the accepted GPT plan into a Telegram-friendly message.
# - No failsafe generation. If GPT plan is invalid → reject and report per-pair reason.
# - Impact gating supports both High & Medium via env NEWS_IMPACTS (default: "high,medium").

import os
import re
import json
import time
import argparse
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# scraping deps
from bs4 import BeautifulSoup
import requests

# model/client deps
from openai import OpenAI

# local modules
from get_data import IQDataFetcher
from strategy_engine import (
    StrategyConfig, OCOPlan, PipMath, WeeklyLedger,
    plan_to_deterministic_json, maybe_reprice_pending, OrderLeg
)
from tele_signals import TelegramNotifier, TyphoonForexAnalyzer, log_today_summary_only

# --------------------- Env & constants ---------------------

load_dotenv()

client = OpenAI()
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
MODEL_ID             = os.getenv("MODEL_ID", "gpt-5-mini")
PAIRS                = [p.strip() for p in os.getenv("PAIRS", "EUR/USD,GBP/USD,USD/JPY,EUR/GBP,EUR/CHF").split(",")]
NEWS_WINDOW_MIN      = int(os.getenv("NEWS_WINDOW_MIN", "20"))
BKK_TZ               = timezone(timedelta(hours=7))

# Which impacts to consider for near-news gating ("high,medium" by default)
IMPACT_SET = set(
    s.strip().lower()
    for s in os.getenv("NEWS_IMPACTS", "high,medium").split(",")
    if s.strip()
)

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
if not TELEGRAM_CHAT_ID:
    raise RuntimeError("Missing TELEGRAM_CHAT_ID (or CHAT_ID)")

# --------------------- Prompt loading ---------------------

def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

SYSTEM_PROMPT = _load_text("prompts/system_gpt5.txt")
USER_TEMPLATE = _load_text("prompts/user_template.txt")

def format_user_prompt(ctx: dict) -> str:
    """
    Compose deterministic, data-first prompt for GPT-5-mini.
    ctx keys expected:
      - pair
      - tif_date (YYYY-MM-DD)
      - recent_news (list[dict] of {event_id,title,impact,actual,forecast,previous,currency,time_iso})
      - indicators: {"H1": {"ema_fast":.., "ema_slow":.., "rsi":..}, "spot": float}
    """
    return USER_TEMPLATE.format(**ctx)

# --------------------- OpenAI & helper ---------------------

def _openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)

# =====================================================================
#                           SCRAPER SECTION
# =====================================================================

def _wait_for_event_rows(page, min_rows: int = 5, timeout_ms: int = 60000):
    start = time.time()
    while True:
        try:
            cnt = page.locator("table.calendar__table tr.calendar__row[data-event-id]").count()
            if cnt >= min_rows:
                return
        except Exception:
            pass
        if (time.time() - start) * 1000 > timeout_ms:
            raise TimeoutError(f"Calendar rows not ready: found < {min_rows} within {timeout_ms}ms")
        time.sleep(0.25)

def scrape_ff_core_rows(day: str = "today", tz: str = "Asia/Bangkok", headless: bool = True) -> list:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        print(f"ℹ️ Playwright not available: {e}. Using requests fallback.")
        return []

    tz_cookie_val = tz.replace("/", "%2F")
    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
          "AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/122.0.0.0 Safari/537.36")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
        )
        context = browser.new_context(user_agent=ua, locale="en-US")
        context.add_cookies([{
            "name": "fftimezone",
            "value": tz_cookie_val,
            "domain": ".forexfactory.com",
            "path": "/"
        }])
        page = context.new_page()
        page.goto(f"https://www.forexfactory.com/calendar?day={day}",
                  wait_until="domcontentloaded", timeout=90000)
        page.wait_for_selector("table.calendar__table", state="visible", timeout=60000)
        try:
            page.wait_for_load_state("load", timeout=30000)
        except Exception:
            pass
        _wait_for_event_rows(page, min_rows=3, timeout_ms=60000)

        rows = page.query_selector_all("table.calendar__table > tbody > tr")
        events, current_date, last_time = [], None, None

        def _impact_from_cell(cell):
            if not cell:
                return ""
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
                try:
                    current_date = (r.query_selector("td.calendar__cell") or r).inner_text().strip()
                except Exception:
                    current_date = current_date or ""
                continue
            if "calendar__details" in rclass or "calendar__row" not in rclass:
                continue

            event_id = r.get_attribute("data-event-id") or ""
            time_cell = r.query_selector("td.calendar__time span")
            time_txt = (time_cell.inner_text().strip() if time_cell else "") if time_cell else ""
            if not time_txt:
                time_txt = last_time or ""
            else:
                last_time = time_txt

            ccy_cell = r.query_selector("td.calendar__currency span")
            currency = (ccy_cell.inner_text() or "").strip() if ccy_cell else ""
            impact = _impact_from_cell(r.query_selector("td.calendar__impact"))
            title_cell = r.query_selector("td.calendar__event .calendar__event-title")
            event_title = (title_cell.inner_text().strip() if title_cell else "")

            def _txt(sel):
                cell = r.query_selector(sel)
                return (cell.inner_text().strip() if cell else "")
            actual = _txt("td.calendar__actual")
            forecast = _txt("td.calendar__forecast")

            previous = ""
            prev_cell = r.query_selector("td.calendar__previous span")
            if prev_cell:
                previous = (prev_cell.inner_text() or "").strip()

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
        rows = table.select("tr.calendar__row")
        out = []
        for row in rows:
            cells = [c.get_text(strip=True) for c in row.find_all('td')]
            if len(cells) < 6:
                continue
            out.append({
                "Time": cells[0] if cells else "",
                "Currency": (cells[1] if len(cells) > 1 else ""),
                "Impact": (cells[2] if len(cells) > 2 else ""),
                "Event": (cells[3] if len(cells) > 3 else ""),
                "Actual": (cells[4] if len(cells) > 4 else ""),
                "Forecast": (cells[5] if len(cells) > 5 else ""),
                "Previous": (cells[6] if len(cells) > 6 else ""),
                "event_id": "",
                "date_label": "",
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
        if mapped:
            return mapped
    except Exception as e:
        print(f"❌ Playwright failed: {e}")
    return scrape_forex_factory_requests()

# --- Impact/time helpers ---

def _impact_allowed(impact_str: str) -> bool:
    return (impact_str or "").strip().lower() in IMPACT_SET

def _parse_ff_time_to_bkk_iso(date_label: str, time_txt: str) -> Optional[str]:
    if not time_txt or time_txt.lower() in ("all day", "tentative"):
        return None
    now_bkk = datetime.now(BKK_TZ)
    try:
        base = datetime.strptime(f"{date_label} {now_bkk.year}", "%a %b %d %Y")
        base = base.replace(tzinfo=BKK_TZ)
    except Exception:
        try:
            base = datetime.strptime(f"{date_label} {now_bkk.year}", "%b %d %Y").replace(tzinfo=BKK_TZ)
        except Exception:
            return None

    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*([ap]m)\s*$", time_txt.strip().lower())
    if not m:
        return None
    hh = int(m.group(1)); mm = int(m.group(2)); ampm = m.group(3)
    if hh == 12: hh = 0
    if ampm == "pm": hh += 12
    dt = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return dt.isoformat()

def filter_news_times_for_pair(events_7: List[Dict], pair: str) -> List[datetime]:
    """Return High/Medium-impact times (per env) for the base/quote currencies of the pair."""
    base, quote = pair.replace(" ", "").upper().split("/")
    allowed_ccy = {base, quote}
    out = []
    for ev in events_7 or []:
        cur = (ev.get("Currency") or "").strip().upper()
        impact = (ev.get("Impact") or "").strip()
        if cur in allowed_ccy and _impact_allowed(impact):
            iso = _parse_ff_time_to_bkk_iso(ev.get("date_label",""), ev.get("Time",""))
            if iso:
                try:
                    out.append(datetime.fromisoformat(iso))
                except Exception:
                    pass
    # Dedup by minute
    seen = set(); unique = []
    for t in sorted(out):
        k = t.replace(second=0, microsecond=0)
        if k not in seen:
            seen.add(k); unique.append(k)
    return unique

def within_news_window(now_bkk: datetime, times: List[datetime], window_min: int) -> bool:
    for t in times:
        if abs((now_bkk - t).total_seconds()) <= window_min * 60:
            return True
    return False

def is_eod_summary_time(now_bkk: datetime) -> bool:
    return (now_bkk.hour == 21 and now_bkk.minute >= 30)

# =====================================================================
#                    MODEL OUTPUT PARSER SECTION
# =====================================================================

_ORDER_ALIASES = {
    "buystop": "Buy Stop", "buy stop": "Buy Stop", "buy-stop": "Buy Stop", "buy_stop": "Buy Stop",
    "sellstop": "Sell Stop", "sell stop": "Sell Stop", "sell-stop": "Sell Stop", "sell_stop": "Sell Stop",
    "buylimit": "Buy Limit", "buy limit": "Buy Limit", "buy-limit": "Buy Limit", "buy_limit": "Buy Limit",
    "selllimit": "Sell Limit", "sell limit": "Sell Limit", "sell-limit": "Sell Limit", "sell_limit": "Sell Limit",
}

def _norm_order_type(s: str) -> str:
    if not isinstance(s, str):
        return ""
    key = s.strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    key = key.replace(" ", "")
    return _ORDER_ALIASES.get(key, "").strip()

def _extract_json_block(text: str) -> dict:
    """
    Extracts a JSON object from a string, supporting markdown code fences.
    Now raises ValueError with more context on JSON decoding errors.
    """
    fence = re.search(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError as e:
            # Raise a more informative error
            raise ValueError(f"Invalid JSON in fenced block: {e}")

    # Fallback to the first complete JSON object found
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if not obj_match:
        raise ValueError("Model output missing JSON plan.")
    try:
        return json.loads(obj_match.group(0))
    except json.JSONDecodeError as e:
        # Raise a more informative error
        raise ValueError(f"Invalid JSON in raw text: {e}")

def parse_model_output_to_plan(text: str, pair: str, ctx: dict, cfg: StrategyConfig) -> OCOPlan:
    """
    Parse GPT text → plan; normalize order types; enforce validations; return OCOPlan.
    Rejection reasons are explicit and short for Telegram.
    Now raises more specific ValueErrors.
    """
    try:
        obj = _extract_json_block(text)
    except ValueError as e:
        # Re-raise the error so the caller can log it
        print(f"DEBUG: JSON extraction failed for pair {pair}: {e}")
        raise

    required = ["pair", "decision", "oco_combo", "orders", "time_in_force",
                "status", "confidence", "reason"]
    for k in required:
        if k not in obj:
            # More specific error for missing keys
            raise ValueError(f"Missing required key in JSON: '{k}'")

    # Normalize combo
    raw_combo = obj["oco_combo"]; combo_norm, unknown = [], []
    for t in raw_combo:
        nt = _norm_order_type(t)
        if not nt: unknown.append(str(t))
        combo_norm.append(nt or str(t).strip())
    if unknown:
        now_utc = datetime.now(BKK_TZ).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return OCOPlan(
            pair=obj["pair"], combo=tuple(combo_norm), legs=tuple(),
            time_in_force=obj["time_in_force"], status="pending",
            confidence=float(obj.get("confidence", 0.0)),
            created_at=obj.get("created_at", now_utc), updated_at=now_utc,
            source=obj.get("source", ctx.get("source", {})),
            decision="reject", reason=f"Unknown order type in combo: {unknown}",
        )
    combo = tuple(combo_norm)

    # Normalize legs
    legs, unknown_legs = [], []
    for o in obj["orders"]:
        t_raw = o["type"]; t_norm = _norm_order_type(t_raw)
        if not t_norm:
            unknown_legs.append(str(t_raw))
            t_norm = str(t_raw).strip()
        legs.append(
            OrderLeg(order_type=t_norm, entry=float(o["entry"]),
                     sl=float(o["sl"]), tp=float(o["tp"]))
        )

    now_utc = datetime.now(BKK_TZ).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    plan = OCOPlan(
        pair=obj["pair"], combo=combo, legs=tuple(legs),
        time_in_force=obj["time_in_force"], status=obj.get("status", "pending"),
        confidence=float(obj["confidence"]),
        created_at=obj.get("created_at", now_utc), updated_at=now_utc,
        source=obj.get("source", ctx.get("source", {})),
        decision=obj["decision"], reason=obj.get("reason", ""),
    )

    if unknown_legs:
        plan.decision = "reject"; plan.reason = f"Unknown order type in legs: {unknown_legs}"
        return plan

    spot = ctx["indicators"]["spot"]
    if spot is None:
        plan.decision = "reject"; plan.reason = "No spot price in context"
        return plan

    # combo policy
    if not plan.validate_combo(cfg):
        plan.decision = "reject"; plan.reason = f"Invalid combo {list(combo)}"
        return plan

    # RR policy (AND across legs, per current policy)
    if not plan.validate_rr(cfg.rr_min, cfg.rr_max):
        # Prepare compact RR list for reason
        rr_list = []
        for leg in plan.legs:
            risk = abs(leg.entry - leg.sl); reward = abs(leg.tp - leg.entry)
            rr_list.append(round((reward / risk), 2) if risk > 0 else float("inf"))
        plan.decision = "reject"; plan.reason = f"RR out of range {rr_list} not in [{cfg.rr_min},{cfg.rr_max}]"
        return plan

    # spot sanity
    min_buf = cfg.buffer_pips(pair)
    if not plan.sanity_vs_spot(spot, min_buf, pair):
        plan.decision = "reject"; plan.reason = f"Entry not sane vs spot (min buffer {min_buf} pips)"
        return plan

    return plan

# =====================================================================
#                         ORCHESTRATION SECTION
# =====================================================================

def run_once(mode: str, dry_run: bool = False):
    """
    mode:
      - "auto": run when near (High/Medium) news, or at EOD summary,
                or if any pair has no order today (quota).
      - "force": run immediately.
    """
    cfg = StrategyConfig()
    ledger = WeeklyLedger(cfg.ledger_path, cfg.tz)
    ledger.reset_if_new_week()

    # 1) News scrape → near-news times
    events_7 = scrape_news_preferred(day="today", tz="Asia/Bangkok")
    now_bkk = datetime.now(BKK_TZ)
    today_iso = now_bkk.date().isoformat()

    near_pairs = []
    for pair in PAIRS:
        p_times = filter_news_times_for_pair(events_7, pair)
        if within_news_window(now_bkk, p_times, NEWS_WINDOW_MIN):
            near_pairs.append(pair)

    # daily quota: process pairs with zero orders today (if ledger supports)
    quota_pairs = []
    if hasattr(ledger, "has_order_for_pair_on_date"):
        for pair in PAIRS:
            if not ledger.has_order_for_pair_on_date(pair, today_iso):
                quota_pairs.append(pair)
    else:
        if is_eod_summary_time(now_bkk):
            quota_pairs = list(PAIRS)

    if mode == "auto":
        if not (near_pairs or is_eod_summary_time(now_bkk) or quota_pairs):
            print("⏭️ Not near (High/Medium) window, not EOD, and no quota pairs. Early exit.")
            return

    # 2) Decide target pairs
    pairs_to_process = list(PAIRS) if mode == "force" else list(dict.fromkeys(near_pairs + quota_pairs))
    if not pairs_to_process:
        print("ℹ️ No pairs to process in this run.")
        return

    fetcher  = IQDataFetcher()
    notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    oa       = _openai_client()
    # Typhoon is formatter only (optional):
    typhoon  = TyphoonForexAnalyzer(os.getenv("TYPHOON_API_KEY", ""))
    typhoon_ready = False
    if typhoon:
        is_cfg = getattr(typhoon, "is_configured", None)
        if callable(is_cfg):
            typhoon_ready = bool(is_cfg())

    # 3) Build a compact news payload for context (cap size)
    recent_news = []
    for ev in events_7[:60]:
        dt_iso = _parse_ff_time_to_bkk_iso(ev.get("date_label", ""), ev.get("Time", ""))
        recent_news.append({
            "event_id": ev.get("event_id", ""),
            "title": ev.get("Event", ""),
            "currency": ev.get("Currency", ""),
            "impact": ev.get("Impact", ""),
            "actual": ev.get("Actual", ""),
            "forecast": ev.get("Forecast", ""),
            "previous": ev.get("Previous", ""),
            "time_iso": dt_iso
        })

    accepted_msgs: List[str] = []
    rejected_pairs: List[Tuple[str, str]] = []  # (pair, reason)

    # 4) Per-pair loop
    for pair in pairs_to_process:
        print(f"\n--- Processing: {pair} ---")
        # Indicators (H1) & spot
        try:
            ind  = fetcher.get_indicators(pair, timeframe="H1", lookback=200)
            spot = float(ind["spot"])
        except Exception as e:
            print(f"❌ Could not fetch indicators for {pair}: {e}")
            rejected_pairs.append((pair, "indicator_fetch_failed"))
            continue

        ctx = {
            "pair": pair,
            "tif_date": (now_bkk.date()).isoformat(),
            "recent_news": recent_news,
            "indicators": {"H1": ind["H1"], "spot": spot},
            "source": {"news_ids": [e.get("event_id") for e in recent_news if e.get("event_id")],
                       "indicators": {"H1": ind["H1"]}},
            "rr_min": cfg.rr_min,
            "rr_max": cfg.rr_max,
            "prefer_rr": cfg.prefer_rr,
            # --- FINAL FIX: Add the specific buffer pips for this pair directly into the prompt context ---
            "min_buffer_pips": cfg.buffer_pips(pair)
        }

        # 4.1 GPT → plan text
        model_text = None
        try:
            user_prompt = format_user_prompt(ctx)
            resp = client.responses.create(
                model="gpt-5-mini",
                instructions=SYSTEM_PROMPT,   # keep static for prompt caching
                input=user_prompt,
                text={"verbosity": "medium"},
                reasoning={"effort": "minimal"},
                max_output_tokens=1200
            )
            model_text = resp.output_text
            # --- DEBUG: Print raw model output ---
            print(f"--- Raw Model Output for {pair} ---\n{model_text}\n------------------------------------")
        except Exception as e:
            model_text = None
            print(f"❌ OpenAI API call failed for {pair}: {e}")

        # 4.2 Parse & policy checks
        plan: Optional[OCOPlan] = None
        if model_text:
            try:
                plan = parse_model_output_to_plan(model_text, pair, ctx, cfg)
            except Exception as e:
                # --- DEBUG: Catch and print specific parsing error ---
                print(f"❌ Error parsing model output for {pair}: {e}")
                plan = None # Ensure plan is None on failure

        # 4.3 If no valid plan → reject with reason
        if plan is None or plan.decision == "reject":
            reason = "model_parse_failed"
            if plan is not None and getattr(plan, "reason", ""):
                reason = plan.reason
            # --- DEBUG: More specific reason if parsing itself failed ---
            elif plan is None:
                reason = "model_parse_exception"
            rejected_pairs.append((pair, reason))
            print(f"➡️ Rejected {pair} due to: {reason}")
            continue

        # 4.4 Deterministic JSON
        plan_obj = plan_to_deterministic_json(plan)
        plan_obj.setdefault("source", {}).setdefault("indicators", {})["spot"] = spot

        # 4.5 Optional: reprice pending limit legs closer (policy-safe)
        plan_obj = maybe_reprice_pending(plan_obj, spot, cfg)

        # 4.6 Persist
        ledger.append_or_update(plan_obj)

        # 4.7 Status mark using H1 bars
        h1_bars = fetcher.get_bars(pair, timeframe="H1", lookback=cfg.h1_lookback)
        ledger.mark_statuses(pair, h1_bars, cfg)

        # 4.8 Notify → Typhoon summary (formatter only) or fallback local summary
        try:
            summary_msg = ""
            if typhoon_ready:
                summary_msg = typhoon.summarize_plan(plan_obj, market_meta={"spot": spot, "tf": "H1"})
            else:
                # local minimal summary (guaranteed no LLM drift)
                summary_msg = (
                    f"✅ {plan_obj['pair']} | {plan_obj['decision'].upper()}\n"
                    f"Combo: {', '.join(plan_obj['oco_combo'])}\n"
                    f"Conf: {plan_obj['confidence']:.2f} | Reason: {plan_obj['reason']}\n"
                    "Orders:\n" + "\n".join(
                        f"- {o['type']} @ {o['entry']} | SL {o['sl']} | TP {o['tp']} | RR {o['rr']}"
                        for o in plan_obj["orders"]
                    )
                )

            if not dry_run:
                notifier.send_message(summary_msg)
            print(f"✅ Notified acceptance for {pair}")
            accepted_msgs.append(pair)
        except Exception as e:
            print(f"❌ Notification failed for {pair}: {e}")
            # If sending failed, at least note the acceptance in a compact line
            accepted_msgs.append(f"{pair} (send_failed)")


    # 5) Post-run notifications
    if rejected_pairs:
        lines = ["ℹ️ Rejected this run:"]
        for p, r in rejected_pairs:
            lines.append(f"- {p} → {r}")
        if not dry_run:
            notifier.send_message("\n".join(lines))

    # 6) EOD summary (once/day)
    if is_eod_summary_time(now_bkk):
        entries = ledger.list_entries()
        lines = [
            f"{e['pair']} | {e['status']} | {e['oco_combo']} | conf={e['confidence']} | created={e['created_at']}"
            for e in entries
        ]
        log_today_summary_only("📒 End-of-day summary\n" + "\n".join(lines))

    fetcher.close_connection() # ensure connection is closed
    print("\n--- Run finished ---")


# --------------------- CLI ---------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "force"], default="auto",
                        help="auto: near (High/Medium) news / EOD / daily-quota; force: run immediately")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run all logic but do not send Telegram messages.")
    args = parser.parse_args()
    run_once(mode=args.mode, dry_run=args.dry_run)