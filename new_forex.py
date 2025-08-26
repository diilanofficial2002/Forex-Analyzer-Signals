# new_forex.py
# Orchestrator: scrape -> prompts -> GPT-5/Typhoon -> parse -> validate -> persist -> daily review -> Telegram.
# Bangkok timezone. Sends full details ONLY for accepted pairs; rejected pairs summarized once per run.

import os, time, re, json, argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from dotenv import load_dotenv

from bs4 import BeautifulSoup
import requests
from playwright.sync_api import sync_playwright
from openai import OpenAI

from get_data import IQDataFetcher
from strategy_engine import (
    StrategyConfig, WeeklyLedger, ReviewEngine, OrderPlanner, TyphoonParser
)
from tele_signals import TelegramNotifier, TyphoonForexAnalyzer, log_today

load_dotenv()

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
TYPHOON_API_KEY    = os.getenv("TYPHOON_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("CHAT_ID")
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/GBP", "EUR/CHF"]


# --------------------- Playwright helpers (from your spec) ---------------------
def _wait_for_event_rows(page, min_rows: int = 5, timeout_ms: int = 60000):
    """Poll until at least `min_rows` event rows (having data-event-id) are attached."""
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
    """High-fidelity DOM scraper for ForexFactory calendar."""
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
                cell = r.query_selector(sel); 
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


def scrape_forex_factory_requests():
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
        rows = table.select("tr.calendar__row")
        out = []
        for row in rows:
            cells = [c.get_text(strip=True) for c in row.find_all('td')]
            if len(cells) < 6: 
                continue
            # map best-effort (Time, Currency, Impact, Event, Actual, Forecast, Previous)
            out.append({
                "Time": cells[0] if cells else "",
                "Currency": (cells[1] if len(cells) > 1 else ""),
                "Impact": (cells[2] if len(cells) > 2 else ""),
                "Event": (cells[3] if len(cells) > 3 else ""),
                "Actual": (cells[4] if len(cells) > 4 else ""),
                "Forecast": (cells[5] if len(cells) > 5 else ""),
                "Previous": (cells[6] if len(cells) > 6 else ""),
            })
        print(f"✅ Requests method extracted {len(out)} events!")
        return out
    except Exception as e:
        print(f"❌ Requests method failed: {e}")
        return []


def core_rows_to_basic_rows(core_events: list) -> list:
    """Map DOM rows -> 7 columns."""
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
        })
    return mapped


def _compact_calendar_lines(events_7col: list, limit: int = 60) -> str:
    """Pretty print a few lines of calendar for logs."""
    lines = []
    for ev in events_7col or []:
        t   = ev.get("Time","")
        cur = ev.get("Currency","")
        imp = ev.get("Impact","")
        name= ev.get("Event","")
        act = ev.get("Actual","")
        fcs = ev.get("Forecast","")
        prv = ev.get("Previous","")
        if cur and name:
            lines.append(f"{t} {cur} [{imp}] {name} A:{act} F:{fcs} P:{prv}")
    return "\n".join(lines[:limit]) if lines else "No events"


def scrape_news_preferred(day: str = "today", tz: str = "Asia/Bangkok") -> list:
    """Try DOM -> fallback requests."""
    core = scrape_ff_core_rows(day=day, tz=tz, headless=True)
    print(f"🔄 Got {len(core)} core rows from Playwright.")
    if core:
        print("Core row sample:", core[min(2, len(core)-1)])
    mapped = core_rows_to_basic_rows(core)
    if mapped:
        return mapped
    return scrape_forex_factory_requests()


# --------------------- GPT-5 Responses API ---------------------
def call_gpt5(system_prompt: str, user_prompt: str, model: str = "gpt-5-mini") -> str:
    """
    Responses API call without temperature (some GPT-5 variants reject it).
    Fallback on token key differences.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    def _extract_text(resp):
        txt = getattr(resp, "output_text", None)
        if txt:
            return txt.strip()
        parts = []
        for item in getattr(resp, "output", []) or []:
            if isinstance(item, dict) and item.get("type") == "message":
                for c in item.get("content", []) or []:
                    t = c.get("text") or c.get("content")
                    if t: parts.append(t)
        return "\n".join(parts).strip()

    try:
        resp = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            text={"verbosity": "medium"},
            reasoning={"effort": "minimal"},
            max_output_tokens=1200,
        )
        return _extract_text(resp)
    except Exception as e1:
        msg = str(e1)
        if "max_output_tokens" in msg or "max_tokens" in msg:
            resp = OpenAI(api_key=OPENAI_API_KEY).responses.create(
                model=model,
                instructions=system_prompt,
                input=user_prompt,
                max_completion_tokens=1200
            )
            return _extract_text(resp)
        raise


# --------------------- Prompts ---------------------
def _load_text_or_default(path: str, default_text: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default_text

def build_system_prompt_gpt5() -> str:
    default = (
        "You are an elite FX strategist generating ONLY pending-order OCO plans.\n"
        "Allowed combos: BUY_STOP+SELL_STOP | BUY_LIMIT+BUY_STOP | SELL_LIMIT+SELL_STOP\n"
        "Risk:Reward within [1.5, 2.0]; prefer ≥1.8.\n"
        "Use given TIF; keep entries realistic vs spot & spread. Weekly target=1000 pips across 5 pairs.\n"
        "Output format strictly:\n"
        "**COMBO:** <combo>\n**TIF:** <YYYY-MM-DD>\n"
        "**PRIMARY ORDER:** <LONG|SHORT> — <Buy Stop|Sell Stop|Buy Limit|Sell Limit> @ <entry>\n"
        "**TP:** <price>\n**SL:** <price>\n"
        "**SECONDARY (OCO):** <LONG|SHORT> — <Buy Stop|Sell Stop|Buy Limit|Sell Limit> @ <entry>\n"
        "**TP:** <price>\n**SL:** <price>\n"
        "- 3–6 short bullets of reasons."
    )
    return _load_text_or_default("prompts/system_gpt5.txt", default)

def build_system_prompt_typhoon() -> str:
    default = (
        "You are a pending-order FX planner. Follow EXACT schema. If insufficient data, reply 'INSUFFICIENT DATA'.\n"
        "Combos: BUY_STOP+SELL_STOP | BUY_LIMIT+BUY_STOP | SELL_LIMIT+SELL_STOP\n"
        "RR in [1.5, 2.0] (prefer ≥1.8). Respect TIF. Ensure stop/limit semantics vs spot.\n"
        "**COMBO:** <combo>\n**TIF:** <YYYY-MM-DD>\n"
        "**PRIMARY ORDER:** <LONG|SHORT> — <Buy Stop|Sell Stop|Buy Limit|Sell Limit> @ <entry>\n"
        "**TP:** <price>\n**SL:** <price>\n"
        "**SECONDARY (OCO):** <LONG|SHORT> — <Buy Stop|Sell Stop|Buy Limit|Sell Limit> @ <entry>\n"
        "**TP:** <price>\n**SL:** <price>\n"
        "- Reason 1\n- Reason 2\n- Reason 3"
    )
    return _load_text_or_default("prompts/system_typhoon.txt", default)

def build_user_prompt(ctx: dict) -> str:
    default = (
        "PAIR: {pair}\n\n"
        "# Market Snapshot (Bangkok)\n"
        "Spot Bid: {spot_bid}\nSpot Ask: {spot_ask}\nSpot Mid: {spot_mid}\nSpread (pips): {spread_pips}\n\n"
        "Prev D1 High/Low/Close: {prev_day_high} / {prev_day_low} / {prev_day_close}\n"
        "D1 ATR(14): {d1_atr14}\nD1 ADR(20): {d1_adr20}\n\n"
        "Latest Candles (Bangkok timestamps)\nH4: {latest_h4_ict}\nH1: {latest_h1_ict}\nM15: {latest_m15_ict}\nD1: {latest_d1_ict}\n\n"
        "# Constraints\nAllowed OCO Combos: {allowed_combos}\nR:R between {rr_min} and {rr_max}\nTIF (expiry): {tif_date}\n\n"
        "# Task\nPropose one OCO plan using ONLY an allowed combo. Respect pending-order semantics:\n"
        "- Buy Limit < spot; Sell Limit > spot; Buy Stop > spot; Sell Stop < spot.\n"
        "TP/SL must be realistic vs ATR/ADR and spread. Return strictly in the specified format."
    )
    try:
        with open("prompts/user_template.txt", "r", encoding="utf-8") as f:
            tmpl = f.read()
    except Exception:
        tmpl = default
    return tmpl.format(**ctx)


# --------------------- Utils & de-dup ---------------------
def _ledger_has_duplicate(ledger, pair: str, entry: float, sl: float, tp: float, tif_date: str) -> bool:
    """Return True if an equivalent open order exists (PENDING/TRIGGERED)."""
    for _, row in ledger.iter_indices(pair):
        try:
            if row.get("status") in ("PENDING", "TRIGGERED") \
               and float(row.get("entry")) == float(entry) \
               and float(row.get("sl")) == float(sl) \
               and float(row.get("tp")) == float(tp) \
               and row.get("tif_date") == tif_date:
                return True
        except Exception:
            continue
    return False


# --------------------- Core run ---------------------
def run_once():
    print("🚀 Starting Pending-Order Strategy Bot...")
    cfg = StrategyConfig()
    ledger = WeeklyLedger(cfg)
    os.makedirs(cfg.state_dir, exist_ok=True)
    if not os.path.exists(cfg.weekly_state_file):
        ledger.save()

    notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    typhoon  = TyphoonForexAnalyzer(TYPHOON_API_KEY) if TYPHOON_API_KEY else None
    fetcher  = IQDataFetcher()

    # 1) News scrape (non-blocking)
    events_7 = scrape_news_preferred(day="today", tz="Asia/Bangkok")
    print(f"📰 Extracted {len(events_7)} events")
    if events_7:
        print(_compact_calendar_lines(events_7, limit=20))
        log_today(f"📰 {len(events_7)} events\n" + _compact_calendar_lines(events_7, limit=30))
    else:
        log_today("ℹ️ No events scraped today (continuing analysis).")

    # 2) Prompts
    system_gpt5 = build_system_prompt_gpt5()
    system_tfn  = build_system_prompt_typhoon()
    weekday_bkk = (datetime.utcnow() + timedelta(hours=7)).weekday()
    tif_date     = OrderPlanner.tif_for_today(weekday_bkk)
    current_date = (datetime.utcnow() + timedelta(hours=7)).strftime("%Y-%m-%d")
    weekly_before = ledger.realized()

    accepted_pairs, rejected_pairs = [], []

    # 3) Per-pair loop
    for pair in PAIRS:
        try:
            tech = fetcher.get_technical_data(pair)
        except Exception as e:
            # Treat as rejected (no details)
            rejected_pairs.append(pair)
            continue

        ctx = {
            "pair": pair,
            "spot_bid": tech["spot_bid"],
            "spot_ask": tech["spot_ask"],
            "spot_mid": tech["spot_mid"],
            "spread_pips": tech["spread_pips"],
            "prev_day_high": tech["prev_day_high"],
            "prev_day_low": tech["prev_day_low"],
            "prev_day_close": tech["prev_day_close"],
            "d1_atr14": tech["d1_atr14"],
            "d1_adr20": tech["d1_adr20"],
            "latest_h4_ict": tech["latest_h4_ict"],
            "latest_h1_ict": tech["latest_h1_ict"],
            "latest_m15_ict": tech["latest_m15_ict"],
            "latest_d1_ict": tech["latest_d1_ict"],
            "tif_date": tif_date,
            "rr_min": cfg.rr_min,
            "rr_max": cfg.rr_max,
            "allowed_combos": "BUY_STOP+SELL_STOP | BUY_LIMIT+BUY_STOP | SELL_LIMIT+SELL_STOP",
        }
        user_prompt = build_user_prompt(ctx)

        summary = None
        try:
            summary = call_gpt5(system_gpt5, user_prompt, model="gpt-5-mini")
        except Exception as e:
            summary = None
        if not summary and typhoon:
            try:
                summary = typhoon.analyze(system_tfn, user_prompt)
            except Exception:
                summary = None

        if not summary:
            rejected_pairs.append(pair)
            continue

        plan = TyphoonParser.parse(summary, pair=pair, current_date=current_date, tif_date=tif_date)
        is_jpy = pair.endswith("/JPY")
        buf = OrderPlanner.buffer_pips(tech["d1_atr14"], float(tech["spread_pips"]), is_jpy)
        spot = float(tech["spot_mid"])

        ok_parse  = bool(plan.order_a and plan.order_b)
        ok_combo  = plan.validate_combo() if ok_parse else False
        ok_rr     = plan.validate_rr(cfg.rr_min, cfg.rr_max) if ok_parse else False
        ok_spot   = plan.sanity_vs_spot(spot, buf, pair) if ok_parse else False

        if ok_parse and ok_combo and ok_rr and ok_spot:
            # Prevent duplicates across hourly runs
            dup_a = _ledger_has_duplicate(ledger, pair, plan.order_a.entry, plan.order_a.sl, plan.order_a.tp, plan.order_a.tif_date) if plan.order_a else False
            dup_b = _ledger_has_duplicate(ledger, pair, plan.order_b.entry, plan.order_b.sl, plan.order_b.tp, plan.order_b.tif_date) if plan.order_b else False
            if not (dup_a or dup_b):
                ledger.add_orders(plan.as_list())
                # Send full details for accepted only
                notifier.send_message(f"✅ {pair}: Plan accepted.\n\n{summary}")
            else:
                notifier.send_message(f"ℹ️ {pair}: Already tracked (skipped duplicate).")
            accepted_pairs.append(pair)
        else:
            rejected_pairs.append(pair)

    # 4) Daily review (Tue–Fri)
    if weekday_bkk in (1, 2, 3, 4):
        total_active = ledger.count_active()
        if total_active == 0:
            notifier.send_message("ℹ️ ไม่มีออร์เดอร์ค้างสำหรับรีวิววันนี้ (ledger ว่าง)")
        else:
            for pair in PAIRS:
                try:
                    tech = fetcher.get_technical_data(pair)
                    d1_open = tech["prev_day_close"]
                    d1_high = tech["prev_day_high"]
                    d1_low  = tech["prev_day_low"]
                    changed = ReviewEngine(cfg, ledger).apply_d1_bar(pair, d1_open, d1_high, d1_low)
                    if changed:
                        notifier.send_message(f"🔁 {pair}: อัปเดตสถานะแล้ว | Weekly realized: {ledger.realized():.1f} pips")
                except Exception as e:
                    notifier.send_message(f"⚠️ {pair}: Review fetch failed → {e}")

    # 5) Summaries (no rejected details)
    if rejected_pairs:
        notifier.send_message("⏸️ Rejected this run: " + ", ".join(sorted(set(rejected_pairs))))
    if accepted_pairs:
        notifier.send_message("✅ Accepted this run: " + ", ".join(sorted(set(accepted_pairs))))

    weekly_after = ledger.realized()
    notifier.send_message(f"📊 Weekly P/L updated: {weekly_before:.1f} → {weekly_after:.1f} pips")

    # Where is the ledger?
    try:
        path = cfg.weekly_state_file
        exists = os.path.exists(path)
        cnt = 0
        if exists:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f); cnt = len(data.get("orders", []))
        notifier.send_message(f"🗂 Ledger: {os.path.abspath(path)} | orders={cnt}")
    except Exception:
        pass

if __name__ == "__main__":
    run_once()