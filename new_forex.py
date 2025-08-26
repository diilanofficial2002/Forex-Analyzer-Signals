# new_forex.py
# Main orchestrator: scrape → build prompts → call GPT-5 (fallback Typhoon)
# → parse plan → validate (RR/combo/sanity) → persist → review Tue–Fri → notify.

import os, time, json, re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import requests

from openai import OpenAI

from get_data import IQDataFetcher
from strategy_engine import (
    StrategyConfig, WeeklyLedger, ReviewEngine, OrderPlanner, TyphoonParser
)
from tele_signals import TelegramNotifier, TyphoonForexAnalyzer, log_today

load_dotenv()

# ---------- ENV / Globals ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID")

PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/GBP", "EUR/CHF"]  # fixed per your plan

# ---------- ForexFactory scraping (keep "as before": Playwright → DOM) ----------

def scrape_ff_core_rows(day: str = "today") -> list:
    """
    Scrape core rows from ForexFactory calendar via Playwright (headless).
    Output shape (list of dicts):
      {event_id, date_label, time, currency, impact, event_title, actual, forecast, previous}
    """
    url = f"https://www.forexfactory.com/calendar?day={day}"
    rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle")
        html = page.content()
        soup = BeautifulSoup(html, "html.parser")
        table = soup.select_one("table.calendar__table")
        if not table:
            browser.close()
            return rows

        for tr in table.select("tr.calendar__row"):
            event_id = tr.get("data-eventid") or ""
            tds = tr.find_all("td")
            if len(tds) < 6:
                continue
            time_txt = (tds[1].get_text(strip=True) or "")
            currency = (tds[2].get_text(strip=True) or "")
            impact = (tds[3].get("title") or tds[3].get_text(strip=True) or "")
            title  = (tds[4].get_text(strip=True) or "")
            actual = (tds[5].get_text(strip=True) or "")
            forecast = (tds[6].get_text(strip=True) or "")
            previous = (tds[7].get_text(strip=True) or "") if len(tds) > 7 else ""
            rows.append({
                "event_id": event_id,
                "date_label": day,
                "time": time_txt, "currency": currency, "impact": impact,
                "event_title": title, "actual": actual, "forecast": forecast, "previous": previous
            })
        browser.close()
    return rows

# ---------- Prompts (GPT-5 & Typhoon) ----------

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_system_prompt_gpt5() -> str:
    return load_text("prompts/system_gpt5.txt")

def build_system_prompt_typhoon() -> str:
    return load_text("prompts/system_typhoon.txt")

def build_user_prompt(ctx: dict) -> str:
    tmpl = load_text("prompts/user_template.txt")
    return tmpl.format(**ctx)

# ---------- GPT/Typhoon callers ----------

def call_gpt5(system_prompt: str, user_prompt: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-5",  # adjust if you use specific variant
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.15,
        max_tokens=1400,
    )
    return resp.choices[0].message.content

# ---------- MAIN ----------

def main():
    print("🚀 Starting Pending-Order Strategy Bot...")
    cfg = StrategyConfig()
    ledger = WeeklyLedger(cfg)
    notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    typhoon = TyphoonForexAnalyzer(TYPHOON_API_KEY) if TYPHOON_API_KEY else None
    fetcher = IQDataFetcher()

    # 1) Scrape news (kept as before)
    core_events = scrape_ff_core_rows(day="today")
    log_today(f"📰 Extracted {len(core_events)} events")

    # 2) Loop pairs → build context → call GPT-5 (fallback Typhoon)
    system_gpt5 = build_system_prompt_gpt5()
    system_tfn  = build_system_prompt_typhoon()

    weekday_jst = (datetime.utcnow() + timedelta(hours=9)).weekday()
    tif_date = OrderPlanner.tif_for_today(weekday_jst)
    current_date = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")

    weekly_before = ledger.realized()

    for pair in PAIRS:
        try:
            tech = fetcher.get_technical_data(pair)
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
                # planning constraints
                "tif_date": tif_date,
                "rr_min": cfg.rr_min,
                "rr_max": cfg.rr_max,
                "allowed_combos": "BUY_STOP+SELL_STOP | BUY_LIMIT+BUY_STOP | SELL_LIMIT+SELL_STOP",
            }
            user_prompt = build_user_prompt(ctx)

            # Prefer GPT-5
            try:
                summary = call_gpt5(system_gpt5, user_prompt)
            except Exception as e:
                summary = None
                log_today(f"⚠️ GPT-5 error: {e}")

            # Fallback → Typhoon
            if not summary and typhoon:
                try:
                    summary = typhoon.analyze(system_tfn, user_prompt)
                except Exception as e:
                    summary = None
                    log_today(f"⚠️ Typhoon error: {e}")

            if not summary:
                notifier.send_message(f"❌ {pair}: No summary produced.")
                continue

            # Parse → OCOPlan
            plan = TyphoonParser.parse(summary, pair=pair, current_date=current_date, tif_date=tif_date)

            # Validate: combo, RR, sanity vs spot
            is_jpy = pair.endswith("/JPY")
            buf = OrderPlanner.buffer_pips(tech["d1_atr14"], float(tech["spread_pips"]), is_jpy)
            spot = float(tech["spot_mid"])

            if plan.order_a and plan.order_b and plan.validate_combo() and plan.validate_rr(cfg.rr_min, cfg.rr_max) and plan.sanity_vs_spot(spot, buf, pair):
                ledger.add_orders(plan.as_list())
                notifier.send_message(f"✅ {pair}: Plan accepted.\n\n{summary}")
            else:
                notifier.send_message(f"⏸️ {pair}: Plan rejected (combo/RR/sanity).\n\n{summary}")

        except Exception as e:
            notifier.send_message(f"❌ {pair}: Exception {e}")

    # 3) Daily review (Tue–Fri)
    if weekday_jst in (1, 2, 3, 4):
        for pair in PAIRS:
            try:
                tech = fetcher.get_technical_data(pair)
                d1_open = tech["prev_day_close"]   # conservative proxy
                d1_high = tech["prev_day_high"]
                d1_low  = tech["prev_day_low"]
                changed = ReviewEngine(cfg, ledger).apply_d1_bar(pair, d1_open, d1_high, d1_low)
                if changed:
                    notifier.send_message(f"🔁 {pair}: Updated states. Weekly realized: {ledger.realized():.1f} pips")
            except Exception as e:
                notifier.send_message(f"⚠️ {pair}: Review error {e}")

    weekly_after = ledger.realized()
    notifier.send_message(f"📊 Weekly P/L updated: {weekly_before:.1f} → {weekly_after:.1f} pips")

if __name__ == "__main__":
    main()
