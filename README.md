# Pending-Order OCO Strategy Bot

- Only 3 OCO combos: (Buy Stop, Sell Stop), (Buy Limit, Buy Stop), (Sell Limit, Sell Stop)
- Enforce RR within [1.5, 2.0]
- Close within week by weekday TIF rule (Mon≤5d ... Fri=1d)
- Weekly ledger + daily review Tue–Fri from D1 bars
- Monday auto-reset
- Playwright ForexFactory scraping preserved

## Quickstart

1. `cp .env.example .env` and fill credentials.
2. `pip install -r requirements.txt`
3. `playwright install`
4. `python new_forex.py`
