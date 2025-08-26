# get_data.py
# Price/technical data provider using IQ Option API + pandas-ta.
# Returns unified fields consumed by new_forex.py (spot/spread/ATR/ADR/D1 prev OHLC + timestamps).

import os
import time
import pandas as pd
import pandas_ta as ta
from iqoptionapi.stable_api import IQ_Option
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

# ---------- Time helpers ----------

def _to_tz_str(epoch_sec: int, tz_hours: int = 7) -> str:
    """Convert epoch seconds to 'YYYY-MM-DD HH:MM TZ' with fixed offset (ICT by default)."""
    if epoch_sec is None:
        return "N/A"
    tz = timezone(timedelta(hours=tz_hours))
    return datetime.fromtimestamp(epoch_sec, tz=tz).strftime("%Y-%m-%d %H:%M ICT")

# ---------- IQ Option wrapper ----------

class IQDataFetcher:
    """
    Thin wrapper over IQ Option API to fetch OHLC and compute technicals.
    Assumes your account can access the requested FX symbols.
    """

    def __init__(self, username: str = None, password: str = None):
        self.username = username or os.getenv("IQ_USER")
        self.password = password or os.getenv("IQ_PASS")
        self.api = None

    def connect(self):
        """Establish IQ Option session if not connected."""
        if self.api:
            return
        self.api = IQ_Option(self.username, self.password)
        self.api.connect()
        # Wait until the connection is confirmed
        for _ in range(20):
            if self.api.check_connect():
                break
            time.sleep(0.5)
        if not self.api.check_connect():
            raise RuntimeError("IQ Option API connection failed")

    def close_connection(self):
        """Explicitly logout/close session."""
        if self.api:
            try:
                self.api.logout()
            except Exception:
                pass

    # ---------- Data fetching ----------

    @staticmethod
    def _tf_to_seconds(tf: str) -> int:
        table = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600, "H4": 14400, "D1": 86400}
        return table[tf]

    def _fetch_ohlc(self, pair: str, tf: str, count: int = 400) -> pd.DataFrame:
        """
        Fetch 'count' candles ending at 'now' (epoch).
        Returns DataFrame with columns: ['from','to','open','max','min','close']
        """
        self.connect()
        seconds = self._tf_to_seconds(tf)
        end = int(time.time())
        ok, candles = self.api.get_candles(pair.replace("/", ""), seconds, count, end)
        if not ok and isinstance(candles, list):
            # IQ API returns list even when ok=False sometimes; proceed
            pass
        df = pd.DataFrame(candles)
        df = df.rename(columns={"max": "high", "min": "low"})
        # Ensure ordering by time ascending
        df = df.sort_values("from").reset_index(drop=True)
        return df

    # ---------- Technicals for strategy ----------

    def get_technical_data(self, pair: str) -> dict:
        """
        Provide unified data for strategy:
          - spot bid/ask/mid, spread(pips)
          - prev_day_high/low/close
          - d1_atr14, d1_adr20
          - latest *_ict timestamps
        """
        self.connect()

        # Spot/Spread (mid from last M1)
        m1 = self._fetch_ohlc(pair, "M1", count=2)
        spot_bid = float(m1.iloc[-1]["close"])
        # For a fair estimation, assume +1 pip for ask (crude). You can replace with real quotes if available.
        is_jpy = pair.endswith("/JPY")
        unit = 0.01 if is_jpy else 0.0001
        spread_pips = float(os.getenv("DEFAULT_SPREAD_PIPS", "1.0"))
        spot_ask = round(spot_bid + (spread_pips * unit), 3 if is_jpy else 5)
        spot_mid = round((spot_bid + spot_ask) / 2, 3 if is_jpy else 5)

        # Higher TFs for timestamps
        h1 = self._fetch_ohlc(pair, "H1", count=2)
        h4 = self._fetch_ohlc(pair, "H4", count=2)
        m15 = self._fetch_ohlc(pair, "M15", count=2)
        d1 = self._fetch_ohlc(pair, "D1", count=120)

        # Prev day OHLC (use last closed D1)
        d1_closed = d1.iloc[-2] if len(d1) >= 2 else d1.iloc[-1]
        prev_day_high = float(d1_closed["high"])
        prev_day_low  = float(d1_closed["low"])
        prev_day_close= float(d1_closed["close"])

        # ATR(14) on D1
        d1_calc = d1.copy()
        d1_calc.rename(columns={"from": "time"}, inplace=True)
        d1_calc["atr14"] = ta.atr(d1_calc["high"], d1_calc["low"], d1_calc["close"], length=14)
        d1_atr14 = float(d1_calc["atr14"].iloc[-1])

        # ADR(20): average of (high-low) over last 20 days
        d1_calc["range"] = d1_calc["high"] - d1_calc["low"]
        adr20 = d1_calc["range"].tail(20).mean()
        d1_adr20 = float(adr20)

        # Latest candle timestamps (ICT string)
        latest_h4_ict  = _to_tz_str(int(h4.iloc[-1]["from"]), tz_hours=7)
        latest_h1_ict  = _to_tz_str(int(h1.iloc[-1]["from"]), tz_hours=7)
        latest_m15_ict = _to_tz_str(int(m15.iloc[-1]["from"]), tz_hours=7)
        latest_d1_ict  = _to_tz_str(int(d1.iloc[-1]["from"]), tz_hours=7)

        return {
            "pair": pair,
            "spot_bid": spot_bid,
            "spot_ask": spot_ask,
            "spot_mid": spot_mid,
            "spread_pips": spread_pips,
            "prev_day_high": prev_day_high,
            "prev_day_low": prev_day_low,
            "prev_day_close": prev_day_close,
            "d1_atr14": d1_atr14,
            "d1_adr20": d1_adr20,
            "latest_h4_ict": latest_h4_ict,
            "latest_h1_ict": latest_h1_ict,
            "latest_m15_ict": latest_m15_ict,
            "latest_d1_ict": latest_d1_ict,
        }
