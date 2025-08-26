# get_data.py
# Robust IQ Option fetcher with candle normalization and Bangkok timestamps.
# Fixes "If using all scalar values, you must pass an index" by forcing
# candles into a list-of-dicts shape before constructing a DataFrame.

import os
import time
import math
import pandas as pd
import pandas_ta as ta
from iqoptionapi.stable_api import IQ_Option
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

# ---------- Time helpers (Bangkok) ----------

def _to_bkk_str(epoch_sec: int) -> str:
    """Convert epoch seconds to 'YYYY-MM-DD HH:MM ICT' in Asia/Bangkok (UTC+7)."""
    if epoch_sec is None or (isinstance(epoch_sec, float) and math.isnan(epoch_sec)):
        return "N/A"
    ict = timezone(timedelta(hours=7))
    return datetime.fromtimestamp(int(epoch_sec), tz=ict).strftime("%Y-%m-%d %H:%M ICT")


# ---------- IQ Option wrapper ----------

class IQDataFetcher:
    """
    Thin wrapper over IQ Option API to fetch OHLC and compute technicals.
    Normalizes candle payloads to avoid pandas scalar dict errors.
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
        for _ in range(40):
            if self.api.check_connect():
                break
            time.sleep(0.25)
        if not self.api.check_connect():
            raise RuntimeError("IQ Option API connection failed")

    def close_connection(self):
        """Explicitly logout/close session."""
        if self.api:
            try:
                self.api.logout()
            except Exception:
                pass

    # ---------- Internals ----------

    @staticmethod
    def _tf_to_seconds(tf: str) -> int:
        table = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600, "H4": 14400, "D1": 86400}
        if tf not in table:
            raise ValueError(f"Unsupported timeframe: {tf}")
        return table[tf]

    @staticmethod
    def _normalize_candles(payload) -> list:
        """
        Force candle payload to a list of dicts with required fields.
        Accepts: list[dict], tuple[dict], dict (single candle), dict with 'candles' key, or iterable.
        Filters out malformed entries and normalizes high/low key names.
        """
        if payload is None:
            return []
        # Unwrap common shapes
        if isinstance(payload, dict):
            if "candles" in payload and isinstance(payload["candles"], (list, tuple)):
                payload = payload["candles"]
            else:
                # Single candle as dict?
                payload = [payload]
        elif not isinstance(payload, (list, tuple)):
            try:
                payload = list(payload)
            except Exception:
                return []

        out = []
        for c in payload:
            if not isinstance(c, dict):
                continue
            d = dict(c)  # shallow copy
            # Normalize high/low naming
            if "high" not in d and "max" in d: d["high"] = d["max"]
            if "low"  not in d and "min" in d: d["low"]  = d["min"]
            # Required fields
            req = ("from", "to", "open", "close", "high", "low")
            if all(k in d for k in req):
                out.append({
                    "from":  int(d["from"]),
                    "to":    int(d["to"]),
                    "open":  float(d["open"]),
                    "high":  float(d["high"]),
                    "low":   float(d["low"]),
                    "close": float(d["close"]),
                })
        return out

    def _fetch_ohlc_once(self, pair: str, tf: str, count: int, end_epoch: int):
        """Single API call with normalization; returns list[dict]."""
        candles = self.api.get_candles(pair.replace("/", ""), self._tf_to_seconds(tf), count, end_epoch)
        # iqoptionapi sometimes returns ok=False but a valid list; proceed anyway
        norm = self._normalize_candles(candles)
        return norm

    def _fetch_ohlc(self, pair: str, tf: str, count: int = 400, retries: int = 3, backoff: float = 0.6) -> pd.DataFrame:
        """
        Fetch 'count' candles ending at 'now'. Retries and normalizes payload.
        Returns DataFrame with columns: ['from','to','open','high','low','close'] sorted ascending.
        """
        self.connect()
        end = int(time.time())
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                data = self._fetch_ohlc_once(pair, tf, count, end)
                if not data:
                    raise RuntimeError(f"No candles received (pair={pair}, tf={tf}, attempt={attempt})")
                df = pd.DataFrame.from_records(data)
                # Ensure expected columns exist
                need = {"from", "to", "open", "high", "low", "close"}
                if not need.issubset(df.columns):
                    raise RuntimeError(f"Candle columns missing: {need - set(df.columns)} (pair={pair}, tf={tf})")
                df = df.sort_values("from").reset_index(drop=True)
                return df
            except Exception as e:
                last_err = e
                time.sleep(backoff * attempt)
        # If we reach here, all retries failed
        raise RuntimeError(f"Failed to fetch candles for {pair} {tf}: {last_err}")

    # ---------- Technicals for strategy ----------

    def get_technical_data(self, pair: str) -> dict:
        """
        Provide unified data for strategy:
          - spot bid/ask/mid, spread(pips est.)
          - prev_day_high/low/close (last closed D1)
          - d1_atr14, d1_adr20
          - latest *_ict timestamps (Bangkok)
        """
        self.connect()

        # Spot/Spread from last M1 close
        m1 = self._fetch_ohlc(pair, "M1", count=2)
        if len(m1) == 0:
            raise RuntimeError(f"No M1 data for {pair}")
        spot_bid = float(m1.iloc[-1]["close"])

        is_jpy = pair.endswith("/JPY")
        unit = 0.01 if is_jpy else 0.0001
        try:
            spread_pips = float(os.getenv("DEFAULT_SPREAD_PIPS", "1.0"))
        except Exception:
            spread_pips = 1.0
        spot_ask = round(spot_bid + (spread_pips * unit), 3 if is_jpy else 5)
        spot_mid = round((spot_bid + spot_ask) / 2, 3 if is_jpy else 5)

        # Higher TFs for timestamps & D1 calculations
        h1  = self._fetch_ohlc(pair, "H1",  count=2)
        h4  = self._fetch_ohlc(pair, "H4",  count=2)
        m15 = self._fetch_ohlc(pair, "M15", count=2)
        d1  = self._fetch_ohlc(pair, "D1",  count=120)

        # Last closed D1 (use -2 if at least 2 bars exist)
        d1_closed = d1.iloc[-2] if len(d1) >= 2 else d1.iloc[-1]
        prev_day_high  = float(d1_closed["high"])
        prev_day_low   = float(d1_closed["low"])
        prev_day_close = float(d1_closed["close"])

        # ATR(14) on D1 (guard against NaN)
        d1_calc = d1.copy()
        d1_calc.rename(columns={"from": "time"}, inplace=True)
        d1_calc["atr14"] = ta.atr(d1_calc["high"], d1_calc["low"], d1_calc["close"], length=14)
        d1_atr14 = d1_calc["atr14"].iloc[-1]
        if pd.isna(d1_atr14):
            # Fallback: use mean of last N true ranges as coarse ATR
            tr = (d1_calc["high"] - d1_calc["low"]).tail(14)
            d1_atr14 = float(tr.mean()) if len(tr) else 0.0
        else:
            d1_atr14 = float(d1_atr14)

        # ADR(20): average of (high-low) over last 20 days
        d1_calc["range"] = d1_calc["high"] - d1_calc["low"]
        adr20 = d1_calc["range"].tail(20).mean()
        d1_adr20 = float(adr20) if not pd.isna(adr20) else 0.0

        # Latest candle timestamps (Bangkok)
        latest_h4_ict  = _to_bkk_str(h4.iloc[-1]["from"])  if len(h4)  else "N/A"
        latest_h1_ict  = _to_bkk_str(h1.iloc[-1]["from"])  if len(h1)  else "N/A"
        latest_m15_ict = _to_bkk_str(m15.iloc[-1]["from"]) if len(m15) else "N/A"
        latest_d1_ict  = _to_bkk_str(d1.iloc[-1]["from"])  if len(d1)  else "N/A"

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
