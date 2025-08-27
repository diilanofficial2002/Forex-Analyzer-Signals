# get_data.py
# Robust IQ Option fetcher with candle normalization, H1-centric indicators,
# and Bangkok timestamps. Provides get_indicators() and get_bars() used by new_forex.py.

import os
import time
import math
import pandas as pd
import pandas_ta as ta
from iqoptionapi.stable_api import IQ_Option
from datetime import datetime, timezone, timedelta

# ---------- Time helpers (Bangkok) ----------

BKK_TZ = timezone(timedelta(hours=7))

def _to_bkk_str(epoch_sec: int) -> str:
    """Convert epoch seconds to 'YYYY-MM-DD HH:MM ICT' in Asia/Bangkok (UTC+7)."""
    if epoch_sec is None or (isinstance(epoch_sec, float) and math.isnan(epoch_sec)):
        return "N/A"
    return datetime.fromtimestamp(int(epoch_sec), tz=BKK_TZ).strftime("%Y-%m-%d %H:%M ICT")


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

    # ---------------- Connection ----------------

    def connect(self):
        """Establish IQ Option session if not connected."""
        if self.api:
            return
        if not self.username or not self.password:
            raise RuntimeError("IQ_USER / IQ_PASS missing in environment")
        self.api = IQ_Option(self.username, self.password)
        self.api.connect()
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

    # ---------------- Internals ----------------

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
            d = dict(c)
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
        symbol = pair.replace("/", "")
        candles = self.api.get_candles(symbol, self._tf_to_seconds(tf), count, end_epoch)
        return self._normalize_candles(candles)

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
                need = {"from", "to", "open", "high", "low", "close"}
                if not need.issubset(df.columns):
                    raise RuntimeError(f"Candle columns missing: {need - set(df.columns)} (pair={pair}, tf={tf})")
                df = df.sort_values("from").reset_index(drop=True)
                return df
            except Exception as e:
                last_err = e
                time.sleep(backoff * attempt)
        raise RuntimeError(f"Failed to fetch candles for {pair} {tf}: {last_err}")

    # ---------------- Public API ----------------

    def get_bars(self, pair: str, timeframe: str = "H1", lookback: int = 200) -> list:
        """
        Return recent bars as list[dict] for the requested timeframe. Keys: from,to,open,high,low,close
        """
        df = self._fetch_ohlc(pair, timeframe, count=max(lookback, 50))
        # Convert to list-of-dicts with numeric types preserved
        recs = df.tail(lookback).to_dict(orient="records")
        return recs

    def get_indicators(self, pair: str, timeframe: str = "H1", lookback: int = 200) -> dict:
        """
        Compute H1 indicators needed by the strategy:
          - spot (mid) based on last M1 close + configurable default spread
          - H1: EMA(21), EMA(55), RSI(14)
        """
        # Spot/Spread from last M1 close
        m1 = self._fetch_ohlc(pair, "M1", count=2)
        if len(m1) == 0:
            raise RuntimeError(f"No M1 data for {pair}")
        spot_bid = float(m1.iloc[-1]["close"])

        is_jpy = pair.upper().endswith("/JPY")
        unit = 0.01 if is_jpy else 0.0001
        try:
            spread_pips = float(os.getenv("DEFAULT_SPREAD_PIPS", "1.0"))
        except Exception:
            spread_pips = 1.0
        spot_ask = round(spot_bid + (spread_pips * unit), 3 if is_jpy else 5)
        spot_mid = round((spot_bid + spot_ask) / 2, 3 if is_jpy else 5)

        # H1 bars for indicators
        h1 = self._fetch_ohlc(pair, timeframe, count=max(lookback, 120))
        # Pandas_ta requires Series; we compute EMA21/EMA55/RSI14
        ema21 = ta.ema(h1["close"], length=21)
        ema55 = ta.ema(h1["close"], length=55)
        rsi14 = ta.rsi(h1["close"], length=14)

        h1_ind = {
            "ema_fast": float(ema21.iloc[-1]) if pd.notna(ema21.iloc[-1]) else float(h1["close"].iloc[-1]),
            "ema_slow": float(ema55.iloc[-1]) if pd.notna(ema55.iloc[-1]) else float(h1["close"].iloc[-1]),
            "rsi": float(rsi14.iloc[-1]) if pd.notna(rsi14.iloc[-1]) else 50.0
        }

        return {
            "H1": h1_ind,
            "spot": spot_mid
        }
