# get_data.py
# IQ Option data wrapper: OHLC fetch, indicator pack (H1), and technical snapshot.
# Handles scalar-payload normalization and Bangkok timestamps.

import os, time, math
import pandas as pd
import pandas_ta as ta
from iqoptionapi.stable_api import IQ_Option
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

BKK_TZ = timezone(timedelta(hours=7))

def _to_bkk_str(epoch_sec: int) -> str:
    """Convert epoch seconds to 'YYYY-MM-DD HH:MM ICT' in Asia/Bangkok (UTC+7)."""
    if epoch_sec is None or (isinstance(epoch_sec, float) and math.isnan(epoch_sec)):
        return "N/A"
    return datetime.fromtimestamp(int(epoch_sec), tz=BKK_TZ).strftime("%Y-%m-%d %H:%M ICT")


class IQDataFetcher:
    """Thin wrapper over IQ Option API with payload normalization."""

    def __init__(self, username: str = None, password: str = None):
        self.username = username or os.getenv("IQ_USER")
        self.password = password or os.getenv("IQ_PASS")
        self.api = None

    def connect(self):
        if self.api:
            return
        self.api = IQ_Option(self.username, self.password)
        self.api.connect()
        for _ in range(40):
            if self.api.check_connect(): break
            time.sleep(0.25)
        if not self.api.check_connect():
            raise RuntimeError("IQ Option API connection failed")

    def close_connection(self):
        if self.api:
            try: self.api.logout()
            except Exception: pass

    @staticmethod
    def _tf_to_seconds(tf: str) -> int:
        table = {"M1":60, "M5":300, "M15":900, "H1":3600, "H4":14400, "D1":86400}
        if tf not in table: raise ValueError(f"Unsupported timeframe: {tf}")
        return table[tf]

    @staticmethod
    def _normalize_candles(payload) -> list:
        """Ensure list[dict] with keys: from,to,open,high,low,close."""
        if payload is None: return []
        if isinstance(payload, dict):
            if "candles" in payload and isinstance(payload["candles"], (list, tuple)):
                payload = payload["candles"]
            else:
                payload = [payload]
        elif not isinstance(payload, (list, tuple)):
            try: payload = list(payload)
            except Exception: return []

        out = []
        for c in payload:
            if not isinstance(c, dict): continue
            d = dict(c)
            if "high" not in d and "max" in d: d["high"] = d["max"]
            if "low"  not in d and "min" in d: d["low"]  = d["min"]
            req = ("from","to","open","close","high","low")
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
        candles = self.api.get_candles(pair.replace("/", ""), self._tf_to_seconds(tf), count, end_epoch)
        return self._normalize_candles(candles)

    def _fetch_ohlc(self, pair: str, tf: str, count: int = 400, retries: int = 3, backoff: float = 0.6) -> pd.DataFrame:
        self.connect()
        end = int(time.time())
        last_err = None
        for attempt in range(1, retries+1):
            try:
                data = self._fetch_ohlc_once(pair, tf, count, end)
                if not data: raise RuntimeError("Empty candles")
                df = pd.DataFrame.from_records(data)
                need = {"from","to","open","high","low","close"}
                if not need.issubset(df.columns):
                    raise RuntimeError("Missing candle columns")
                return df.sort_values("from").reset_index(drop=True)
            except Exception as e:
                last_err = e; time.sleep(backoff*attempt)
        raise RuntimeError(f"Failed to fetch candles for {pair} {tf}: {last_err}")

    # ---- Public helpers ----

    def get_bars(self, pair: str, timeframe: str = "H1", lookback: int = 200) -> pd.DataFrame:
        """Return ascending OHLC DataFrame for timeframe."""
        return self._fetch_ohlc(pair, timeframe, count=max(lookback, 50))

    def get_indicators(self, pair: str, timeframe: str = "H1", lookback: int = 200) -> dict:
        """
        Compute simple indicator pack on timeframe:
          - H1 EMA21/EMA55/RSI14
          - spot from last M1 close (bid proxy)
        """
        tf_df = self.get_bars(pair, timeframe=timeframe, lookback=lookback)
        df = tf_df.copy()
        df["ema21"] = ta.ema(df["close"], length=21)
        df["ema55"] = ta.ema(df["close"], length=55)
        df["rsi14"] = ta.rsi(df["close"], length=14)
        last = df.iloc[-1]
        h1 = {"ema_fast": float(last["ema21"]), "ema_slow": float(last["ema55"]), "rsi": float(last["rsi14"])}

        # Spot from last M1 close + simple spread model
        m1 = self.get_bars(pair, timeframe="M1", lookback=2)
        spot_bid = float(m1.iloc[-1]["close"])
        is_jpy = pair.replace(" ", "").upper().endswith("/JPY")
        unit = 0.01 if is_jpy else 0.0001
        spread_pips = float(os.getenv("DEFAULT_SPREAD_PIPS", "1.0"))
        spot_ask = spot_bid + spread_pips * unit
        spot_mid = (spot_bid + spot_ask) / 2.0

        return {"H1": h1, "spot": round(spot_mid, 5 if not is_jpy else 3)}

    def get_technical_data(self, pair: str) -> dict:
        """D1 ATR/ADR + last closed D1 + latest timestamps (Bangkok)."""
        # Spot/Spread
        m1 = self._fetch_ohlc(pair, "M1", count=2)
        if len(m1) == 0:
            raise RuntimeError(f"No M1 data for {pair}")
        spot_bid = float(m1.iloc[-1]["close"])
        is_jpy = pair.replace(" ", "").upper().endswith("/JPY")
        unit = 0.01 if is_jpy else 0.0001
        spread_pips = float(os.getenv("DEFAULT_SPREAD_PIPS", "1.0"))
        spot_ask = round(spot_bid + (spread_pips * unit), 3 if is_jpy else 5)
        spot_mid = round((spot_bid + spot_ask) / 2, 3 if is_jpy else 5)

        # Higher TFs
        h1  = self._fetch_ohlc(pair, "H1",  count=2)
        h4  = self._fetch_ohlc(pair, "H4",  count=2)
        m15 = self._fetch_ohlc(pair, "M15", count=2)
        d1  = self._fetch_ohlc(pair, "D1",  count=120)

        d1_closed = d1.iloc[-2] if len(d1) >= 2 else d1.iloc[-1]
        prev_day_high  = float(d1_closed["high"])
        prev_day_low   = float(d1_closed["low"])
        prev_day_close = float(d1_closed["close"])

        d1_calc = d1.copy()
        d1_calc["atr14"] = ta.atr(d1_calc["high"], d1_calc["low"], d1_calc["close"], length=14)
        d1_atr14 = float(d1_calc["atr14"].iloc[-1])
        d1_calc["range"] = d1_calc["high"] - d1_calc["low"]
        d1_adr20 = float(d1_calc["range"].tail(20).mean())

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
