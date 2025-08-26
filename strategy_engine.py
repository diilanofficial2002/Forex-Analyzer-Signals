# strategy_engine.py
# OOP engine for pending-order strategy with weekly tracking, OCO combos,
# RR enforcement, daily D1 review, and robust parsing of model outputs.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
import os, json, re
from datetime import datetime, timedelta, timezone

# ------------------------------
# Config & utilities
# ------------------------------

@dataclass
class StrategyConfig:
    timezone_name: str = os.getenv("TRADING_TZ", "Asia/Bangkok")
    pairs: List[str] = None
    weekly_pip_goal: int = 1000
    rr_min: float = 1.5
    rr_max: float = 2.0
    default_spread_pips: float = float(os.getenv("DEFAULT_SPREAD_PIPS", "1"))
    state_dir: str = "data"
    weekly_state_file: str = "data/weekly_ledger.json"

    def __post_init__(self):
        if self.pairs is None:
            self.pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/GBP", "EUR/CHF"]


class PipMath:
    """Pip conversion helpers for JPY/non-JPY pairs."""
    @staticmethod
    def is_jpy(pair: str) -> bool:
        return pair.endswith("/JPY")

    @staticmethod
    def pips_between(pair: str, price_a: float, price_b: float) -> float:
        if price_a is None or price_b is None:
            return 0.0
        diff = abs(price_b - price_a)
        return diff / 0.01 if PipMath.is_jpy(pair) else diff / 0.0001

    @staticmethod
    def add_pips(pair: str, price: float, pips: float) -> float:
        unit = 0.01 if PipMath.is_jpy(pair) else 0.0001
        # Keep sensible digits (3 for JPY, 5 otherwise)
        return round(price + (pips * unit), 3 if PipMath.is_jpy(pair) else 5)


# ------------------------------
# Order structures
# ------------------------------

@dataclass
class PendingOrder:
    pair: str
    side: str          # "LONG" or "SHORT"
    order_type: str    # "Buy Stop" | "Sell Stop" | "Buy Limit" | "Sell Limit"
    entry: float
    sl: float
    tp: float
    tif_date: str      # "YYYY-MM-DD"
    created_at: str    # "YYYY-MM-DD"
    status: str = "PENDING"  # PENDING | TRIGGERED | CLOSED | CANCELLED
    result: Optional[str] = None  # "TP" | "SL" | "UNCERTAIN" | None

    def rr(self) -> Optional[float]:
        """Compute R:R from entry/SL/TP respecting side direction."""
        if self.side == "LONG":
            risk = self.entry - self.sl
            reward = self.tp - self.entry
        else:
            risk = self.sl - self.entry
            reward = self.entry - self.tp
        if risk <= 0:
            return None
        return round(reward / risk, 2)


@dataclass
class OCOPlan:
    """OCO = One Cancels Other; we allow the three combos specified by user."""
    order_a: Optional[PendingOrder]
    order_b: Optional[PendingOrder]

    def as_list(self) -> List[PendingOrder]:
        return [o for o in (self.order_a, self.order_b) if o]

    def validate_combo(self) -> bool:
        """
        Only allow EXACTLY these combos:
          1) (Buy Stop, Sell Stop)       -> breakout both sides
          2) (Buy Limit, Buy Stop)       -> both LONG (mean reversion + breakout)
          3) (Sell Limit, Sell Stop)     -> both SHORT (mean reversion + breakout)
        """
        if not (self.order_a and self.order_b):
            return False
        types = (self.order_a.order_type, self.order_b.order_type)
        allowed = {
            ("Buy Stop", "Sell Stop"),
            ("Sell Stop", "Buy Stop"),
            ("Buy Limit", "Buy Stop"),
            ("Sell Limit", "Sell Stop"),
        }
        return types in allowed

    def validate_rr(self, rr_min: float, rr_max: float) -> bool:
        for o in self.as_list():
            r = o.rr()
            if r is None or r < rr_min or r > rr_max:
                return False
        return True

    def sanity_vs_spot(self, spot: float, buffer_pips: float, pair: str) -> bool:
        """
        Basic sanity: check entry vs spot + small buffer for stops; limits should be across the spot.
        """
        unit = 0.01 if PipMath.is_jpy(pair) else 0.0001
        buf = buffer_pips * unit
        ok = True
        for o in self.as_list():
            if o.order_type == "Buy Limit" and not (o.entry < spot):
                ok = False
            if o.order_type == "Sell Limit" and not (o.entry > spot):
                ok = False
            if o.order_type == "Buy Stop" and not (o.entry > spot + buf):
                ok = False
            if o.order_type == "Sell Stop" and not (o.entry < spot - buf):
                ok = False
        return ok


# ------------------------------
# Weekly state (persist/reset Monday JST)
# ------------------------------

class WeeklyLedger:
    """
    Weekly ledger auto-resets every Monday (JST by default).
    Stores orders + realized pips for the running week.
    """
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.path = cfg.weekly_state_file
        os.makedirs(cfg.state_dir, exist_ok=True)
        self._state = self._load()

    def _monday_jst(self) -> datetime.date:
        jst = timezone(timedelta(hours=9))
        today_jst = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(jst).date()
        return (today_jst - timedelta(days=today_jst.weekday()))

    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return {"week_monday": str(self._monday_jst()), "orders": [], "realized_pips": 0.0}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"week_monday": str(self._monday_jst()), "orders": [], "realized_pips": 0.0}
        # Reset if file belongs to older week
        if data.get("week_monday") != str(self._monday_jst()):
            data = {"week_monday": str(self._monday_jst()), "orders": [], "realized_pips": 0.0}
        return data

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)

    # API
    def add_orders(self, orders: List[PendingOrder]):
        for o in orders:
            self._state["orders"].append(asdict(o))
        self.save()

    def iter_indices(self, pair: Optional[str] = None):
        """Yield (global_idx, row) to safely update records."""
        for i, row in enumerate(self._state.get("orders", [])):
            if (pair is None) or (row["pair"] == pair):
                yield i, row

    def update_order(self, idx: int, row: dict):
        self._state["orders"][idx] = row
        self.save()

    def realized(self) -> float:
        return float(self._state.get("realized_pips", 0.0))

    def add_realized(self, pips: float):
        self._state["realized_pips"] = float(self.realized() + pips)
        self.save()


# ------------------------------
# Execution Simulator (D1 candles)
# ------------------------------

class ExecutionSimulator:
    """
    Update orders using D1 OHLC bars (conservative):
    - Trigger if entry lies within bar (with correct semantics for stop/limit).
    - If both TP and SL lie within the same bar post-trigger -> mark "UNCERTAIN".
    """

    @staticmethod
    def _triggered(o: dict, d_open: float, d_high: float, d_low: float) -> bool:
        typ  = o["order_type"]
        entry = float(o["entry"])
        if typ == "Buy Stop":   return d_high >= entry
        if typ == "Sell Stop":  return d_low  <= entry
        if typ == "Buy Limit":  return d_low  <= entry
        if typ == "Sell Limit": return d_high >= entry
        return False

    @staticmethod
    def _tp_sl_hit_in_bar(o: dict, d_high: float, d_low: float) -> Tuple[bool, bool]:
        tp = float(o["tp"]); sl = float(o["sl"])
        if o["side"] == "LONG":
            tp_hit = d_high >= tp
            sl_hit = d_low  <= sl
        else:
            tp_hit = d_low  <= tp
            sl_hit = d_high >= sl
        return tp_hit, sl_hit

    @staticmethod
    def step_daily(order_row: dict, d_open: float, d_high: float, d_low: float) -> dict:
        if order_row["status"] in ("CLOSED", "CANCELLED"):
            return order_row

        # Trigger transition
        if order_row["status"] == "PENDING" and ExecutionSimulator._triggered(order_row, d_open, d_high, d_low):
            order_row["status"] = "TRIGGERED"

        # Post-trigger: evaluate TP/SL
        if order_row["status"] == "TRIGGERED":
            tp_hit, sl_hit = ExecutionSimulator._tp_sl_hit_in_bar(order_row, d_high, d_low)
            if tp_hit and sl_hit:
                order_row["result"] = "UNCERTAIN"  # ambiguous -> keep open and review next day
            elif tp_hit:
                order_row["status"] = "CLOSED"; order_row["result"] = "TP"
            elif sl_hit:
                order_row["status"] = "CLOSED"; order_row["result"] = "SL"
        return order_row


# ------------------------------
# Typhoon/GPT summary parser → OCOPlan
# ------------------------------

class TyphoonParser:
    """
    Parse a standardized block. We accept either:
    1) Bold markers:
       **COMBO:** BUY_LIMIT+BUY_STOP
       **TIF:** 2025-08-28
       **PRIMARY ORDER:** LONG — Buy Limit @ 1.23450
       **TP:** 1.23650
       **SL:** 1.23350
       **SECONDARY (OCO):** LONG — Buy Stop @ 1.23520
       **TP:** 1.23720
       **SL:** 1.23420

    2) Or the same fields without bold (fallback).
    """

    @staticmethod
    def _find(pattern: str, text: str, flags=re.I) -> Optional[re.Match]:
        return re.search(pattern, text, flags)

    @staticmethod
    def _findall(pattern: str, text: str, flags=re.I) -> List[str]:
        return re.findall(pattern, text, flags)

    @staticmethod
    def parse(summary_text: str, pair: str, current_date: str, tif_date: str) -> OCOPlan:
        combo = TyphoonParser._find(r"(?:\*\*COMBO:\*\*|COMBO:)\s*([A-Z_+\s]+)", summary_text)
        # Primary
        prim = TyphoonParser._find(
            r"(?:\*\*PRIMARY ORDER:\*\*|PRIMARY ORDER:)\s*(LONG|SHORT)\s*—\s*(Buy Stop|Sell Stop|Buy Limit|Sell Limit)\s*@\s*([\d\.]+)",
            summary_text
        )
        # Secondary
        sec = TyphoonParser._find(
            r"(?:\*\*SECONDARY\s*\(OCO\):\*\*|SECONDARY\s*\(OCO\):)\s*(LONG|SHORT)\s*—\s*(Buy Stop|Sell Stop|Buy Limit|Sell Limit)\s*@\s*([\d\.]+)",
            summary_text
        )
        # Capture TP/SL per block (allow two sets)
        tps  = TyphoonParser._findall(r"(?:\*\*TP:\*\*|TP:)\s*([\d\.]+)", summary_text)
        sls  = TyphoonParser._findall(r"(?:\*\*SL:\*\*|SL:)\s*([\d\.]+)", summary_text)

        def mk_order(m: Optional[re.Match], tp: Optional[float], sl: Optional[float]) -> Optional[PendingOrder]:
            if not (m and tp and sl):
                return None
            side, otype, entry = m.group(1).upper(), m.group(2), float(m.group(3))
            return PendingOrder(
                pair=pair, side=side, order_type=otype,
                entry=entry, sl=float(sl), tp=float(tp),
                tif_date=tif_date, created_at=current_date
            )

        # Heuristic: first TP/SL → primary, second → secondary (if present)
        p_tp = float(tps[0]) if len(tps) >= 1 else None
        p_sl = float(sls[0]) if len(sls) >= 1 else None
        s_tp = float(tps[1]) if len(tps) >= 2 else p_tp
        s_sl = float(sls[1]) if len(sls) >= 2 else p_sl

        order_a = mk_order(prim, p_tp, p_sl)
        order_b = mk_order(sec,  s_tp, s_sl)

        # If model outputs the straddle (Buy Stop & Sell Stop), ensure sides
        # are consistent with types to avoid mistakes:
        def infer_side(otype: str) -> str:
            return "LONG" if "Buy" in otype else "SHORT"

        for o in (order_a, order_b):
            if o and o.side not in ("LONG", "SHORT"):
                o.side = infer_side(o.order_type)

        return OCOPlan(order_a=order_a, order_b=order_b)


# ------------------------------
# Daily review / planning
# ------------------------------

class ReviewEngine:
    """Daily review Tue-Fri; Monday is implicit weekly reset via WeeklyLedger load()."""

    @staticmethod
    def max_hold_days(weekday: int) -> int:
        # Monday=0..Friday=4
        return {0: 5, 1: 4, 2: 3, 3: 2, 4: 1}.get(weekday, 0)

    def __init__(self, cfg: StrategyConfig, ledger: WeeklyLedger):
        self.cfg = cfg
        self.ledger = ledger

    def apply_d1_bar(self, pair: str, d_open: float, d_high: float, d_low: float):
        changed = False
        for idx, row in self.ledger.iter_indices(pair):
            new_row = ExecutionSimulator.step_daily(row, d_open, d_high, d_low)

            # Realize pips on close
            if row != new_row and new_row["status"] == "CLOSED":
                if new_row["result"] == "TP":
                    pips = PipMath.pips_between(pair, float(new_row["entry"]), float(new_row["tp"]))
                elif new_row["result"] == "SL":
                    pips = -PipMath.pips_between(pair, float(new_row["entry"]), float(new_row["sl"]))
                else:
                    pips = 0.0
                self.ledger.add_realized(pips)
                changed = True

            # Expire if TIF passed
            if new_row["status"] == "PENDING":
                try:
                    if datetime.strptime(new_row["tif_date"], "%Y-%m-%d").date() < datetime.utcnow().date():
                        new_row["status"] = "CANCELLED"; changed = True
                except Exception:
                    pass

            if row != new_row:
                self.ledger.update_order(idx, new_row)
        return changed


class OrderPlanner:
    """Helpers for buffer and TIF calculations."""
    @staticmethod
    def buffer_pips(atr14: Optional[float], spread_pips: float, is_jpy: bool) -> float:
        # Use 10% of ATR as trigger buffer, but never below (spread + 2p)
        try:
            if atr14 and float(atr14) > 0:
                atr_pips = float(atr14) / (0.01 if is_jpy else 0.0001)
                return max(atr_pips * 0.10, spread_pips + 2.0)
        except Exception:
            pass
        return 12.0 if not is_jpy else 0.12

    @staticmethod
    def tif_for_today(weekday: int) -> str:
        max_days = ReviewEngine.max_hold_days(weekday)
        return (datetime.utcnow() + timedelta(days=max_days)).strftime("%Y-%m-%d")
