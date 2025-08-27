# strategy_engine.py
# Core strategy components: config, OCO plan object, validation, pip math, ledger, repricing.
# NOW WITH DYNAMIC, DAY-BASED STRATEGY CONFIGURATION.

import os
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime, timezone, timedelta


# --- Normalizer for order type (defensive layer in strategy) ---

def normalize_order_type(s: str) -> str:
    if not isinstance(s, str):
        return ""
    key = s.strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    key = key.replace(" ", "")
    mapping = {
        "buystop":  "Buy Stop",
        "sellstop": "Sell Stop",
        "buylimit": "Buy Limit",
        "selllimit":"Sell Limit",
    }
    return mapping.get(key, "")

# ----------------------------- Pip math -----------------------------

class PipMath:
    @staticmethod
    def pip_unit(pair: str) -> float:
        """Return the unit price of 1 pip for the pair."""
        if pair.upper().endswith("/JPY"):
            return 0.01
        return 0.0001

# ----------------------------- Config -----------------------------

@dataclass
class StrategyConfig:
    """
    Policy knobs for validation. Now dynamically adjusts based on the day of the week.
    - Mon/Tue: Conservative (higher RR, wider buffer)
    - Wed/Thu: Balanced (standard settings)
    - Fri: Aggressive (lower RR, tighter buffer for short-term moves)
    """
    tz: timezone = timezone(timedelta(hours=7))
    ledger_path: str = os.path.join("data", "weekly_ledger.json")
    h1_lookback: int = int(os.getenv("H1_LOOKBACK", "30"))

    # These will be set dynamically in __post_init__
    rr_min: float = field(init=False)
    rr_max: float = field(init=False)
    prefer_rr: float = field(init=False)
    buf_nonjpy: int = field(init=False)
    buf_jpy: int = field(init=False)

    def __post_init__(self):
        """Set strategy parameters dynamically based on the current day."""
        weekday = datetime.now(self.tz).weekday() # Monday is 0 and Sunday is 6

        if weekday in [0, 1]: # Monday, Tuesday (Conservative)
            print("Config: Conservative Mode (Mon/Tue)")
            self.rr_min = float(os.getenv("RR_MIN_CONSERVATIVE", "1.5"))
            self.prefer_rr = float(os.getenv("PREFER_RR_CONSERVATIVE", "2.0"))
            self.buf_nonjpy = int(os.getenv("BUFFER_PIPS_NONJPY_CONSERVATIVE", "8"))
            self.buf_jpy = int(os.getenv("BUFFER_PIPS_JPY_CONSERVATIVE", "80"))

        elif weekday in [2, 3]: # Wednesday, Thursday (Balanced)
            print("💡 Config: Balanced Mode (Wed/Thu)")
            self.rr_min = float(os.getenv("RR_MIN_BALANCED", "1.25"))
            self.prefer_rr = float(os.getenv("PREFER_RR_BALANCED", "1.5"))
            self.buf_nonjpy = int(os.getenv("BUFFER_PIPS_NONJPY_BALANCED", "5"))
            self.buf_jpy = int(os.getenv("BUFFER_PIPS_JPY_BALANCED", "60"))

        else: # Friday, Saturday, Sunday (Aggressive short-term)
            print("💡 Config: Aggressive Mode (Fri/Weekend)")
            self.rr_min = float(os.getenv("RR_MIN_AGGRESSIVE", "1.1"))
            self.prefer_rr = float(os.getenv("PREFER_RR_AGGRESSIVE", "1.25"))
            self.buf_nonjpy = int(os.getenv("BUFFER_PIPS_NONJPY_AGGRESSIVE", "4"))
            self.buf_jpy = int(os.getenv("BUFFER_PIPS_JPY_AGGRESSIVE", "50"))

        # rr_max can remain constant
        self.rr_max = float(os.getenv("RR_MAX", "5.0"))


    def buffer_pips(self, pair: str) -> int:
        return self.buf_jpy if pair.upper().endswith("/JPY") else self.buf_nonjpy

# ----------------------------- Order Legs -----------------------------

@dataclass
class OrderLeg:
    order_type: str   # "Buy Stop", "Sell Stop", "Buy Limit", "Sell Limit"
    entry: float
    sl: float
    tp: float

    def rr(self) -> float:
        risk = abs(self.entry - self.sl)
        reward = abs(self.tp - self.entry)
        if risk <= 0:
            return 0.0
        return reward / risk

# ----------------------------- OCO Plan -----------------------------

@dataclass
class OCOPlan:
    pair: str
    combo: Tuple[str, str]
    legs: Tuple[OrderLeg, OrderLeg]
    time_in_force: dict
    status: str
    confidence: float
    created_at: str
    updated_at: str
    source: dict = field(default_factory=dict)
    decision: str = "pending"
    reason: str = ""

    def validate_combo(self, cfg: StrategyConfig) -> bool:
        allowed = {
            ("Buy Stop","Sell Stop"),
            ("Buy Limit","Buy Stop"),
            ("Sell Limit","Sell Stop"),
            ("Sell Limit","Buy Limit"),
        }
        # Normalize again defensively
        a = normalize_order_type(self.combo[0])
        b = normalize_order_type(self.combo[1])
        if not a or not b:
            return False
        return (a, b) in allowed

    def validate_rr(self, rr_min: float, rr_max: float) -> bool:
        good = [rr_min <= leg.rr() <= rr_max for leg in self.legs]
        # Ensure at least one leg meets the minimum RR, and all legs are at least 1.0
        return any(good) and all(leg.rr() >= 1.0 for leg in self.legs)

    def sanity_vs_spot(self, spot: float, buffer_pips: int, pair: str) -> bool:
        """Check semantics vs spot and buffer distance (defensive normalization)."""
        unit = PipMath.pip_unit(pair)
        buf = buffer_pips * unit
        for leg in self.legs:
            t = normalize_order_type(leg.order_type) or leg.order_type  # try normalize; fallback raw
            if t == "Buy Stop":
                if not (leg.entry > spot + buf):
                    return False
            elif t == "Sell Stop":
                if not (leg.entry < spot - buf):
                    return False
            elif t == "Buy Limit":
                if not (leg.entry < spot - buf):
                    return False
            elif t == "Sell Limit":
                if not (leg.entry > spot + buf):
                    return False
            else:
                # Unknown type
                return False
        return True

# ----------------------------- Ledger -----------------------------

class WeeklyLedger:
    """
    JSON-backed ledger of all OCO plans for the current week.
    Keys: pair, status, combo, created_at, updated_at, orders[legs...]
    """
    def __init__(self, path: str, tz: timezone):
        self.path = path
        self.tz = tz
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            self.data = {"orders": []}
            self._save()
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {"orders": []}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def reset_if_new_week(self):
        now = datetime.now(self.tz)
        year, week, _ = now.isocalendar()
        if self.data.get("week") != f"{year}-{week}":
            self.data = {"week": f"{year}-{week}", "orders": []}
            self._save()

    def append_or_update(self, plan_obj: dict):
        """Add new plan or update existing (same pair+created_at)."""
        found = False
        for o in self.data.get("orders", []):
            if o["pair"] == plan_obj["pair"] and o["created_at"] == plan_obj["created_at"]:
                o.update(plan_obj)
                found = True
                break
        if not found:
            self.data["orders"].append(plan_obj)
        self._save()

    def list_entries(self):
        return self.data.get("orders", [])

    def has_order_for_pair_on_date(self, pair: str, date_iso: str) -> bool:
        """Return True if pair already has an order created on date_iso."""
        for o in self.data.get("orders", []):
            if o.get("pair") == pair:
                if o.get("created_at","").startswith(date_iso):
                    return True
        return False

    def mark_statuses(self, pair: str, h1_bars: List[dict], cfg: StrategyConfig):
        """Simplified: update status by comparing entries vs bar highs/lows."""
        for o in self.data.get("orders", []):
            if o["pair"] != pair or o["status"] not in ("pending","accepted"):
                continue
            try:
                entry_prices = [leg["entry"] for leg in o["orders"]]
                highs = [float(b["high"]) for b in h1_bars]
                lows  = [float(b["low"])  for b in h1_bars]
                if any(ep <= max(highs) and ep >= min(lows) for ep in entry_prices):
                    o["status"] = "triggered"
            except Exception:
                continue
        self._save()

# ----------------------------- JSON utils -----------------------------

def plan_to_deterministic_json(plan: OCOPlan) -> dict:
    """Flatten OCOPlan -> stable dict for ledger."""
    return {
        "pair": plan.pair,
        "oco_combo": list(plan.combo),
        "orders": [
            {
                "type": leg.order_type,
                "entry": round(float(leg.entry), 5),
                "sl": round(float(leg.sl), 5),
                "tp": round(float(leg.tp), 5),
                "rr": round(leg.rr(), 2),
            }
            for leg in plan.legs
        ],
        "time_in_force": plan.time_in_force,
        "status": plan.status,
        "confidence": plan.confidence,
        "created_at": plan.created_at,
        "updated_at": plan.updated_at,
        "source": plan.source,
        "decision": plan.decision,
        "reason": plan.reason,
    }

def maybe_reprice_pending(plan_obj: dict, spot: float, cfg: StrategyConfig) -> dict:
    """
    If plan is pending with Limit legs and spot has moved closer, reprice entry within buffer.
    This function now also recalculates the RR after repricing.
    """
    updated = False
    unit = PipMath.pip_unit(plan_obj["pair"])
    buf = cfg.buffer_pips(plan_obj["pair"]) * unit
    for o in plan_obj["orders"]:
        original_entry = o["entry"]
        if "Limit" in o["type"]:
            if o["type"] == "Buy Limit" and spot - o["entry"] > buf:
                o["entry"] = round(spot - buf, 5)
                updated = True
            elif o["type"] == "Sell Limit" and o["entry"] - spot > buf:
                o["entry"] = round(spot + buf, 5)
                updated = True

        # If entry was updated, recalculate RR
        if updated and o["entry"] != original_entry:
            risk = abs(o["entry"] - o["sl"])
            reward = abs(o["tp"] - o["entry"])
            o["rr"] = round(reward / risk, 2) if risk > 0 else 0.0


    if updated:
        plan_obj["updated_at"] = datetime.now(cfg.tz).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        plan_obj["reason"] = plan_obj.get("reason","") + " | repriced closer"

    return plan_obj