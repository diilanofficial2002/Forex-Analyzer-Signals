# strategy_engine.py
# Core strategy types, pip math, validations, ledger, and helpers.

from __future__ import annotations
import os, json, math
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Iterable
from datetime import datetime, timedelta, timezone

BKK_TZ = timezone(timedelta(hours=7))


# ---------------- Pip math utilities ----------------

class PipMath:
    """Pip helpers for JPY/non-JPY pairs and RR calculation."""
    @staticmethod
    def pip_unit(pair: str) -> float:
        return 0.01 if pair.replace(" ", "").upper().endswith("/JPY") else 0.0001

    @staticmethod
    def rr(entry: float, sl: float, tp: float) -> Optional[float]:
        try:
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            if risk == 0:
                return None
            return reward / risk
        except Exception:
            return None

    @staticmethod
    def to_pips(pair: str, price_delta: float) -> float:
        return price_delta / PipMath.pip_unit(pair)

    @staticmethod
    def from_pips(pair: str, pips: float) -> float:
        return pips * PipMath.pip_unit(pair)


# ---------------- Strategy config ----------------

@dataclass
class StrategyConfig:
    # RR policy
    rr_min: float = 1.0
    rr_max: float = 5.0
    prefer_rr: float = 1.5  # soft preference (NOT rejection)

    # Spot sanity buffer (minimum distance between spot and pending entries)
    buffer_pips_non_jpy: float = 5.0
    buffer_pips_jpy: float = 50.0

    # Pending repricing policy (Limit legs only)
    reprice_threshold_pips_non_jpy: float = 12.0
    reprice_threshold_pips_jpy: float = 120.0
    reprice_nudge_pips_non_jpy: float = 6.0
    reprice_nudge_pips_jpy: float = 60.0

    # Ledger + H1 config
    tz: timezone = BKK_TZ
    ledger_dir: str = "data"
    ledger_file: str = "weekly_ledger.json"
    h1_lookback: int = 200

    # Weekly objective + day profile (Mon..Sun)
    weekly_target_pips: int = 1000
    pairs_per_week: int = 5
    weekday_profile: dict = field(default_factory=lambda: {
        0: {"name": "Mon", "tp_atr_mult": (1.4, 2.0), "plans_per_pair": 1},
        1: {"name": "Tue", "tp_atr_mult": (1.2, 1.8), "plans_per_pair": 1},
        2: {"name": "Wed", "tp_atr_mult": (1.0, 1.6), "plans_per_pair": 1},
        3: {"name": "Thu", "tp_atr_mult": (0.8, 1.2), "plans_per_pair": 1},
        4: {"name": "Fri", "tp_atr_mult": (0.5, 1.0), "plans_per_pair": 2},
        5: {"name": "Sat", "tp_atr_mult": (0.8, 1.2), "plans_per_pair": 0},
        6: {"name": "Sun", "tp_atr_mult": (0.8, 1.2), "plans_per_pair": 0},
    })

    @property
    def ledger_path(self) -> str:
        os.makedirs(self.ledger_dir, exist_ok=True)
        return os.path.join(self.ledger_dir, self.ledger_file)

    def buffer_pips(self, pair: str) -> float:
        return self.buffer_pips_jpy if pair.replace(" ", "").upper().endswith("/JPY") else self.buffer_pips_non_jpy

    def reprice_threshold_pips(self, pair: str) -> float:
        return self.reprice_threshold_pips_jpy if pair.replace(" ", "").upper().endswith("/JPY") else self.reprice_threshold_pips_non_jpy

    def reprice_nudge_pips(self, pair: str) -> float:
        return self.reprice_nudge_pips_jpy if pair.replace(" ", "").upper().endswith("/JPY") else self.reprice_nudge_pips_non_jpy


# ---------------- Order legs & OCO plan ----------------

@dataclass
class OrderLeg:
    order_type: str  # "Buy Limit" | "Sell Limit" | "Buy Stop" | "Sell Stop"
    entry: float
    sl: float
    tp: float

    def rr(self) -> Optional[float]:
        return PipMath.rr(self.entry, self.sl, self.tp)

    def is_buy(self) -> bool:
        return self.order_type.lower().startswith("buy")

    def is_sell(self) -> bool:
        return self.order_type.lower().startswith("sell")


@dataclass
class OCOPlan:
    pair: str
    combo: Tuple[str, str]                 # e.g., ("Buy Limit","Buy Stop")
    legs: Tuple[OrderLeg, OrderLeg]
    time_in_force: Dict                    # e.g., {"type":"weekly","expires_at": "...Z"}
    status: str                            # "pending" | "triggered" | "tp" | "sl" | ...
    confidence: float
    created_at: str
    updated_at: str
    source: Dict = field(default_factory=dict)
    decision: str = "accept"               # "accept" | "reject"
    reason: str = ""

    def validate_combo(self, cfg: StrategyConfig) -> bool:
        allowed = {
            ("Buy Stop", "Sell Stop"),
            ("Buy Limit", "Buy Stop"),
            ("Sell Limit", "Sell Stop"),
            ("Sell Limit", "Buy Limit"),
        }
        return tuple(self.combo) in allowed

    def validate_rr(self, rr_min: float, rr_max: float) -> bool:
        rrs = [leg.rr() for leg in self.legs]
        return all((r is not None) and (rr_min <= r <= rr_max) for r in rrs)

    def sanity_vs_spot(self, spot: float, buffer_pips: float, pair: str) -> bool:
        """Pending semantic check vs spot + basic TP/SL direction sanity."""
        unit = PipMath.pip_unit(pair)
        buf  = buffer_pips * unit
        for leg in self.legs:
            t = leg.order_type.lower().strip()
            # pending semantics vs spot + buffer
            if t == "buy limit" and not (leg.entry < spot - buf): return False
            if t == "sell limit" and not (leg.entry > spot + buf): return False
            if t == "buy stop"  and not (leg.entry > spot + buf): return False
            if t == "sell stop" and not (leg.entry < spot - buf): return False
            # TP/SL directional sanity
            if leg.is_buy() and not (leg.tp > leg.entry and leg.sl < leg.entry): return False
            if leg.is_sell() and not (leg.tp < leg.entry and leg.sl > leg.entry): return False
        return True


# ---------------- JSON helpers ----------------

def plan_to_deterministic_json(plan: OCOPlan) -> Dict:
    """Convert plan to deterministic, audit-friendly JSON shape."""
    orders_json = []
    for leg in plan.legs:
        orders_json.append({
            "type": leg.order_type,
            "entry": round(float(leg.entry), 6),
            "sl":    round(float(leg.sl),    6),
            "tp":    round(float(leg.tp),    6),
            "rr":    round(float(leg.rr() or 0.0), 3),
        })
    return {
        "pair": plan.pair,
        "decision": plan.decision,
        "reason": plan.reason,
        "oco_combo": list(plan.combo),
        "orders": orders_json,
        "time_in_force": plan.time_in_force,
        "status": plan.status,
        "confidence": float(plan.confidence),
        "created_at": plan.created_at,
        "updated_at": plan.updated_at,
        "source": plan.source,
    }


def maybe_reprice_pending(plan_obj: Dict, spot: float, cfg: StrategyConfig) -> Dict:
    """
    Gently reprice LIMIT legs closer to spot when too far, preserving distances to keep RR constant.
    - Only when status == 'pending'
    - Never move a STOP leg
    - Always keep semantics and spot buffer
    """
    if (plan_obj.get("status") or "").lower() != "pending":
        return plan_obj

    pair = plan_obj.get("pair", "EUR/USD")
    unit = PipMath.pip_unit(pair)
    threshold = cfg.reprice_threshold_pips(pair) * unit
    nudge     = cfg.reprice_nudge_pips(pair) * unit
    buf       = cfg.buffer_pips(pair) * unit

    changed = False
    for o in plan_obj.get("orders", []):
        t = (o.get("type") or "").lower().strip()
        if "limit" not in t:
            continue
        entry = float(o["entry"]); sl = float(o["sl"]); tp = float(o["tp"])

        if t == "buy limit":
            too_far = (spot - entry) > threshold
            if too_far:
                delta = min(nudge, max(0.0, (spot - buf) - entry))
                if delta > 0:
                    o["entry"] = round(entry + delta, 6)
                    o["sl"]    = round(sl    + delta, 6)
                    o["tp"]    = round(tp    + delta, 6)
                    changed = True

        elif t == "sell limit":
            too_far = (entry - spot) > threshold
            if too_far:
                delta = min(nudge, max(0.0, entry - (spot + buf)))
                if delta > 0:
                    o["entry"] = round(entry - delta, 6)
                    o["sl"]    = round(sl    - delta, 6)
                    o["tp"]    = round(tp    - delta, 6)
                    changed = True

    if changed:
        plan_obj["reason"] = (plan_obj.get("reason") or "") + " | repriced_limit_closer_to_spot"
    return plan_obj


# ---------------- Ledger ----------------

class WeeklyLedger:
    """
    JSON-backed ledger with week partition (Mon-start).
    Shape:
    {
      "week_start_iso": "YYYY-MM-DD",
      "entries": [ plan_json, ... ]
    }
    """
    def __init__(self, ledger_path: str, tz: timezone):
        self.path = ledger_path
        self.tz = tz
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            self._write({"week_start_iso": self._week_start_iso(), "entries": []})

    # --- basic IO ---
    def _read_safe(self) -> Dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
                if not raw:
                    return {"week_start_iso": self._week_start_iso(), "entries": []}
                return json.loads(raw)
        except Exception:
            return {"week_start_iso": self._week_start_iso(), "entries": []}

    def _write(self, data: Dict):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def _week_start_iso(self, dt: Optional[datetime] = None) -> str:
        dt = dt or datetime.now(self.tz)
        monday = dt - timedelta(days=dt.weekday())
        return monday.date().isoformat()

    # --- public ops ---
    def reset_if_new_week(self):
        data = self._read_safe()
        cur_week = self._week_start_iso()
        if data.get("week_start_iso") != cur_week:
            self._write({"week_start_iso": cur_week, "entries": []})

    def append_or_update(self, plan_obj: Dict):
        data = self._read_safe()
        entries = data.get("entries", [])
        # try to match existing by (pair + created_at) or exact orders signature
        key_pair = plan_obj.get("pair")
        key_created = plan_obj.get("created_at")
        sig = (tuple((o["type"], float(o["entry"]), float(o["sl"]), float(o["tp"])) for o in plan_obj.get("orders", [])))

        idx = -1
        for i, e in enumerate(entries):
            if e.get("pair") == key_pair and e.get("created_at") == key_created:
                idx = i; break
            esig = tuple((o["type"], float(o["entry"]), float(o["sl"]), float(o["tp"])) for o in e.get("orders", []))
            if e.get("pair") == key_pair and esig == sig:
                idx = i; break

        if idx >= 0:
            entries[idx] = plan_obj
        else:
            entries.append(plan_obj)

        data["entries"] = entries
        self._write(data)

    def list_entries(self) -> List[Dict]:
        return self._read_safe().get("entries", [])

    def count_active(self) -> int:
        return sum(1 for e in self.list_entries() if (e.get("status") or "").lower() in ("pending", "triggered"))

    # --- day quota helpers ---
    def has_order_for_pair_on_date(self, pair: str, date_iso: str) -> bool:
        data = self._read_safe()
        for e in data.get("entries", []):
            if e.get("pair") != pair:
                continue
            ts = (e.get("created_at") or "")[:10]
            if ts == date_iso and (e.get("status", "").lower() in ("pending", "executed", "triggered", "tp", "sl")):
                return True
        return False

    def count_orders_for_pair_on_date(self, pair: str, date_iso: str) -> int:
        data = self._read_safe()
        return sum(1 for e in data.get("entries", []) if e.get("pair") == pair and (e.get("created_at") or "")[:10] == date_iso)

    # --- H1 status marking ---
    def mark_statuses(self, pair: str, h1_bars, cfg: StrategyConfig):
        """
        Update pending/triggered entries for `pair` using recent H1 bars DataFrame
        with columns ['from','to','open','high','low','close'] ascending by 'from'.
        """
        if h1_bars is None or len(h1_bars) == 0:
            return
        data = self._read_safe()
        entries = data.get("entries", [])

        # fast index by time
        highs = h1_bars["high"].values.tolist()
        lows  = h1_bars["low"].values.tolist()
        # We assume bars sorted ascending

        def hit_buy_stop(entry):  return any(h >= entry for h in highs)
        def hit_sell_stop(entry): return any(l <= entry for l in lows)
        def hit_buy_limit(entry): return any(l <= entry for l in lows)
        def hit_sell_limit(entry): return any(h >= entry for h in highs)

        def hit_tp_buy(tp):  return any(h >= tp for h in highs)
        def hit_tp_sell(tp): return any(l <= tp for l in lows)
        def hit_sl_buy(sl):  return any(l <= sl for l in lows)
        def hit_sl_sell(sl): return any(h >= sl for h in highs)

        changed = False
        for e in entries:
            if e.get("pair") != pair:
                continue
            status = (e.get("status") or "pending").lower()
            if status not in ("pending", "triggered"):
                continue

            # Evaluate each leg independently; if either leg triggers -> status "triggered"
            triggered = (status == "triggered")
            for o in e.get("orders", []):
                t = (o.get("type") or "").lower().strip()
                entry = float(o["entry"])
                tp    = float(o["tp"])
                sl    = float(o["sl"])

                # Trigger
                if not triggered:
                    if t == "buy stop" and hit_buy_stop(entry): triggered = True
                    if t == "sell stop" and hit_sell_stop(entry): triggered = True
                    if t == "buy limit" and hit_buy_limit(entry): triggered = True
                    if t == "sell limit" and hit_sell_limit(entry): triggered = True

                # If triggered, check TP/SL (simple order: TP first)
                if triggered:
                    if t.startswith("buy"):
                        if hit_tp_buy(tp):
                            e["status"] = "tp"; e["closed_at"] = datetime.now(BKK_TZ).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                            e["pnl_pips"] = round(PipMath.to_pips(pair, tp - entry), 1); changed = True; break
                        if hit_sl_buy(sl):
                            e["status"] = "sl"; e["closed_at"] = datetime.now(BKK_TZ).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                            e["pnl_pips"] = round(PipMath.to_pips(pair, sl - entry), 1); changed = True; break
                    else:
                        if hit_tp_sell(tp):
                            e["status"] = "tp"; e["closed_at"] = datetime.now(BKK_TZ).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                            e["pnl_pips"] = round(PipMath.to_pips(pair, entry - tp), 1); changed = True; break
                        if hit_sl_sell(sl):
                            e["status"] = "sl"; e["closed_at"] = datetime.now(BKK_TZ).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                            e["pnl_pips"] = round(PipMath.to_pips(pair, entry - sl), 1); changed = True; break

            if changed and "status" not in e:
                # If only "triggered" without TP/SL, persist triggered
                e["status"] = "triggered"; changed = True

        if changed:
            data["entries"] = entries
            self._write(data)
