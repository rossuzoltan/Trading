from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite


@dataclass(frozen=True)
class SizingResult:
    lots: float
    reason: str


def _round_down_to_step(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    return floor(float(value) / float(step)) * float(step)


def compute_lot_size(
    *,
    equity: float,
    risk_fraction: float,
    stop_distance_pips: float,
    pip_value_per_lot: float,
    lot_min: float,
    lot_max: float,
    lot_step: float,
) -> SizingResult:
    """
    Deterministic position sizing contract (pure function).

    - Uses a fixed-fractional risk budget: equity * risk_fraction.
    - Converts the risk budget into a lot size using stop_distance_pips and pip_value_per_lot.
    - Rounds down to lot_step and clamps to [lot_min, lot_max].

    This module is intentionally not wired into live order creation yet; it exists to
    make the sizing contract explicit and testable before deployment wiring.
    """
    equity = float(equity)
    risk_fraction = float(risk_fraction)
    stop_distance_pips = float(stop_distance_pips)
    pip_value_per_lot = float(pip_value_per_lot)
    lot_min = float(lot_min)
    lot_max = float(lot_max)
    lot_step = float(lot_step)

    if not (isfinite(equity) and equity > 0):
        return SizingResult(lots=0.0, reason="sizing_invalid_equity")
    if not (isfinite(risk_fraction) and risk_fraction > 0):
        return SizingResult(lots=0.0, reason="sizing_invalid_risk_fraction")
    if not (isfinite(stop_distance_pips) and stop_distance_pips > 0):
        return SizingResult(lots=0.0, reason="sizing_invalid_stop_distance")
    if not (isfinite(pip_value_per_lot) and pip_value_per_lot > 0):
        return SizingResult(lots=0.0, reason="sizing_invalid_pip_value")
    if not (isfinite(lot_max) and lot_max > 0):
        return SizingResult(lots=0.0, reason="sizing_invalid_lot_max")
    if not (isfinite(lot_min) and lot_min >= 0):
        return SizingResult(lots=0.0, reason="sizing_invalid_lot_min")
    if lot_min > lot_max:
        return SizingResult(lots=0.0, reason="sizing_invalid_lot_bounds")

    risk_budget_usd = equity * risk_fraction
    risk_per_lot_usd = stop_distance_pips * pip_value_per_lot
    if risk_per_lot_usd <= 0 or not isfinite(risk_per_lot_usd):
        return SizingResult(lots=0.0, reason="sizing_invalid_risk_per_lot")

    raw_lots = risk_budget_usd / risk_per_lot_usd
    if not isfinite(raw_lots) or raw_lots <= 0:
        return SizingResult(lots=0.0, reason="sizing_zero_risk_budget")

    rounded = _round_down_to_step(raw_lots, lot_step)
    clamped = min(max(rounded, lot_min), lot_max)

    if clamped <= 0:
        return SizingResult(lots=0.0, reason="sizing_clamped_to_zero")
    if clamped == lot_min and clamped < raw_lots:
        return SizingResult(lots=clamped, reason="sizing_clamped_to_min")
    if clamped == lot_max and clamped < raw_lots:
        return SizingResult(lots=clamped, reason="sizing_clamped_to_max")
    if clamped < raw_lots:
        return SizingResult(lots=clamped, reason="sizing_rounded_down")
    return SizingResult(lots=clamped, reason="sizing_ok")

