from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _float_attr(obj: Any, name: str) -> float | None:
    if obj is None or not hasattr(obj, name):
        return None
    value = getattr(obj, name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_attr(obj: Any, name: str) -> int | None:
    if obj is None or not hasattr(obj, name):
        return None
    value = getattr(obj, name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_attr(obj: Any, name: str) -> bool | None:
    if obj is None or not hasattr(obj, name):
        return None
    return bool(getattr(obj, name))


def _infer_digits(value: float | None) -> int | None:
    if value is None or value <= 0:
        return None
    rendered = f"{float(value):.10f}".rstrip("0")
    if "." not in rendered:
        return 0
    return len(rendered.split(".", 1)[1])


@dataclass(frozen=True)
class Mt5SymbolCaps:
    symbol: str
    visible: bool | None
    point: float | None
    digits: int | None
    trade_mode: int | None
    trade_stops_level: float | None
    trade_freeze_level: float | None
    volume_min: float | None
    volume_max: float | None
    volume_step: float | None
    tick_size: float | None
    tick_value: float | None
    contract_size: float | None

    @property
    def price_increment(self) -> float | None:
        if self.tick_size is not None and self.tick_size > 0:
            return float(self.tick_size)
        if self.point is not None and self.point > 0:
            return float(self.point)
        return None

    @property
    def price_digits(self) -> int:
        if self.digits is not None and self.digits >= 0:
            return int(self.digits)
        inferred = _infer_digits(self.price_increment)
        if inferred is not None:
            return inferred
        return 8

    @property
    def minimum_stop_distance(self) -> float:
        point = self.point if self.point is not None and self.point > 0 else self.price_increment
        if point is None or point <= 0:
            return 0.0
        stops = float(self.trade_stops_level or 0.0)
        freeze = float(self.trade_freeze_level or 0.0)
        return max(stops, freeze) * point

    def normalize_volume(self, requested_volume: float, *, fallback_step: float = 0.01) -> float:
        requested = float(requested_volume)
        volume_min = float(self.volume_min if self.volume_min not in (None, 0) else requested)
        volume_max = float(self.volume_max if self.volume_max not in (None, 0) else requested)
        volume_step = float(self.volume_step if self.volume_step not in (None, 0) else fallback_step)
        clipped = min(max(requested, volume_min), volume_max)
        if volume_step <= 0:
            return round(clipped, 8)
        steps = round((clipped - volume_min) / volume_step)
        normalized = volume_min + (steps * volume_step)
        return round(max(volume_min, min(volume_max, normalized)), 8)

    def normalize_price(self, price: float | None) -> float | None:
        if price is None:
            return None
        raw = float(price)
        increment = self.price_increment
        if increment is not None and increment > 0:
            raw = round(raw / increment) * increment
        return round(raw, self.price_digits)


def read_symbol_caps(symbol: str, symbol_info: Any) -> Mt5SymbolCaps:
    return Mt5SymbolCaps(
        symbol=symbol.upper(),
        visible=_bool_attr(symbol_info, "visible"),
        point=_float_attr(symbol_info, "point"),
        digits=_int_attr(symbol_info, "digits"),
        trade_mode=_int_attr(symbol_info, "trade_mode"),
        trade_stops_level=_float_attr(symbol_info, "trade_stops_level"),
        trade_freeze_level=_float_attr(symbol_info, "trade_freeze_level"),
        volume_min=_float_attr(symbol_info, "volume_min"),
        volume_max=_float_attr(symbol_info, "volume_max"),
        volume_step=_float_attr(symbol_info, "volume_step"),
        tick_size=_float_attr(symbol_info, "trade_tick_size"),
        tick_value=_float_attr(symbol_info, "trade_tick_value"),
        contract_size=_float_attr(symbol_info, "trade_contract_size"),
    )


def describe_trade_mode(mt5_module: Any, trade_mode: int | None) -> str:
    if trade_mode is None:
        return "unknown"
    candidates = {
        "disabled": getattr(mt5_module, "SYMBOL_TRADE_MODE_DISABLED", None),
        "long_only": getattr(mt5_module, "SYMBOL_TRADE_MODE_LONGONLY", None),
        "short_only": getattr(mt5_module, "SYMBOL_TRADE_MODE_SHORTONLY", None),
        "close_only": getattr(mt5_module, "SYMBOL_TRADE_MODE_CLOSEONLY", None),
        "full": getattr(mt5_module, "SYMBOL_TRADE_MODE_FULL", None),
    }
    for label, value in candidates.items():
        if value is not None and int(value) == int(trade_mode):
            return label
    return f"mode_{int(trade_mode)}"


def trade_mode_allows_open(mt5_module: Any, trade_mode: int | None, direction: int | None) -> bool:
    if trade_mode is None or direction is None:
        return True
    disabled = getattr(mt5_module, "SYMBOL_TRADE_MODE_DISABLED", None)
    close_only = getattr(mt5_module, "SYMBOL_TRADE_MODE_CLOSEONLY", None)
    long_only = getattr(mt5_module, "SYMBOL_TRADE_MODE_LONGONLY", None)
    short_only = getattr(mt5_module, "SYMBOL_TRADE_MODE_SHORTONLY", None)
    if disabled is not None and int(trade_mode) == int(disabled):
        return False
    if close_only is not None and int(trade_mode) == int(close_only):
        return False
    if direction > 0 and short_only is not None and int(trade_mode) == int(short_only):
        return False
    if direction < 0 and long_only is not None and int(trade_mode) == int(long_only):
        return False
    return True
