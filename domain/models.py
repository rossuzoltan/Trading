from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .enums import ActionType

BAR_DTYPE = np.dtype([
    ('timestamp_s', np.float64),
    ('open', np.float32),
    ('high', np.float32),
    ('low', np.float32),
    ('close', np.float32),
    ('volume', np.float32),
    ('avg_spread', np.float32),
    ('time_delta_s', np.float32),
    ('start_time_msc', np.int64),
    ('end_time_msc', np.int64),
])


@dataclass(frozen=True)
class TickEvent:
    time_msc: int
    bid: float
    ask: float
    volume: float = 1.0

    @property
    def mid_price(self) -> float:
        return (float(self.bid) + float(self.ask)) / 2.0

    @property
    def spread(self) -> float:
        return abs(float(self.ask) - float(self.bid))

    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.time_msc / 1000.0, tz=timezone.utc)


@dataclass(frozen=True)
class VolumeBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    avg_spread: float
    time_delta_s: float
    start_time_msc: int
    end_time_msc: int

    def to_series(self) -> pd.Series:
        return pd.Series(
            {
                "Open": self.open,
                "High": self.high,
                "Low": self.low,
                "Close": self.close,
                "Volume": self.volume,
                "avg_spread": self.avg_spread,
                "time_delta_s": self.time_delta_s,
            },
            name=self.timestamp,
        )


@dataclass
class TickCursor:
    time_msc: int = 0
    offset: int = 0


@dataclass
class BarBuilderState:
    ticks_per_bar: int
    tick_count: int = 0
    bar_open: float | None = None
    bar_high: float | None = None
    bar_low: float | None = None
    spread_total: float = 0.0
    bar_start_time_msc: int | None = None
    last_emitted_bar_start_time_msc: int | None = None


@dataclass(frozen=True)
class BrokerPositionSnapshot:
    symbol: str
    direction: int = 0
    volume: float = 0.0
    entry_price: float | None = None
    entry_reference_price: float | None = None
    entry_bar_index: int | None = None
    sl_price: float | None = None
    tp_price: float | None = None
    broker_ticket: int | None = None
    order_id: int | None = None
    last_confirmed_time_msc: int | None = None
    entry_spread_slippage_cost_usd: float = 0.0
    entry_spread_cost_usd: float = 0.0
    entry_slippage_cost_usd: float = 0.0
    entry_commission_usd: float = 0.0


@dataclass(frozen=True)
class ActionSpec:
    action_type: ActionType
    direction: int | None = None
    sl_value: float | None = None
    tp_value: float | None = None


@dataclass
class ConfirmedPosition:
    direction: int = 0
    entry_price: float | None = None
    sl_price: float | None = None
    tp_price: float | None = None
    volume: float = 0.0
    broker_ticket: int | None = None
    order_id: int | None = None
    time_in_trade_bars: int = 0
    last_reward: float = 0.0
    last_confirmed_price: float | None = None
    last_confirmed_time_msc: int | None = None

    @property
    def is_flat(self) -> bool:
        return self.direction == 0 or self.entry_price is None or self.volume <= 0

    def reset(self) -> None:
        self.direction = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.volume = 0.0
        self.broker_ticket = None
        self.order_id = None
        self.time_in_trade_bars = 0
        self.last_confirmed_price = None
        self.last_confirmed_time_msc = None


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    action: ActionSpec
    volume: float
    submitted_time_msc: int
    requested_price: float
    sl_price: float | None = None
    tp_price: float | None = None
    broker_ticket: int | None = None
    sl_distance_price: float | None = None
    tp_distance_price: float | None = None
    risk_fraction: float | None = None
    lot_size_min: float | None = None
    lot_size_max: float | None = None


@dataclass(frozen=True)
class SubmitResult:
    accepted: bool
    order_id: int | None = None
    error: str | None = None
    retcode: int | None = None
    fill_price: float | None = None


@dataclass
class AccountState:
    equity: float
    balance: float
    used_margin: float
    free_margin: float
    daily_pnl: float
    drawdown: float
