from __future__ import annotations

import os
from datetime import datetime

from typing import TYPE_CHECKING

from domain.models import (
    BrokerPositionSnapshot,
    ConfirmedPosition,
)

if TYPE_CHECKING:
    from runtime.runtime_engine import RuntimeSnapshot




class RiskLimits:
    def __init__(
        self,
        max_drawdown_fraction: float = 0.15,
        daily_loss_fraction: float = 0.05,
        stale_feed_ms: int = 30_000,
        max_broker_failures: int = 3,
        risk_per_trade_fraction: float = 0.01,
        lot_size_min: float = 0.01,
        lot_size_max: float = 0.10,
        safe_mode_on_kill: bool = True,
    ) -> None:
        self.max_drawdown_fraction = float(max_drawdown_fraction)
        self.daily_loss_fraction = float(daily_loss_fraction)
        self.stale_feed_ms = int(stale_feed_ms)
        self.max_broker_failures = int(max_broker_failures)
        self.risk_per_trade_fraction = float(risk_per_trade_fraction)
        self.lot_size_min = float(lot_size_min)
        self.lot_size_max = float(lot_size_max)
        self.safe_mode_on_kill = bool(safe_mode_on_kill)

    @classmethod
    def from_env(cls) -> RiskLimits:
        return cls(
            max_drawdown_fraction=float(os.environ.get("LIVE_MAX_DRAWDOWN_FRACTION", "0.15")),
            daily_loss_fraction=float(os.environ.get("LIVE_DAILY_LOSS_FRACTION", "0.05")),
            stale_feed_ms=int(os.environ.get("LIVE_STALE_FEED_MS", "30000")),
            max_broker_failures=int(os.environ.get("LIVE_MAX_BROKER_FAILURES", "3")),
            risk_per_trade_fraction=float(os.environ.get("LIVE_RISK_PER_TRADE_FRACTION", "0.01")),
            lot_size_min=float(os.environ.get("LIVE_LOT_SIZE_MIN", "0.01")),
            lot_size_max=float(os.environ.get("LIVE_LOT_SIZE_MAX", "0.10")),
            safe_mode_on_kill=os.environ.get("LIVE_SAFE_MODE_ON_KILL", "1") != "0",
        )


class RiskEngine:
    def __init__(self, limits: RiskLimits, *, snapshot: RuntimeSnapshot, initial_equity: float) -> None:
        self.limits = limits
        self.high_water_mark = float(snapshot.high_water_mark or initial_equity)
        self.day_start_equity = float(snapshot.day_start_equity or initial_equity)
        self.last_reset_utc_date = snapshot.last_reset_utc_date
        self.kill_switch_active = bool(snapshot.kill_switch_active)
        self.kill_switch_reason = snapshot.kill_switch_reason
        self.safe_mode_active = bool(snapshot.safe_mode_active)

    def observe_equity(self, equity: float, event_time: datetime) -> tuple[bool, str]:
        date_key = event_time.date().isoformat()
        if self.last_reset_utc_date != date_key:
            self.day_start_equity = float(equity)
            self.last_reset_utc_date = date_key
        self.high_water_mark = max(self.high_water_mark, float(equity))
        drawdown = (self.high_water_mark - equity) / max(self.high_water_mark, 1e-6)
        daily_loss = (self.day_start_equity - equity) / max(self.day_start_equity, 1e-6)
        if drawdown >= self.limits.max_drawdown_fraction:
            return False, f"Max drawdown breached: {drawdown:.2%}"
        if daily_loss >= self.limits.daily_loss_fraction:
            return False, f"Daily loss breached: {daily_loss:.2%}"
        return True, "OK"

    def check_stale_feed(self, *, now_utc: datetime, last_tick_time_msc: int | None) -> tuple[bool, str]:
        if last_tick_time_msc is None:
            return True, "OK"
        stale_ms = int(now_utc.timestamp() * 1000) - int(last_tick_time_msc)
        if stale_ms > self.limits.stale_feed_ms:
            return False, f"Stale feed detected: no ticks for {stale_ms}ms."
        return True, "OK"

    def check_broker_failures(self, consecutive_failures: int) -> tuple[bool, str]:
        if consecutive_failures >= self.limits.max_broker_failures:
            return False, f"Broker failure limit reached: {consecutive_failures}"
        return True, "OK"

    def trigger_kill_switch(self, reason: str) -> None:
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        self.safe_mode_active = bool(self.limits.safe_mode_on_kill)


def sync_confirmed_position(
    target: ConfirmedPosition,
    snapshot: BrokerPositionSnapshot,
    *,
    last_reward: float,
) -> None:
    previous_ticket = target.broker_ticket
    previous_direction = target.direction
    target.direction = int(snapshot.direction)
    target.entry_price = snapshot.entry_price
    target.sl_price = snapshot.sl_price
    target.tp_price = snapshot.tp_price
    target.volume = float(snapshot.volume)
    target.broker_ticket = snapshot.broker_ticket
    target.order_id = snapshot.order_id
    target.last_confirmed_time_msc = snapshot.last_confirmed_time_msc
    target.last_confirmed_price = snapshot.entry_price
    target.last_reward = float(last_reward)
    if snapshot.direction == 0 or snapshot.entry_price is None:
        target.time_in_trade_bars = 0
    elif previous_ticket == snapshot.broker_ticket and previous_direction == snapshot.direction:
        target.time_in_trade_bars += 1
    else:
        target.time_in_trade_bars = 0
