from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from feature_engine import FEATURE_COLS, FeatureEngine
from runtime_common import (
    ActionSpec,
    ActionType,
    ConfirmedPosition,
    build_action_mask,
    build_observation,
)
from symbol_utils import pip_size_for_symbol, pip_value_for_volume, price_to_pips


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


class VolumeBarBuilder:
    def __init__(self, ticks_per_bar: int, state: BarBuilderState | None = None) -> None:
        self.ticks_per_bar = int(ticks_per_bar)
        self._state = state or BarBuilderState(ticks_per_bar=self.ticks_per_bar)

    @property
    def state(self) -> BarBuilderState:
        return self._state

    def push_tick(self, tick: TickEvent) -> VolumeBar | None:
        state = self._state
        mid = tick.mid_price
        if state.tick_count == 0:
            state.bar_open = mid
            state.bar_high = mid
            state.bar_low = mid
            state.spread_total = 0.0
            state.bar_start_time_msc = tick.time_msc

        state.bar_high = max(float(state.bar_high), mid) if state.bar_high is not None else mid
        state.bar_low = min(float(state.bar_low), mid) if state.bar_low is not None else mid
        state.spread_total += tick.spread
        state.tick_count += 1

        if state.tick_count < self.ticks_per_bar:
            return None

        start_time_msc = int(state.bar_start_time_msc or tick.time_msc)
        if state.last_emitted_bar_start_time_msc is None:
            time_delta_s = 0.0
        else:
            time_delta_s = max(0.0, (start_time_msc - state.last_emitted_bar_start_time_msc) / 1000.0)
        bar = VolumeBar(
            timestamp=datetime.fromtimestamp(start_time_msc / 1000.0, tz=timezone.utc),
            open=float(state.bar_open if state.bar_open is not None else mid),
            high=float(state.bar_high if state.bar_high is not None else mid),
            low=float(state.bar_low if state.bar_low is not None else mid),
            close=mid,
            volume=float(state.tick_count),
            avg_spread=float(state.spread_total / max(state.tick_count, 1)),
            time_delta_s=time_delta_s,
            start_time_msc=start_time_msc,
            end_time_msc=tick.time_msc,
        )
        state.last_emitted_bar_start_time_msc = start_time_msc
        state.tick_count = 0
        state.bar_open = None
        state.bar_high = None
        state.bar_low = None
        state.spread_total = 0.0
        state.bar_start_time_msc = None
        return bar


def _as_tick_event(record: Any) -> TickEvent:
    if isinstance(record, Mapping):
        time_msc = int(record.get("time_msc") or int(record["time"]) * 1000)
        bid = float(record["bid"])
        ask = float(record["ask"])
        volume = float(record.get("volume_real", record.get("volume", 1.0)))
        return TickEvent(time_msc=time_msc, bid=bid, ask=ask, volume=volume)
    time_msc = int(getattr(record, "time_msc", getattr(record, "time", 0) * 1000))
    return TickEvent(
        time_msc=time_msc,
        bid=float(getattr(record, "bid")),
        ask=float(getattr(record, "ask")),
        volume=float(getattr(record, "volume_real", getattr(record, "volume", 1.0))),
    )


def _filter_ticks_by_cursor(ticks: Sequence[TickEvent], cursor: TickCursor) -> list[TickEvent]:
    filtered: list[TickEvent] = []
    same_time_seen = 0
    current_time = cursor.time_msc
    for tick in ticks:
        if tick.time_msc < current_time:
            continue
        if tick.time_msc == current_time and same_time_seen < cursor.offset:
            same_time_seen += 1
            continue
        filtered.append(tick)
        if tick.time_msc == current_time:
            same_time_seen += 1
    return filtered


def advance_cursor(cursor: TickCursor, ticks: Sequence[TickEvent]) -> TickCursor:
    if not ticks:
        return TickCursor(time_msc=cursor.time_msc, offset=cursor.offset)
    last_time = ticks[-1].time_msc
    same_time_count = sum(1 for tick in ticks if tick.time_msc == last_time)
    return TickCursor(time_msc=last_time, offset=same_time_count)


class Mt5CursorTickSource:
    def __init__(
        self,
        mt5_module: Any,
        *,
        batch_size: int = 10_000,
        initial_lookback_seconds: int = 10,
    ) -> None:
        self.mt5 = mt5_module
        self.batch_size = int(batch_size)
        self.initial_lookback_seconds = int(initial_lookback_seconds)

    def fetch(self, symbol: str, cursor: TickCursor) -> tuple[list[TickEvent], TickCursor]:
        all_ticks: list[TickEvent] = []
        working_cursor = TickCursor(time_msc=cursor.time_msc, offset=cursor.offset)
        epoch = datetime.fromtimestamp(0, tz=timezone.utc)
        if working_cursor.time_msc > 0:
            start_dt = epoch + timedelta(milliseconds=working_cursor.time_msc)
        else:
            start_dt = datetime.now(timezone.utc) - timedelta(seconds=self.initial_lookback_seconds)

        while True:
            request_count = self.batch_size + max(working_cursor.offset, 0)
            raw = self.mt5.copy_ticks_from(symbol, start_dt, request_count, self.mt5.COPY_TICKS_ALL)
            if raw is None:
                raise RuntimeError("MT5 copy_ticks_from() returned None.")
            if len(raw) == 0:
                break

            converted = sorted((_as_tick_event(record) for record in raw), key=lambda item: item.time_msc)
            new_ticks = _filter_ticks_by_cursor(converted, working_cursor)
            if new_ticks:
                all_ticks.extend(new_ticks)
                working_cursor = advance_cursor(working_cursor, new_ticks)
                start_dt = epoch + timedelta(milliseconds=working_cursor.time_msc)

            if len(raw) < request_count or not new_ticks:
                break

        return all_ticks, working_cursor


@dataclass
class RuntimeSnapshot:
    cursor: TickCursor = field(default_factory=TickCursor)
    bar_builder: BarBuilderState = field(default_factory=lambda: BarBuilderState(ticks_per_bar=0))
    confirmed_position: ConfirmedPosition = field(default_factory=ConfirmedPosition)
    last_equity: float = 0.0
    high_water_mark: float = 0.0
    day_start_equity: float = 0.0
    last_reset_utc_date: str | None = None
    consecutive_broker_failures: int = 0
    last_tick_time_msc: int | None = None
    kill_switch_active: bool = False
    kill_switch_reason: str | None = None
    safe_mode_active: bool = False


class JsonStateStore:
    def __init__(self, path: str | Path, *, ticks_per_bar: int) -> None:
        self.path = Path(path)
        self.ticks_per_bar = int(ticks_per_bar)

    @property
    def backup_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".bak")

    def _load_snapshot_payload(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def load(self) -> RuntimeSnapshot:
        if not self.path.exists():
            return RuntimeSnapshot(bar_builder=BarBuilderState(ticks_per_bar=self.ticks_per_bar))
        raw: dict[str, Any] | None = None
        errors: list[str] = []
        for candidate in (self.path, self.backup_path):
            if not candidate.exists():
                continue
            try:
                raw = self._load_snapshot_payload(candidate)
                break
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                errors.append(f"{candidate}: {exc}")
        if raw is None:
            details = "; ".join(errors) if errors else "no readable state file found"
            raise RuntimeError(f"Unable to load runtime state from {self.path}: {details}")
        return RuntimeSnapshot(
            cursor=TickCursor(**raw.get("cursor", {})),
            bar_builder=BarBuilderState(
                ticks_per_bar=int(raw.get("bar_builder", {}).get("ticks_per_bar", self.ticks_per_bar)),
                tick_count=int(raw.get("bar_builder", {}).get("tick_count", 0)),
                bar_open=raw.get("bar_builder", {}).get("bar_open"),
                bar_high=raw.get("bar_builder", {}).get("bar_high"),
                bar_low=raw.get("bar_builder", {}).get("bar_low"),
                spread_total=float(raw.get("bar_builder", {}).get("spread_total", 0.0)),
                bar_start_time_msc=raw.get("bar_builder", {}).get("bar_start_time_msc"),
                last_emitted_bar_start_time_msc=raw.get("bar_builder", {}).get("last_emitted_bar_start_time_msc"),
            ),
            confirmed_position=ConfirmedPosition(**raw.get("confirmed_position", {})),
            last_equity=float(raw.get("last_equity", 0.0)),
            high_water_mark=float(raw.get("high_water_mark", 0.0)),
            day_start_equity=float(raw.get("day_start_equity", 0.0)),
            last_reset_utc_date=raw.get("last_reset_utc_date"),
            consecutive_broker_failures=int(raw.get("consecutive_broker_failures", 0)),
            last_tick_time_msc=raw.get("last_tick_time_msc"),
            kill_switch_active=bool(raw.get("kill_switch_active", False)),
            kill_switch_reason=raw.get("kill_switch_reason"),
            safe_mode_active=bool(raw.get("safe_mode_active", False)),
        )

    def save(self, snapshot: RuntimeSnapshot) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cursor": asdict(snapshot.cursor),
            "bar_builder": asdict(snapshot.bar_builder),
            "confirmed_position": asdict(snapshot.confirmed_position),
            "last_equity": snapshot.last_equity,
            "high_water_mark": snapshot.high_water_mark,
            "day_start_equity": snapshot.day_start_equity,
            "last_reset_utc_date": snapshot.last_reset_utc_date,
            "consecutive_broker_failures": snapshot.consecutive_broker_failures,
            "last_tick_time_msc": snapshot.last_tick_time_msc,
            "kill_switch_active": snapshot.kill_switch_active,
            "kill_switch_reason": snapshot.kill_switch_reason,
            "safe_mode_active": snapshot.safe_mode_active,
        }
        serialized = json.dumps(payload, indent=2)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(serialized, encoding="utf-8")
        if self.path.exists():
            shutil.copyfile(self.path, self.backup_path)
        temp_path.replace(self.path)


@dataclass(frozen=True)
class BrokerPositionSnapshot:
    symbol: str
    direction: int = 0
    volume: float = 0.0
    entry_price: float | None = None
    sl_price: float | None = None
    tp_price: float | None = None
    broker_ticket: int | None = None
    order_id: int | None = None
    last_confirmed_time_msc: int | None = None


@dataclass(frozen=True)
class SubmitResult:
    accepted: bool
    order_id: int | None = None
    error: str | None = None
    retcode: int | None = None
    fill_price: float | None = None


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


class BaseBroker:
    def advance_bar(self, bar: VolumeBar) -> None:
        return None

    def submit_order(self, intent: OrderIntent) -> SubmitResult:
        raise NotImplementedError

    def current_position(self, symbol: str) -> BrokerPositionSnapshot:
        raise NotImplementedError

    def current_equity(self, symbol: str, mark_price: float | None = None) -> float:
        raise NotImplementedError


class ReplayBroker(BaseBroker):
    def __init__(
        self,
        *,
        symbol: str,
        initial_equity: float = 1_000.0,
        account_currency: str = "USD",
        commission_per_lot: float = 7.0,
        slippage_pips: float = 0.25,
        partial_fill_ratio: float = 1.0,
    ) -> None:
        self.symbol = symbol
        self.account_currency = account_currency
        self.equity = float(initial_equity)
        self.commission_per_lot = float(commission_per_lot)
        self.slippage_pips = float(slippage_pips)
        self.partial_fill_ratio = float(partial_fill_ratio)
        self.next_order_id = 1
        self.next_ticket = 1000
        self._pending: list[OrderIntent] = []
        self._position = BrokerPositionSnapshot(symbol=symbol)
        self.trade_log: list[dict[str, float | int | str]] = []

    def submit_order(self, intent: OrderIntent) -> SubmitResult:
        self._pending.append(intent)
        order_id = self.next_order_id
        self.next_order_id += 1
        return SubmitResult(accepted=True, order_id=order_id)

    def _apply_commission(self, volume: float) -> None:
        self.equity -= self.commission_per_lot * float(volume)

    def _slippage_price(self, direction: int) -> float:
        pip_size = pip_size_for_symbol(self.symbol)
        signed = self.slippage_pips * pip_size
        return signed if direction > 0 else -signed

    def _close_position(self, exit_price: float, reason: str, time_msc: int) -> None:
        if self._position.direction == 0 or self._position.entry_price is None:
            return
        pip_pnl = price_to_pips(self.symbol, exit_price - self._position.entry_price)
        if self._position.direction < 0:
            pip_pnl = -pip_pnl
        pip_value = pip_value_for_volume(
            self.symbol,
            price=exit_price,
            volume_lots=self._position.volume,
            account_currency=self.account_currency,
        )
        pnl = pip_pnl * pip_value
        self.equity += pnl
        self._apply_commission(self._position.volume)
        self.trade_log.append(
            {
                "reason": reason,
                "ticket": int(self._position.broker_ticket or 0),
                "net_pips": float(pip_pnl),
                "equity": float(self.equity),
            }
        )
        self._position = BrokerPositionSnapshot(symbol=self.symbol, last_confirmed_time_msc=time_msc)

    def _fill_pending(self, bar: VolumeBar) -> None:
        if not self._pending:
            return
        intents = self._pending
        self._pending = []
        for intent in intents:
            fill_volume = round(float(intent.volume) * self.partial_fill_ratio, 2)
            if fill_volume <= 0:
                continue
            if intent.action.action_type == ActionType.CLOSE:
                if self._position.direction != 0:
                    close_price = bar.open - self._slippage_price(self._position.direction)
                    self._close_position(close_price, "MANUAL", bar.start_time_msc)
                continue
            if intent.action.action_type != ActionType.OPEN or intent.action.direction is None:
                continue
            if self._position.direction != 0:
                continue
            open_price = bar.open + self._slippage_price(intent.action.direction)
            if intent.action.direction > 0:
                open_price += bar.avg_spread / 2.0
            else:
                open_price -= bar.avg_spread / 2.0
            self._apply_commission(fill_volume)
            self._position = BrokerPositionSnapshot(
                symbol=self.symbol,
                direction=int(intent.action.direction),
                volume=fill_volume,
                entry_price=float(open_price),
                sl_price=intent.sl_price,
                tp_price=intent.tp_price,
                broker_ticket=self.next_ticket,
                order_id=intent.broker_ticket,
                last_confirmed_time_msc=bar.start_time_msc,
            )
            self.next_ticket += 1

    def _mark_stops(self, bar: VolumeBar) -> None:
        position = self._position
        if position.direction == 0 or position.entry_price is None:
            return
        if position.direction > 0:
            if position.sl_price is not None and bar.low <= position.sl_price:
                self._close_position(float(position.sl_price), "SL", bar.end_time_msc)
            elif position.tp_price is not None and bar.high >= position.tp_price:
                self._close_position(float(position.tp_price), "TP", bar.end_time_msc)
        else:
            if position.sl_price is not None and bar.high >= position.sl_price:
                self._close_position(float(position.sl_price), "SL", bar.end_time_msc)
            elif position.tp_price is not None and bar.low <= position.tp_price:
                self._close_position(float(position.tp_price), "TP", bar.end_time_msc)

    def advance_bar(self, bar: VolumeBar) -> None:
        self._fill_pending(bar)
        self._mark_stops(bar)

    def current_position(self, symbol: str) -> BrokerPositionSnapshot:
        if symbol.upper() != self.symbol.upper():
            return BrokerPositionSnapshot(symbol=symbol.upper())
        return self._position

    def current_equity(self, symbol: str, mark_price: float | None = None) -> float:
        if symbol.upper() != self.symbol.upper():
            return self.equity
        if self._position.direction == 0 or self._position.entry_price is None or mark_price is None:
            return self.equity
        pip_pnl = price_to_pips(self.symbol, float(mark_price) - float(self._position.entry_price))
        if self._position.direction < 0:
            pip_pnl = -pip_pnl
        pip_value = pip_value_for_volume(
            self.symbol,
            price=float(mark_price),
            volume_lots=self._position.volume,
            account_currency=self.account_currency,
        )
        return self.equity + pip_pnl * pip_value


class ModelPolicy:
    def __init__(self, model: Any, action_map: Sequence[ActionSpec], obs_normalizer: Any | None = None) -> None:
        self.model = model
        self.action_map = tuple(action_map)
        self.obs_normalizer = obs_normalizer

    def decide(self, observation: np.ndarray, action_mask: np.ndarray) -> tuple[int, ActionSpec]:
        if self.obs_normalizer is not None:
            observation = self.obs_normalizer.normalize_obs(observation)
        action, _ = self.model.predict(observation, action_masks=action_mask, deterministic=True)
        action_index = int(np.asarray(action).item())
        return action_index, self.action_map[action_index]


@dataclass
class RiskLimits:
    max_drawdown_fraction: float = 0.15
    daily_loss_fraction: float = 0.05
    stale_feed_ms: int = 30_000
    max_broker_failures: int = 3
    risk_per_trade_fraction: float = 0.01
    lot_size_min: float = 0.01
    lot_size_max: float = 0.10
    safe_mode_on_kill: bool = True

    @classmethod
    def from_env(cls) -> "RiskLimits":
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


@dataclass
class ProcessResult:
    bar: VolumeBar
    action_index: int
    action: ActionSpec
    features: np.ndarray
    observation: np.ndarray
    reward: float
    equity: float
    position_direction: int
    submit_result: SubmitResult | None
    kill_switch_active: bool = False
    kill_switch_reason: str | None = None
    reward_components: dict[str, float] = field(default_factory=dict)


class RuntimeEngine:
    def __init__(
        self,
        *,
        symbol: str,
        feature_engine: FeatureEngine,
        policy: ModelPolicy,
        broker: BaseBroker,
        action_map: Sequence[ActionSpec],
        risk_engine: RiskEngine,
        state_store: JsonStateStore | None = None,
        snapshot: RuntimeSnapshot | None = None,
        account_currency: str = "USD",
        reward_scale: float = 10_000.0,
        drawdown_penalty: float = 2.0,
        transaction_penalty: float = 1.0,
        reward_clip_low: float = -5.0,
        reward_clip_high: float = 5.0,
    ) -> None:
        self.symbol = symbol.upper()
        self.feature_engine = feature_engine
        self.policy = policy
        self.broker = broker
        self.action_map = tuple(action_map)
        self.risk_engine = risk_engine
        self.state_store = state_store
        self.snapshot = snapshot or RuntimeSnapshot()
        self.account_currency = account_currency
        self.confirmed_position = self.snapshot.confirmed_position
        self.last_equity = float(self.snapshot.last_equity or 0.0)
        self.reward_scale = float(reward_scale)
        self.reward_drawdown_penalty = float(drawdown_penalty)
        self.reward_transaction_penalty = float(transaction_penalty)
        self.reward_clip_low = float(reward_clip_low)
        self.reward_clip_high = float(reward_clip_high)

    def startup_reconcile(self) -> None:
        self._refresh_confirmed_position(last_reward=self.confirmed_position.last_reward)
        if self.state_store is not None:
            self.persist()

    def _refresh_confirmed_position(self, *, last_reward: float) -> None:
        broker_position = self.broker.current_position(self.symbol)
        sync_confirmed_position(self.confirmed_position, broker_position, last_reward=last_reward)

    def _sync_snapshot_risk_state(self) -> None:
        self.snapshot.high_water_mark = self.risk_engine.high_water_mark
        self.snapshot.day_start_equity = self.risk_engine.day_start_equity
        self.snapshot.last_reset_utc_date = self.risk_engine.last_reset_utc_date
        self.snapshot.kill_switch_active = self.risk_engine.kill_switch_active
        self.snapshot.kill_switch_reason = self.risk_engine.kill_switch_reason
        self.snapshot.safe_mode_active = self.risk_engine.safe_mode_active

    def _kill_result(
        self,
        *,
        bar: VolumeBar,
        reward: float,
        equity: float,
        feature_vector: np.ndarray,
        observation: np.ndarray,
        submit_result: SubmitResult | None,
        reason: str,
        reward_components: dict[str, float],
    ) -> ProcessResult:
        self.risk_engine.trigger_kill_switch(reason)
        if not self.confirmed_position.is_flat:
            close_result = self.broker.submit_order(self._build_close_intent(bar))
            submit_result = close_result
            self._refresh_confirmed_position(last_reward=self.confirmed_position.last_reward)
        self.snapshot.last_equity = equity
        self._sync_snapshot_risk_state()
        if self.state_store is not None:
            self.persist()
        action = ActionSpec(ActionType.CLOSE) if not self.confirmed_position.is_flat else ActionSpec(ActionType.HOLD)
        action_index = 1 if action.action_type == ActionType.CLOSE else 0
        return ProcessResult(
            bar=bar,
            action_index=action_index,
            action=action,
            features=feature_vector.copy(),
            observation=observation.copy(),
            reward=float(reward),
            equity=equity,
            position_direction=self.confirmed_position.direction,
            submit_result=submit_result,
            kill_switch_active=True,
            kill_switch_reason=reason,
            reward_components=reward_components,
        )

    def _estimate_transaction_cost_ratio(
        self,
        turnover_lots: float,
        equity_base: float,
        *,
        current_price: float,
        avg_spread: float,
    ) -> float:
        if turnover_lots <= 0:
            return 0.0
        pip_value_per_lot = pip_value_for_volume(
            self.symbol,
            price=float(current_price),
            volume_lots=1.0,
            account_currency=self.account_currency,
        )
        spread_pips = abs(price_to_pips(self.symbol, float(avg_spread)))
        commission_per_lot = float(getattr(self.broker, "commission_per_lot", 0.0))
        slippage_pips = max(float(getattr(self.broker, "slippage_pips", 0.0)), 0.0)
        estimated_cost_usd = turnover_lots * (
            commission_per_lot + pip_value_per_lot * ((spread_pips / 2.0) + slippage_pips)
        )
        return float(estimated_cost_usd / max(equity_base, 1e-6))

    def _calc_reward(
        self,
        equity: float,
        *,
        current_price: float,
        turnover_lots: float = 0.0,
        avg_spread: float = 0.0,
    ) -> float:
        return float(
            self._build_reward_components(
                equity,
                current_price=current_price,
                turnover_lots=turnover_lots,
                avg_spread=avg_spread,
            )["reward_clipped"]
        )

    def _build_reward_components(
        self,
        equity: float,
        *,
        current_price: float,
        turnover_lots: float = 0.0,
        avg_spread: float = 0.0,
    ) -> dict[str, float]:
        if equity <= 0:
            return {
                "log_return": 0.0,
                "reward_unclipped": float(self.reward_clip_low),
                "reward_clipped": float(self.reward_clip_low),
                "drawdown": 1.0,
                "estimated_cost_ratio": 0.0,
                "turnover_lots": float(turnover_lots),
                "drawdown_penalty_applied": 0.0,
                "transaction_penalty_applied": 0.0,
            }
        previous_equity = float(self.last_equity)
        if previous_equity <= 0:
            return {
                "log_return": 0.0,
                "reward_unclipped": 0.0,
                "reward_clipped": 0.0,
                "drawdown": 0.0,
                "estimated_cost_ratio": 0.0,
                "turnover_lots": float(turnover_lots),
                "drawdown_penalty_applied": 0.0,
                "transaction_penalty_applied": 0.0,
            }

        log_return = float(math.log(max(equity, 1e-6) / max(previous_equity, 1e-6)))
        reward_unclipped = self.reward_scale * log_return
        high_water_mark = max(float(self.risk_engine.high_water_mark), float(equity), previous_equity)
        drawdown = max((high_water_mark - float(equity)) / max(high_water_mark, 1e-6), 0.0)

        transaction_cost_ratio = self._estimate_transaction_cost_ratio(
            turnover_lots,
            previous_equity,
            current_price=current_price,
            avg_spread=avg_spread,
        )
        reward_clipped = float(np.clip(reward_unclipped, self.reward_clip_low, self.reward_clip_high))
        return {
            "log_return": log_return,
            "reward_unclipped": float(reward_unclipped),
            "reward_clipped": reward_clipped,
            "drawdown": float(drawdown),
            "estimated_cost_ratio": float(transaction_cost_ratio),
            "turnover_lots": float(turnover_lots),
            "drawdown_penalty_applied": 0.0,
            "transaction_penalty_applied": 0.0,
        }

    def _build_open_intent(self, action: ActionSpec, bar: VolumeBar, equity: float) -> OrderIntent:
        latest = self.feature_engine._buffer.iloc[-1] if self.feature_engine._buffer is not None else pd.Series()
        atr = float(latest.get("atr_14", pip_size_for_symbol(self.symbol) * 20.0))
        if action.direction is None or action.sl_value is None or action.tp_value is None:
            raise RuntimeError("OPEN action missing direction or SL/TP values.")
        sl_distance = float(action.sl_value) * atr
        tp_distance = float(action.tp_value) * atr
        if action.direction > 0:
            sl_price = bar.close - sl_distance
            tp_price = bar.close + tp_distance
        else:
            sl_price = bar.close + sl_distance
            tp_price = bar.close - tp_distance
        sl_pips = max(abs(price_to_pips(self.symbol, bar.close - sl_price)), 1e-6)
        pip_value_per_lot = pip_value_for_volume(
            self.symbol,
            price=bar.close,
            volume_lots=1.0,
            account_currency=self.account_currency,
        )
        raw_lots = (equity * self.risk_engine.limits.risk_per_trade_fraction) / max(sl_pips * pip_value_per_lot, 1e-6)
        volume = round(
            max(self.risk_engine.limits.lot_size_min, min(self.risk_engine.limits.lot_size_max, raw_lots)),
            2,
        )
        return OrderIntent(
            symbol=self.symbol,
            action=action,
            volume=volume,
            submitted_time_msc=bar.end_time_msc,
            requested_price=bar.close,
            sl_price=sl_price,
            tp_price=tp_price,
        )

    def _build_close_intent(self, bar: VolumeBar) -> OrderIntent:
        return OrderIntent(
            symbol=self.symbol,
            action=ActionSpec(ActionType.CLOSE),
            volume=float(self.confirmed_position.volume),
            submitted_time_msc=bar.end_time_msc,
            requested_price=bar.close,
            broker_ticket=self.confirmed_position.broker_ticket,
        )

    def process_bar(self, bar: VolumeBar, *, action_index_override: int | None = None) -> ProcessResult:
        self.broker.advance_bar(bar)
        self.feature_engine.push(bar.to_series())
        feature_vector = self.feature_engine.latest_observation
        latest_row = self.feature_engine._buffer.iloc[-1] if self.feature_engine._buffer is not None else pd.Series()
        spread_z = float(latest_row.get("spread_z", 0.0))
        avg_spread = float(latest_row.get("avg_spread", 0.0))
        pre_action_equity = float(self.broker.current_equity(self.symbol, mark_price=bar.close))
        prev_exposure_lots = abs(float(self.broker.current_position(self.symbol).volume))
        observation = build_observation(
            feature_vector,
            position=self.confirmed_position,
            current_price=bar.close,
            symbol=self.symbol,
            window_size=1,
        )

        mask = build_action_mask(self.action_map, position=self.confirmed_position, spread_z=spread_z)
        if self.risk_engine.kill_switch_active or self.risk_engine.safe_mode_active:
            action = ActionSpec(ActionType.CLOSE) if not self.confirmed_position.is_flat else ActionSpec(ActionType.HOLD)
            action_index = 1 if action.action_type == ActionType.CLOSE else 0
        elif action_index_override is not None:
            requested = int(action_index_override)
            if not (0 <= requested < len(self.action_map)) or not bool(mask[requested]):
                action_index = 0
                action = ActionSpec(ActionType.HOLD)
            else:
                action_index = requested
                action = self.action_map[action_index]
        else:
            action_index, action = self.policy.decide(observation, mask)

        submit_result: SubmitResult | None = None
        if action.action_type == ActionType.OPEN and self.confirmed_position.is_flat:
            submit_result = self.broker.submit_order(self._build_open_intent(action, bar, pre_action_equity))
        elif action.action_type == ActionType.CLOSE and not self.confirmed_position.is_flat:
            submit_result = self.broker.submit_order(self._build_close_intent(bar))
        if submit_result is not None and not submit_result.accepted:
            self.snapshot.consecutive_broker_failures += 1
        else:
            self.snapshot.consecutive_broker_failures = 0

        equity = float(self.broker.current_equity(self.symbol, mark_price=bar.close))
        current_exposure_lots = abs(float(self.broker.current_position(self.symbol).volume))
        turnover_lots = abs(current_exposure_lots - prev_exposure_lots)
        risk_ok, risk_reason = self.risk_engine.observe_equity(equity, bar.timestamp)
        reward_components = self._build_reward_components(
            equity,
            current_price=bar.close,
            turnover_lots=turnover_lots if submit_result is not None and submit_result.accepted else 0.0,
            avg_spread=avg_spread,
        )
        reward = float(reward_components["reward_clipped"])
        self._refresh_confirmed_position(last_reward=reward)
        self.last_equity = equity
        self.snapshot.last_equity = equity

        if not risk_ok:
            return self._kill_result(
                bar=bar,
                reward=reward,
                equity=equity,
                feature_vector=feature_vector,
                observation=observation,
                submit_result=submit_result,
                reason=risk_reason,
                reward_components=reward_components,
            )

        broker_fail_ok, broker_fail_reason = self.risk_engine.check_broker_failures(
            self.snapshot.consecutive_broker_failures
        )
        if not broker_fail_ok:
            return self._kill_result(
                bar=bar,
                reward=reward,
                equity=equity,
                feature_vector=feature_vector,
                observation=observation,
                submit_result=submit_result,
                reason=broker_fail_reason,
                reward_components=reward_components,
            )

        self._sync_snapshot_risk_state()
        if self.state_store is not None:
            self.persist()

        return ProcessResult(
            bar=bar,
            action_index=action_index,
            action=action,
            features=feature_vector.copy(),
            observation=observation.copy(),
            reward=float(reward),
            equity=equity,
            position_direction=self.confirmed_position.direction,
            submit_result=submit_result,
            kill_switch_active=self.risk_engine.kill_switch_active,
            kill_switch_reason=self.risk_engine.kill_switch_reason,
            reward_components=reward_components,
        )

    def flatten_open_position(self, bar: VolumeBar) -> SubmitResult | None:
        self._refresh_confirmed_position(last_reward=self.confirmed_position.last_reward)
        if self.confirmed_position.is_flat:
            return None
        submit_result = self.broker.submit_order(self._build_close_intent(bar))
        if submit_result.accepted:
            self._refresh_confirmed_position(last_reward=self.confirmed_position.last_reward)
        return submit_result

    def persist(self) -> None:
        if self.state_store is None:
            return
        self.snapshot.confirmed_position = self.confirmed_position
        self.state_store.save(self.snapshot)
