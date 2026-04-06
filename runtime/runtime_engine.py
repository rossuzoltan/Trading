from __future__ import annotations

import json
import logging
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from domain.enums import ActionType
from domain.models import (
    ActionSpec,
    BarBuilderState,
    BrokerPositionSnapshot,
    ConfirmedPosition,
    OrderIntent,
    SubmitResult,
    TickCursor,
    TickEvent,
    VolumeBar,
)
from execution.broker import BaseBroker
from edge_research import BaselineAlphaGate
from feature_engine import FEATURE_COLS, FeatureEngine
from risk.risk_engine import RiskEngine, sync_confirmed_position
from runtime_common import (
    apply_execution_action_guards,
    build_action_mask,
    build_observation,
)
from symbol_utils import (
    pip_size_for_symbol,
    pip_value_for_volume,
    price_to_pips,
)

logger = logging.getLogger(__name__)
SPREAD_Z_INDEX = FEATURE_COLS.index("spread_z")


@dataclass
class TickCursor:
    time_msc: int = 0
    offset: int = 0


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


class ModelPolicy:
    def __init__(self, model: Any, action_map: Sequence[Any], *, obs_normalizer: Any | None = None) -> None:
        self.model = model
        self.action_map = action_map
        self.obs_normalizer = obs_normalizer

    def decide(self, observation: np.ndarray, mask: np.ndarray) -> tuple[int, Any]:
        obs = observation
        if self.obs_normalizer is not None:
            if hasattr(self.obs_normalizer, "normalize_obs"):
                obs = self.obs_normalizer.normalize_obs(observation)
            else:
                obs = self.obs_normalizer.normalize(observation)
        from sb3_contrib.common.maskable.utils import get_action_masks
        action_idx, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return int(action_idx), self.action_map[int(action_idx)]


@dataclass
class ProcessResult:
    bar: VolumeBar
    action_index: int
    action: Any
    features: np.ndarray
    observation: np.ndarray
    reward: float
    equity: float
    position_direction: int
    submit_result: SubmitResult | None = None
    kill_switch_active: bool = False
    kill_switch_reason: str | None = None
    executed_events: list[dict[str, Any]] = field(default_factory=list)
    closed_trades: list[dict[str, Any]] = field(default_factory=list)
    reward_components: dict[str, float] = field(default_factory=dict)
    policy_mode: str = "RL"


class RuntimeEngine:
    def __init__(
        self,
        *,
        symbol: str,
        feature_engine: FeatureEngine,
        policy: Any,
        broker: BaseBroker,
        action_map: Sequence[Any],
        risk_engine: RiskEngine,
        snapshot: RuntimeSnapshot,
        state_store: JsonStateStore | None = None,
        account_currency: str = "USD",
        reward_scale: float = 10_000.0,
        reward_drawdown_penalty: float = 2.0,
        reward_transaction_penalty: float = 1.0,
        reward_clip_low: float = -5.0,
        reward_clip_high: float = 5.0,
        window_size: int = 1,
        minimal_post_cost_reward: bool = False,
        force_fast_window_benchmark: bool = False,
        alpha_gate: BaselineAlphaGate | None = None,
        churn_min_hold_bars: int = 0,
        churn_action_cooldown: int = 0,
        entry_spread_z_limit: float = 1.5,
        drawdown_penalty: float | None = None,
        transaction_penalty: float | None = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.feature_engine = feature_engine
        self.policy = policy
        self.broker = broker
        self.action_map = action_map
        self.risk_engine = risk_engine
        self.snapshot = snapshot
        self.state_store = state_store
        self.account_currency = account_currency
        if drawdown_penalty is not None:
            reward_drawdown_penalty = float(drawdown_penalty)
        if transaction_penalty is not None:
            reward_transaction_penalty = float(transaction_penalty)
        self.reward_scale = float(reward_scale)
        self.reward_drawdown_penalty = float(reward_drawdown_penalty)
        self.reward_transaction_penalty = float(reward_transaction_penalty)
        self.reward_clip_low = float(reward_clip_low)
        self.reward_clip_high = float(reward_clip_high)
        self.window_size = max(int(window_size), 1)
        self.minimal_post_cost_reward = bool(minimal_post_cost_reward)
        self.force_fast_window_benchmark = bool(force_fast_window_benchmark)
        self.alpha_gate = alpha_gate
        self.churn_min_hold_bars = max(int(churn_min_hold_bars), 0)
        self.churn_action_cooldown = max(int(churn_action_cooldown), 0)
        self.entry_spread_z_limit = float(entry_spread_z_limit)
        self.last_alpha_gate_scores: dict[str, float] | None = None
        self.confirmed_position = snapshot.confirmed_position
        self.last_equity = float(snapshot.last_equity)
        self.processed_bars_count = 0
        self.last_close_bar_index = None
        self.policy_mode = "RL"  # Default to Reinforcement Learning
        self._perf = {
            "process_bar_calls": 0,
            "process_bar_total_ns": 0,
            "push_record_branch_calls": 0,
            "push_series_fallback_calls": 0,
            "window_gt1_calls": 0,
            "force_fast_window_benchmark_calls": 0,
        }
        setattr(self.feature_engine, "_force_fast_window_benchmark", self.force_fast_window_benchmark)

    def startup_reconcile(self) -> None:
        pos_snapshot = self.broker.current_position(self.symbol)
        sync_confirmed_position(self.confirmed_position, pos_snapshot, last_reward=0.0)
        self.last_equity = float(self.broker.current_equity(self.symbol))
        self.snapshot.last_equity = self.last_equity

    def perf_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            key: int(value) if key.endswith("_calls") else float(value)
            for key, value in self._perf.items()
        }
        calls = int(self._perf.get("process_bar_calls", 0))
        total_ns = int(self._perf.get("process_bar_total_ns", 0))
        snapshot["process_bar_mean_ns"] = float(total_ns / calls) if calls else 0.0
        if hasattr(self.feature_engine, "perf_snapshot"):
            snapshot["feature_engine"] = self.feature_engine.perf_snapshot()
        return snapshot

    def _refresh_confirmed_position(self, *, last_reward: float) -> None:
        pos_snapshot = self.broker.current_position(self.symbol)
        sync_confirmed_position(self.confirmed_position, pos_snapshot, last_reward=last_reward)

    def _execution_events_since(self, start_index: int) -> list[dict[str, Any]]:
        execution_log = getattr(self.broker, "execution_log", None)
        if not isinstance(execution_log, list):
            return []
        return [dict(item) for item in execution_log[start_index:] if isinstance(item, dict)]

    def _closed_trades_since(self, start_index: int) -> list[dict[str, Any]]:
        trade_log = getattr(self.broker, "trade_log", None)
        if not isinstance(trade_log, list):
            return []
        return [dict(item) for item in trade_log[start_index:] if isinstance(item, dict)]

    def _sync_snapshot_risk_state(self) -> None:
        self.snapshot.high_water_mark = self.risk_engine.high_water_mark
        self.snapshot.day_start_equity = self.risk_engine.day_start_equity
        self.snapshot.last_reset_utc_date = self.risk_engine.last_reset_utc_date
        self.snapshot.kill_switch_active = self.risk_engine.kill_switch_active
        self.snapshot.kill_switch_reason = self.risk_engine.kill_switch_reason
        self.snapshot.safe_mode_active = self.risk_engine.safe_mode_active

    def _current_equity(self, *, mark_price: float, avg_spread: float) -> float:
        try:
            return float(self.broker.current_equity(self.symbol, mark_price=mark_price, avg_spread=avg_spread))
        except TypeError:
            return float(self.broker.current_equity(self.symbol, mark_price=mark_price))

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
        execution_log_start: int,
        trade_log_start: int,
        reward_components: dict[str, float],
    ) -> ProcessResult:
        self.risk_engine.trigger_kill_switch(reason)
        if not self.confirmed_position.is_flat:
            forced_summary = self.force_flatten(bar, reason="KILL_SWITCH_FORCED_CLOSE")
            equity = float(forced_summary["equity"])
            submit_result = SubmitResult(accepted=True, fill_price=float(bar.close))
        self.snapshot.last_equity = float(equity)
        self.last_equity = float(equity)
        self._sync_snapshot_risk_state()
        if self.state_store is not None:
            self.persist()
        action = ActionSpec(ActionType.CLOSE)
        action_index = 1
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
            executed_events=self._execution_events_since(execution_log_start),
            closed_trades=self._closed_trades_since(trade_log_start),
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
                "reward_raw_unclipped": float(self.reward_clip_low),
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
                "reward_raw_unclipped": 0.0,
                "reward_unclipped": 0.0,
                "reward_clipped": 0.0,
                "drawdown": 0.0,
                "estimated_cost_ratio": 0.0,
                "turnover_lots": float(turnover_lots),
                "drawdown_penalty_applied": 0.0,
                "transaction_penalty_applied": 0.0,
            }

        log_return = float(math.log(max(equity, 1e-6) / max(previous_equity, 1e-6)))
        reward_raw_unclipped = self.reward_scale * log_return
        high_water_mark = max(float(self.risk_engine.high_water_mark), float(equity), previous_equity)
        drawdown = max((high_water_mark - float(equity)) / max(high_water_mark, 1e-6), 0.0)

        transaction_cost_ratio = self._estimate_transaction_cost_ratio(
            turnover_lots,
            previous_equity,
            current_price=current_price,
            avg_spread=avg_spread,
        )
        drawdown_penalty_applied = self.reward_drawdown_penalty * drawdown
        transaction_penalty_applied = self.reward_transaction_penalty * self.reward_scale * transaction_cost_ratio
        reward_unclipped = reward_raw_unclipped - drawdown_penalty_applied - transaction_penalty_applied
        reward_clipped = float(np.clip(reward_unclipped, self.reward_clip_low, self.reward_clip_high))
        return {
            "log_return": log_return,
            "reward_raw_unclipped": float(reward_raw_unclipped),
            "reward_unclipped": float(reward_unclipped),
            "reward_clipped": reward_clipped,
            "drawdown": float(drawdown),
            "estimated_cost_ratio": float(transaction_cost_ratio),
            "turnover_lots": float(turnover_lots),
            "drawdown_penalty_applied": float(drawdown_penalty_applied),
            "transaction_penalty_applied": float(transaction_penalty_applied),
        }

    def _build_open_intent(self, action: ActionSpec, bar: VolumeBar, equity: float) -> OrderIntent:
        latest_aux = getattr(self.feature_engine, "latest_aux_data", {}) or {}
        atr = float(latest_aux.get("atr_14", 0.0) or 0.0)
        if atr <= 0.0 and self.feature_engine._buffer is not None and not self.feature_engine._buffer.empty:
            atr = float(self.feature_engine._buffer.iloc[-1].get("atr_14", 0.0) or 0.0)
        if atr <= 0.0:
            atr = float(pip_size_for_symbol(self.symbol) * 20.0)
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
            sl_distance_price=sl_distance,
            tp_distance_price=tp_distance,
            risk_fraction=self.risk_engine.limits.risk_per_trade_fraction,
            lot_size_min=self.risk_engine.limits.lot_size_min,
            lot_size_max=self.risk_engine.limits.lot_size_max,
        )

    def _build_close_intent(self, bar: Any) -> OrderIntent:
        info = self._extract_bar_info(bar)
        return OrderIntent(
            symbol=self.symbol,
            action=ActionSpec(ActionType.CLOSE),
            volume=float(self.confirmed_position.volume),
            submitted_time_msc=info["end_time_msc"],
            requested_price=info["close"],
            broker_ticket=self.confirmed_position.broker_ticket,
        )

    def _extract_bar_info(self, bar: Any) -> dict[str, Any]:
        """Lazy extraction of bar fields to support both dataclass and numpy records."""
        if hasattr(bar, "close"):
            # Dataclass or BarView
            return {
                "close": float(bar.close),
                "avg_spread": float(bar.avg_spread),
                "end_time_msc": int(bar.end_time_msc),
                "timestamp": getattr(bar, "timestamp", None)
            }
        # Numpy record (np.void)
        return {
            "close": float(bar['close']),
            "avg_spread": float(bar['avg_spread']),
            "end_time_msc": int(bar['end_time_msc']),
            "timestamp": None # Will be derived later if needed
        }

    def process_bar(self, bar: Any, *, action_index_override: int | None = None) -> ProcessResult:
        start_ns = time.perf_counter_ns()
        execution_log_start = len(getattr(self.broker, "execution_log", []))
        trade_log_start = len(getattr(self.broker, "trade_log", []))
        was_flat_before = self.confirmed_position.is_flat
        
        info = self._extract_bar_info(bar)
        close_price = info["close"]
        avg_spread = info["avg_spread"]
        
        executed_turnover_lots = float(self.broker.advance_bar(bar) or 0.0)
        self._refresh_confirmed_position(last_reward=self.confirmed_position.last_reward)
        
        # Use the structured-array fast path only when the bar actually carries a raw record.
        raw_record = getattr(bar, "row", None)
        if hasattr(self.feature_engine, "push_record") and isinstance(raw_record, np.void):
            self._perf["push_record_branch_calls"] += 1
            if self.window_size > 1:
                self._perf["window_gt1_calls"] += 1
            if self.force_fast_window_benchmark:
                self._perf["force_fast_window_benchmark_calls"] += 1
            self.feature_engine.push_record(
                raw_record,
                refresh_buffer=(self.window_size > 1 and not self.force_fast_window_benchmark),
            )
        else:
            self._perf["push_series_fallback_calls"] += 1
            self.feature_engine.push(bar.to_series() if hasattr(bar, "to_series") else bar)

        feature_rows = self.feature_engine.recent_observation_window(self.window_size)
        feature_vector = feature_rows[-1]
        latest_raw_features = getattr(self.feature_engine, "latest_features_raw", np.zeros(len(FEATURE_COLS), dtype=np.float32))
        spread_z = float(latest_raw_features[SPREAD_Z_INDEX]) if len(latest_raw_features) > SPREAD_Z_INDEX else 0.0

        pre_action_equity = self._current_equity(mark_price=close_price, avg_spread=avg_spread)
        prev_exposure_lots = abs(float(self.broker.current_position(self.symbol).volume))
        
        observation = build_observation(
            feature_rows,
            position=self.confirmed_position,
            current_price=close_price,
            symbol=self.symbol,
            window_size=self.window_size,
        )

        mask = build_action_mask(self.action_map, position=self.confirmed_position, spread_z=spread_z)
        mask = apply_execution_action_guards(
            mask,
            position=self.confirmed_position,
            spread_z=spread_z,
            entry_spread_z_limit=self.entry_spread_z_limit,
            churn_min_hold_bars=self.churn_min_hold_bars,
            current_bar_index=self.processed_bars_count,
            last_close_bar_index=self.last_close_bar_index,
            churn_action_cooldown=self.churn_action_cooldown,
        )
        self.last_alpha_gate_scores = None
        if self.confirmed_position.is_flat and self.alpha_gate is not None:
            feature_row = (
                {col: float(latest_raw_features[idx]) for idx, col in enumerate(FEATURE_COLS)}
                if len(latest_raw_features) == len(FEATURE_COLS)
                else (self.feature_engine._buffer.iloc[-1] if self.feature_engine._buffer is not None else pd.Series())
            )
            allow_long, allow_short, scores = self.alpha_gate.allowed_directions(feature_row)
            self.last_alpha_gate_scores = dict(scores)
            for idx, candidate in enumerate(self.action_map):
                if candidate.action_type != ActionType.OPEN:
                    continue
                direction = int(candidate.direction or 0)
                if direction > 0 and not allow_long:
                    mask[idx] = False
                if direction < 0 and not allow_short:
                    mask[idx] = False
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
            
        used_policy_mode = self.policy_mode

        submit_result: SubmitResult | None = None
        if action.action_type == ActionType.OPEN and self.confirmed_position.is_flat:
            submit_result = self.broker.submit_order(self._build_open_intent(action, bar, pre_action_equity))
        elif action.action_type == ActionType.CLOSE and not self.confirmed_position.is_flat:
            submit_result = self.broker.submit_order(self._build_close_intent(bar))
        if submit_result is not None and not submit_result.accepted:
            self.snapshot.consecutive_broker_failures += 1
        else:
            self.snapshot.consecutive_broker_failures = 0

        equity = self._current_equity(mark_price=close_price, avg_spread=avg_spread)
        current_exposure_lots = abs(float(self.broker.current_position(self.symbol).volume))
        turnover_lots = float(executed_turnover_lots + abs(current_exposure_lots - prev_exposure_lots))
        risk_ok, risk_reason = self.risk_engine.observe_equity(equity, bar.timestamp)
        reward_components = self._build_reward_components(
            equity,
            current_price=bar.close,
            turnover_lots=turnover_lots,
            avg_spread=avg_spread,
        )
        reward = float(equity - self.last_equity) if self.minimal_post_cost_reward else float(reward_components["reward_clipped"])
        if self.minimal_post_cost_reward:
            reward_components["reward_raw_unclipped"] = float(reward)
            reward_components["reward_unclipped"] = float(reward)
            reward_components["reward_clipped"] = float(reward)
        self._refresh_confirmed_position(last_reward=reward)
        if not was_flat_before and self.confirmed_position.is_flat:
            self.last_close_bar_index = self.processed_bars_count
        self.processed_bars_count += 1
        self.last_equity = equity
        self.snapshot.last_equity = equity

        if not risk_ok:
            result = self._kill_result(
                bar=bar,
                reward=reward,
                equity=equity,
                feature_vector=feature_vector,
                observation=observation,
                submit_result=submit_result,
                reason=risk_reason,
                execution_log_start=execution_log_start,
                trade_log_start=trade_log_start,
                reward_components=reward_components,
            )
            self._perf["process_bar_calls"] += 1
            self._perf["process_bar_total_ns"] += max(time.perf_counter_ns() - start_ns, 0)
            return result

        broker_fail_ok, broker_fail_reason = self.risk_engine.check_broker_failures(
            self.snapshot.consecutive_broker_failures
        )
        if not broker_fail_ok:
            result = self._kill_result(
                bar=bar,
                reward=reward,
                equity=equity,
                feature_vector=feature_vector,
                observation=observation,
                submit_result=submit_result,
                reason=broker_fail_reason,
                execution_log_start=execution_log_start,
                trade_log_start=trade_log_start,
                reward_components=reward_components,
            )
            self._perf["process_bar_calls"] += 1
            self._perf["process_bar_total_ns"] += max(time.perf_counter_ns() - start_ns, 0)
            return result

        self._sync_snapshot_risk_state()
        if self.state_store is not None:
            self.persist()

        result = ProcessResult(
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
            executed_events=self._execution_events_since(execution_log_start),
            closed_trades=self._closed_trades_since(trade_log_start),
            reward_components=reward_components,
            policy_mode=used_policy_mode,
        )
        self._perf["process_bar_calls"] += 1
        self._perf["process_bar_total_ns"] += max(time.perf_counter_ns() - start_ns, 0)
        return result

    def flatten_open_position(self, bar: VolumeBar) -> SubmitResult | None:
        self._refresh_confirmed_position(last_reward=self.confirmed_position.last_reward)
        if self.confirmed_position.is_flat:
            return None
        submit_result = self.broker.submit_order(self._build_close_intent(bar))
        if submit_result.accepted:
            self._refresh_confirmed_position(last_reward=self.confirmed_position.last_reward)
        return submit_result

    def force_flatten(
        self,
        bar: VolumeBar,
        *,
        reason: str = "FORCED_END_OF_PATH",
    ) -> dict[str, float | bool]:
        self._refresh_confirmed_position(last_reward=self.confirmed_position.last_reward)
        if self.confirmed_position.is_flat:
            return {
                "forced_close": False,
                "turnover_lots": 0.0,
                "equity": float(self._current_equity(mark_price=bar.close, avg_spread=bar.avg_spread)),
            }
        turnover_lots = 0.0
        broker_force_flatten = getattr(self.broker, "force_flatten", None)
        if callable(broker_force_flatten):
            turnover_lots = float(broker_force_flatten(bar, reason=reason) or 0.0)
        else:
            submit_result = self.broker.submit_order(self._build_close_intent(bar))
            if submit_result.accepted:
                turnover_lots = float(self.confirmed_position.volume)
        equity = float(self._current_equity(mark_price=bar.close, avg_spread=bar.avg_spread))
        self._refresh_confirmed_position(last_reward=self.confirmed_position.last_reward)
        self.last_equity = equity
        self.snapshot.last_equity = equity
        self._sync_snapshot_risk_state()
        return {
            "forced_close": bool(turnover_lots > 0.0),
            "turnover_lots": float(turnover_lots),
            "equity": equity,
        }

    def persist(self) -> None:
        if self.state_store is None:
            return
        self.snapshot.confirmed_position = self.confirmed_position
        self.state_store.save(self.snapshot)

    def get_training_diagnostics(self) -> dict[str, Any]:
        """Provides internal metrics for centralized diagnostic aggregation."""
        return {
            "processed_bars": int(self.processed_bars_count),
            "last_close_bar_index": (
                int(self.last_close_bar_index) if self.last_close_bar_index is not None else None
            ),
            "risk_kill_switch": bool(self.risk_engine.kill_switch_active),
            "confirmed_position_type": int(self.confirmed_position.direction) if self.confirmed_position else 0,
            "confirmed_position_volume": float(self.confirmed_position.volume if self.confirmed_position else 0.0),
            "alpha_gate": {
                "enabled": bool(self.alpha_gate is not None),
                "model_kind": getattr(self.alpha_gate, "model_kind", None),
                "last_scores": dict(self.last_alpha_gate_scores or {}),
            },
        }
