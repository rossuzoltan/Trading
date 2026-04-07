from __future__ import annotations

import copy
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM = True
except ImportError:
    import gym  # type: ignore
    from gym import spaces  # type: ignore
    _GYM = False

from domain.models import VolumeBar, BAR_DTYPE
from execution.replay_broker import ReplayBroker
from risk.risk_engine import RiskEngine, RiskLimits
from runtime.runtime_engine import RuntimeEngine, RuntimeSnapshot

from edge_research import BaselineAlphaGate
from feature_engine import FEATURE_COLS, FeatureEngine, WARMUP_BARS
from runtime_common import (
    STATE_FEATURE_COUNT,
    ActionSpec,
    ActionType,
    action_label,
    apply_execution_action_guards,
    build_action_mask,
    build_observation,
)
from symbol_utils import pip_value_for_volume



class BarView:
    __slots__ = ['row', '_ts']
    
    def __init__(self, row: np.void):
        self.row = row
        self._ts = None

    @property
    def timestamp(self):
        if self._ts is None:
            self._ts = pd.Timestamp(self.row['timestamp_s'], unit='s', tz='UTC').to_pydatetime()
        return self._ts
    
    @property
    def open(self): return float(self.row['open'])
    @property
    def high(self): return float(self.row['high'])
    @property
    def low(self): return float(self.row['low'])
    @property
    def close(self): return float(self.row['close'])
    @property
    def volume(self): return float(self.row['volume'])
    @property
    def avg_spread(self): return float(self.row['avg_spread'])
    @property
    def time_delta_s(self): return float(self.row['time_delta_s'])
    @property
    def start_time_msc(self): return int(self.row['start_time_msc'])
    @property
    def end_time_msc(self): return int(self.row['end_time_msc'])

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


@dataclass(frozen=True)
class RuntimeGymConfig:
    initial_equity: float = 1_000.0
    commission_per_lot: float = 7.0
    slippage_pips: float = 0.25
    partial_fill_ratio: float = 1.0
    reward_scale: float = 10_000.0
    drawdown_penalty: float = 2.0
    transaction_penalty: float = 1.0
    reward_clip_low: float = -5.0
    reward_clip_high: float = 5.0
    churn_min_hold_bars: int = 0
    churn_action_cooldown: int = 0
    churn_penalty_usd: float = 0.0
    downside_risk_penalty: float = 0.0
    turnover_penalty: float = 0.0
    net_return_coef: float = 1.0
    entry_spread_z_limit: float = 1.5
    alpha_gate_warmup_steps: int = 0
    alpha_gate_warmup_threshold_delta: float = 0.0
    alpha_gate_warmup_margin_scale: float = 1.0
    window_size: int = 1
    minimal_post_cost_reward: bool = True
    force_fast_window_benchmark: bool = False
    random_start: bool = False
    slim_info: bool = False


def compute_participation_bonus(
    *,
    prev_position: int,
    new_position: int,
    global_step: int,
    episode_bonus_count: int,
    last_bonus_step: int,
    cfg: dict[str, Any] | None,
) -> float:
    if not cfg:
        return 0.0
    pcfg = cfg.get("participation_bonus", {}) or {}
    if not bool(pcfg.get("enabled", False)):
        return 0.0
    if int(global_step) > int(pcfg.get("active_until_step", 0)):
        return 0.0
    if int(episode_bonus_count) >= int(pcfg.get("max_bonus_per_episode", 0)):
        return 0.0
    if int(global_step) - int(last_bonus_step) < int(pcfg.get("cooldown_steps", 0)):
        return 0.0

    mode = str(pcfg.get("mode", "entry")).lower()
    if mode == "per_bar":
        if int(new_position) != 0:
            return float(pcfg.get("bonus_value", 0.0))
        return 0.0

    entry_happened = int(prev_position) == 0 and int(new_position) != 0
    if bool(pcfg.get("only_from_flat", True)) and not entry_happened:
        return 0.0
    if not entry_happened:
        return 0.0
    return float(pcfg.get("bonus_value", 0.0))


def compose_final_reward(
    *,
    base_reward_unclipped: float,
    net_return_coef: float,
    turnover_penalty: float,
    downside_risk_penalty: float,
    rapid_reversal_penalty: float,
    holding_penalty: float,
    participation_bonus: float,
    clip_low: float,
    clip_high: float,
) -> tuple[float, dict[str, float]]:
    net_return_adjustment = float(base_reward_unclipped) * (float(net_return_coef) - 1.0)
    pre_bonus_reward = (
        float(base_reward_unclipped)
        + float(net_return_adjustment)
        - float(turnover_penalty)
        - float(downside_risk_penalty)
        - float(rapid_reversal_penalty)
        - float(holding_penalty)
    )
    unclipped_final_reward = float(pre_bonus_reward) + float(participation_bonus)
    final_reward = float(np.clip(unclipped_final_reward, float(clip_low), float(clip_high)))
    return final_reward, {
        "net_return_adjustment_applied": float(net_return_adjustment),
        "pre_bonus_reward": float(pre_bonus_reward),
        "final_reward_unclipped": float(unclipped_final_reward),
        "final_reward_clipped_low": float(unclipped_final_reward <= float(clip_low)),
        "final_reward_clipped_high": float(unclipped_final_reward >= float(clip_high)),
    }


class TrainingDiagnostics:
    RAPID_REVERSAL_WINDOW_STEPS = 3

    def __init__(self) -> None:
        self.action_counts: dict[str, int] = {}
        self.trade_stats: dict[str, float | int] = {}
        self.economic_stats: dict[str, float] = {}
        self.reward_stats: dict[str, float] = {}
        self._position_duration_samples: deque[int] = deque(maxlen=2048)
        self.reset()

    def reset(self) -> None:
        self.total_steps = 0
        self.action_counts = {
            "hold": 0,
            "close": 0,
            "long": 0,
            "short": 0,
        }
        self.trade_stats = {
            "action_selected_count": 0,
            "action_accepted_count": 0,
            "accepted_open_count": 0,
            "accepted_close_count": 0,
            "order_executed_count": 0,
            "executed_open_count": 0,
            "executed_close_count": 0,
            "entered_long_count": 0,
            "entered_short_count": 0,
            "entry_signal_long_count": 0,
            "entry_signal_short_count": 0,
            "closed_trade_count": 0,
            "trade_attempt_count": 0,
            "trade_reject_count": 0,
            "forced_close_count": 0,
            "flat_steps": 0,
            "long_steps": 0,
            "short_steps": 0,
            "position_duration_sum": 0,
            "position_duration_count": 0,
            "rapid_reversals": 0,
            "mask_observation_count": 0,
            "alpha_gate_observation_count": 0,
            "alpha_gate_long_allowed_steps": 0,
            "alpha_gate_short_allowed_steps": 0,
            "alpha_gate_block_all_steps": 0,
            "entry_spread_blocked_steps": 0,
            "cooldown_blocked_steps": 0,
            "min_hold_forced_hold_steps": 0,
        }
        self.economic_stats = {
            "gross_pnl_usd": 0.0,
            "net_pnl_usd": 0.0,
            "transaction_cost_usd": 0.0,
            "commission_usd": 0.0,
            "spread_slippage_cost_usd": 0.0,
            "spread_cost_usd": 0.0,
            "slippage_cost_usd": 0.0,
        }
        self.reward_stats = {
            "pnl_reward_sum": 0.0,
            "slippage_penalty_sum": 0.0,
            "participation_bonus_sum": 0.0,
            "holding_penalty_sum": 0.0,
            "drawdown_penalty_sum": 0.0,
            "turnover_penalty_sum": 0.0,
            "downside_risk_penalty_sum": 0.0,
            "rapid_reversal_penalty_sum": 0.0,
            "net_return_adjustment_sum": 0.0,
            "final_reward_clipped_low_count": 0.0,
            "final_reward_clipped_high_count": 0.0,
            "net_reward_sum": 0.0,
        }
        self._position_duration_samples.clear()
        self.reset_episode_state()

    def reset_episode_state(self) -> None:
        self._episode_bonus_count = 0
        self._last_bonus_step = -(10**9)
        self._last_close_step: int | None = None
        self._last_closed_direction = 0

    @property
    def episode_bonus_count(self) -> int:
        return int(self._episode_bonus_count)

    @property
    def last_bonus_step(self) -> int:
        return int(self._last_bonus_step)

    def mark_bonus_awarded(self, global_step: int) -> None:
        self._last_bonus_step = int(global_step)
        self._episode_bonus_count += 1

    def record_mask_observation(
        self,
        *,
        spread_blocked: bool,
        cooldown_blocked: bool,
        min_hold_blocked: bool,
        alpha_gate_enabled: bool,
        alpha_gate_allow_long: bool | None = None,
        alpha_gate_allow_short: bool | None = None,
    ) -> None:
        self.trade_stats["mask_observation_count"] += 1
        if spread_blocked:
            self.trade_stats["entry_spread_blocked_steps"] += 1
        if cooldown_blocked:
            self.trade_stats["cooldown_blocked_steps"] += 1
        if min_hold_blocked:
            self.trade_stats["min_hold_forced_hold_steps"] += 1
        if not alpha_gate_enabled:
            return
        self.trade_stats["alpha_gate_observation_count"] += 1
        if bool(alpha_gate_allow_long):
            self.trade_stats["alpha_gate_long_allowed_steps"] += 1
        if bool(alpha_gate_allow_short):
            self.trade_stats["alpha_gate_short_allowed_steps"] += 1
        if alpha_gate_allow_long is False and alpha_gate_allow_short is False:
            self.trade_stats["alpha_gate_block_all_steps"] += 1

    def is_rapid_reversal_candidate(self, entry_filled_direction: int) -> bool:
        direction = int(entry_filled_direction)
        if direction == 0 or self._last_close_step is None:
            return False
        if (int(self.total_steps) - int(self._last_close_step)) > self.RAPID_REVERSAL_WINDOW_STEPS:
            return False
        return (direction > 0 and self._last_closed_direction < 0) or (
            direction < 0 and self._last_closed_direction > 0
        )

    def _record_action(self, action: ActionSpec) -> None:
        if action.action_type == ActionType.HOLD:
            self.action_counts["hold"] += 1
            return
        if action.action_type == ActionType.CLOSE:
            self.action_counts["close"] += 1
            return
        if action.direction is not None and int(action.direction) > 0:
            self.action_counts["long"] += 1
        elif action.direction is not None and int(action.direction) < 0:
            self.action_counts["short"] += 1

    def _record_executed_events(self, executed_events: Sequence[dict[str, Any]] | None) -> None:
        for event in list(executed_events or []):
            self.trade_stats["order_executed_count"] += 1
            side = str(event.get("side", "")).lower()
            if side == "open":
                self.trade_stats["executed_open_count"] += 1
            elif side == "close":
                self.trade_stats["executed_close_count"] += 1

    def _record_closed_trades(
        self,
        *,
        closed_trades: Sequence[dict[str, Any]] | None,
        prev_position: int,
        prev_position_duration: int,
        entry_filled_direction: int,
    ) -> None:
        closed_trades = list(closed_trades or [])
        if not closed_trades:
            return
        self.trade_stats["closed_trade_count"] += int(len(closed_trades))
        self.trade_stats["forced_close_count"] += int(
            sum(1 for trade in closed_trades if bool(trade.get("forced_close", False)))
        )
        close_direction = int(prev_position) if int(prev_position) != 0 else int(entry_filled_direction)
        duration = 0
        if int(prev_position) != 0:
            duration = max(int(prev_position_duration), 1)
        elif int(entry_filled_direction) != 0:
            duration = 1
        if duration > 0:
            self.trade_stats["position_duration_sum"] += duration
            self.trade_stats["position_duration_count"] += 1
            self._position_duration_samples.append(duration)
        if close_direction != 0:
            self._last_close_step = int(self.total_steps)
            self._last_closed_direction = close_direction
            for trade in closed_trades:
                self.economic_stats["gross_pnl_usd"] += float(trade.get("gross_pnl_usd", 0.0))
                self.economic_stats["net_pnl_usd"] += float(trade.get("net_pnl_usd", 0.0))
                self.economic_stats["transaction_cost_usd"] += float(trade.get("transaction_cost_usd", 0.0))
                self.economic_stats["commission_usd"] += float(trade.get("commission_usd", 0.0))
                self.economic_stats["spread_slippage_cost_usd"] += float(trade.get("spread_slippage_cost_usd", 0.0))
                self.economic_stats["spread_cost_usd"] += float(trade.get("spread_cost_usd", 0.0))
                self.economic_stats["slippage_cost_usd"] += float(trade.get("slippage_cost_usd", 0.0))

    def record_step(
        self,
        *,
        action: ActionSpec,
        submit_result: Any,
        prev_position: int,
        new_position: int,
        prev_position_duration: int,
        entry_signal_direction: int = 0,
        entry_filled_direction: int = 0,
        executed_events: Sequence[dict[str, Any]] | None = None,
        closed_trades: Sequence[dict[str, Any]] | None = None,
        reward_components: dict[str, float],
        reward: float,
    ) -> None:
        self.total_steps += 1
        
        # Fast-path for most common event: HOLD with no trades/events
        # We only need to track steps and positions
        if action.action_type == ActionType.HOLD and not executed_events and not closed_trades:
            self.action_counts["hold"] += 1
            if new_position == 0: self.trade_stats["flat_steps"] += 1
            elif new_position > 0: self.trade_stats["long_steps"] += 1
            else: self.trade_stats["short_steps"] += 1
            self.trade_stats["action_selected_count"] += 1
            
            # Record rewards
            rstats = self.reward_stats
            rstats["pnl_reward_sum"] += reward_components.get("pnl_reward", 0.0)
            rstats["net_reward_sum"] += reward
            return

        # Slow path for actions/events
        self.trade_stats["action_selected_count"] += 1
        self._record_action(action)
        if action.action_type != ActionType.HOLD:
            self.trade_stats["trade_attempt_count"] += 1
        
        if submit_result is not None and bool(getattr(submit_result, "accepted", False)):
            self.trade_stats["action_accepted_count"] += 1
            at = action.action_type
            if at == ActionType.CLOSE:
                self.trade_stats["accepted_close_count"] += 1
            elif at == ActionType.OPEN:
                self.trade_stats["accepted_open_count"] += 1
        elif submit_result is not None:
            self.trade_stats["trade_reject_count"] += 1

        if entry_signal_direction > 0:
            self.trade_stats["entry_signal_long_count"] += 1
        elif entry_signal_direction < 0:
            self.trade_stats["entry_signal_short_count"] += 1

        if entry_filled_direction > 0:
            self.trade_stats["entered_long_count"] += 1
            if self._last_closed_direction < 0 and self._last_close_step is not None:
                if (self.total_steps - self._last_close_step) <= self.RAPID_REVERSAL_WINDOW_STEPS:
                    self.trade_stats["rapid_reversals"] += 1
        elif int(entry_filled_direction) < 0:
            self.trade_stats["entered_short_count"] += 1
            if (
                self._last_closed_direction > 0
                and self._last_close_step is not None
                and (self.total_steps - self._last_close_step) <= self.RAPID_REVERSAL_WINDOW_STEPS
            ):
                self.trade_stats["rapid_reversals"] += 1

        self._record_executed_events(executed_events)
        self._record_closed_trades(
            closed_trades=closed_trades,
            prev_position=prev_position,
            prev_position_duration=prev_position_duration,
            entry_filled_direction=entry_filled_direction,
        )

        if int(new_position) == 0:
            self.trade_stats["flat_steps"] += 1
        elif int(new_position) > 0:
            self.trade_stats["long_steps"] += 1
        else:
            self.trade_stats["short_steps"] += 1

        self.reward_stats["pnl_reward_sum"] += float(
            reward_components.get("pnl_reward", reward_components.get("reward_raw_unclipped", 0.0))
        )
        self.reward_stats["slippage_penalty_sum"] -= float(reward_components.get("slippage_penalty_applied", 0.0))
        self.reward_stats["participation_bonus_sum"] += float(
            reward_components.get("participation_bonus_applied", 0.0)
        )
        self.reward_stats["holding_penalty_sum"] -= float(reward_components.get("holding_penalty_applied", 0.0))
        self.reward_stats["drawdown_penalty_sum"] -= float(reward_components.get("drawdown_penalty_applied", 0.0))
        self.reward_stats["turnover_penalty_sum"] -= float(reward_components.get("turnover_penalty_applied", 0.0))
        self.reward_stats["downside_risk_penalty_sum"] -= float(
            reward_components.get("downside_risk_penalty_applied", 0.0)
        )
        self.reward_stats["rapid_reversal_penalty_sum"] -= float(
            reward_components.get("rapid_reversal_penalty_applied", 0.0)
        )
        self.reward_stats["net_return_adjustment_sum"] += float(
            reward_components.get("net_return_adjustment_applied", 0.0)
        )
        self.reward_stats["final_reward_clipped_low_count"] += float(
            reward_components.get("final_reward_clipped_low", 0.0)
        )
        self.reward_stats["final_reward_clipped_high_count"] += float(
            reward_components.get("final_reward_clipped_high", 0.0)
        )
        self.reward_stats["net_reward_sum"] += float(reward)

    def record_forced_close(
        self,
        *,
        prev_position: int,
        prev_position_duration: int,
        executed_events: Sequence[dict[str, Any]] | None,
        closed_trades: Sequence[dict[str, Any]] | None,
    ) -> None:
        self._record_executed_events(executed_events)
        self._record_closed_trades(
            closed_trades=closed_trades,
            prev_position=prev_position,
            prev_position_duration=prev_position_duration,
            entry_filled_direction=0,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_steps": int(self.total_steps),
            "action_counts": {key: int(value) for key, value in self.action_counts.items()},
            "trade_stats": {
                **{key: int(value) for key, value in self.trade_stats.items() if key not in {"position_duration_sum"}},
                "position_duration_sum": float(self.trade_stats["position_duration_sum"]),
                "position_durations_sample": list(self._position_duration_samples),
            },
            "economics": {key: float(value) for key, value in self.economic_stats.items()},
            "reward_components": {key: float(value) for key, value in self.reward_stats.items()},
        }

    def snapshot_delta(self, baseline: dict[str, Any] | None) -> dict[str, Any]:
        current = self.snapshot()
        if not isinstance(baseline, dict):
            return current

        def _delta_map(
            current_map: dict[str, Any] | None,
            baseline_map: dict[str, Any] | None,
            *,
            float_keys: set[str] | None = None,
        ) -> dict[str, Any]:
            out: dict[str, Any] = {}
            current_map = dict(current_map or {})
            baseline_map = dict(baseline_map or {})
            keys = set(current_map) | set(baseline_map)
            for key in keys:
                if key == "position_durations_sample":
                    current_sample = list(current_map.get(key, []) or [])
                    baseline_sample = list(baseline_map.get(key, []) or [])
                    if len(current_sample) >= len(baseline_sample):
                        out[key] = current_sample[len(baseline_sample):]
                    else:
                        out[key] = current_sample
                    continue
                current_value = current_map.get(key, 0.0)
                baseline_value = baseline_map.get(key, 0.0)
                if float_keys and key in float_keys:
                    out[key] = float(current_value) - float(baseline_value)
                else:
                    out[key] = int(round(float(current_value) - float(baseline_value)))
            return out

        return {
            "total_steps": max(int(current.get("total_steps", 0)) - int(baseline.get("total_steps", 0)), 0),
            "action_counts": _delta_map(current.get("action_counts"), baseline.get("action_counts")),
            "trade_stats": _delta_map(
                current.get("trade_stats"),
                baseline.get("trade_stats"),
                float_keys={"position_duration_sum"},
            ),
            "economics": _delta_map(
                current.get("economics"),
                baseline.get("economics"),
                float_keys=set((current.get("economics") or {}).keys()) | set((baseline.get("economics") or {}).keys()),
            ),
            "reward_components": _delta_map(
                current.get("reward_components"),
                baseline.get("reward_components"),
                float_keys=set((current.get("reward_components") or {}).keys())
                | set((baseline.get("reward_components") or {}).keys()),
            ),
        }


class RuntimeGymEnv(gym.Env):
    """
    Gym wrapper around the shared replay/live execution stack:
    FeatureEngine + RuntimeEngine + ReplayBroker.

    Reward semantics are taken from RuntimeEngine. The optimized reward is the
    clipped net log-equity delta after drawdown and transaction penalties,
    while the raw components remain available via reward_components.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        symbol: str,
        bars_frame: pd.DataFrame | None = None,
        bars: np.ndarray | list[VolumeBar] | None = None,
        scaler: Any,

        action_map: Sequence[ActionSpec],
        risk_limits: RiskLimits | None = None,
        config: RuntimeGymConfig | None = None,
        recovery_config: dict[str, Any] | None = None,
        alpha_gate: BaselineAlphaGate | None = None,
    ) -> None:
        super().__init__()
        self.symbol = symbol.upper()
        self.action_map = tuple(action_map)
        self.risk_limits = risk_limits or RiskLimits()
        self.config = config or RuntimeGymConfig()

        if bars is not None:
            self._bars = bars
            # Use small slice of bars_frame for observation space reference if provided, else dummy
            self._frame = bars_frame if bars_frame is not None else pd.DataFrame()
        elif bars_frame is not None:
            frame = bars_frame.copy()
            if not isinstance(frame.index, pd.DatetimeIndex):
                if "Gmt time" in frame.columns:
                    frame["Gmt time"] = pd.to_datetime(frame["Gmt time"], utc=True, errors="coerce")
                    frame = frame.dropna(subset=["Gmt time"]).set_index("Gmt time")
                else:
                    raise ValueError("bars_frame must have a DatetimeIndex or a 'Gmt time' column.")
            if frame.index.tz is None:
                frame.index = frame.index.tz_localize("UTC")
            else:
                frame.index = frame.index.tz_convert("UTC")

            if "avg_spread" not in frame.columns:
                frame["avg_spread"] = 0.0
            if "time_delta_s" not in frame.columns:
                frame["time_delta_s"] = frame.index.to_series().diff().dt.total_seconds().fillna(0.0)

            self._frame = frame.sort_index()
            if len(self._frame) < WARMUP_BARS + 10:
                raise ValueError(f"bars_frame too short: need >= {WARMUP_BARS + 10}, got {len(self._frame)}")
            self._bars = self._frame_to_bars(self._frame)
        else:
            raise ValueError("Either bars_frame or bars must be provided to RuntimeGymEnv.")

        self.action_space = spaces.Discrete(len(self.action_map))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.window_size, len(FEATURE_COLS) + STATE_FEATURE_COUNT),
            dtype=np.float32,
        )

        self._scaler = scaler
        self._bar_index = 0
        self._runtime: RuntimeEngine | None = None
        self._last_observation: np.ndarray | None = None
        self.max_slippage_pips = float(self.config.slippage_pips)
        self._drawdown_penalty = float(self.config.drawdown_penalty)
        self._transaction_penalty = float(self.config.transaction_penalty)
        self._episode_start_index = 0
        self._recovery_config = copy.deepcopy(recovery_config) if recovery_config is not None else None
        self._global_step = 0
        self._training_diagnostics = TrainingDiagnostics()
        self._episode_diagnostics_baseline: dict[str, Any] | None = None
        self._alpha_gate = alpha_gate
        self._last_alpha_gate_scores: dict[str, float] | None = None
        self._perf = {
            "action_masks_calls": 0,
            "action_masks_total_ns": 0,
            "step_calls": 0,
            "step_total_ns": 0,
            "step_info_build_ns": 0,
        }

        self._reset_internal()

    def set_participation_bonus_value(self, value: float) -> None:
        """Dynamically update the participation bonus magnitude from the curriculum callback."""
        if self._recovery_config and "participation_bonus" in self._recovery_config:
            self._recovery_config["participation_bonus"]["bonus_value"] = float(value)

    def _reset_internal(self) -> None:
        self._bar_index = 0
        self._runtime = None
        self._last_observation = None
        self._episode_start_index = 0
        self._pending_entry_direction = 0
        self._episode_finalized = False
        self._episode_diagnostics_baseline = None
        self._last_alpha_gate_scores = None

    @staticmethod
    def _frame_to_bars(frame: pd.DataFrame) -> np.ndarray:
        arr = np.empty(len(frame), dtype=BAR_DTYPE)
        
        timestamps_s = frame.index.view(np.int64) / 10**9
        arr['timestamp_s'] = timestamps_s
        arr['open'] = frame['Open'].to_numpy(dtype=np.float32)
        arr['high'] = frame['High'].to_numpy(dtype=np.float32)
        arr['low'] = frame['Low'].to_numpy(dtype=np.float32)
        arr['close'] = frame['Close'].to_numpy(dtype=np.float32)
        arr['volume'] = frame['Volume'].to_numpy(dtype=np.float32)
        arr['avg_spread'] = frame.get("avg_spread", pd.Series(0.0, index=frame.index)).to_numpy(dtype=np.float32)
        
        time_delta_s = frame.get("time_delta_s", pd.Series(0.0, index=frame.index)).to_numpy(dtype=np.float32)
        arr['time_delta_s'] = time_delta_s
        
        start_time_msc = (timestamps_s * 1000).astype(np.int64)
        arr['start_time_msc'] = start_time_msc
        arr['end_time_msc'] = start_time_msc + (np.maximum(time_delta_s, 0.0) * 1000).astype(np.int64)
        
        return arr

    def _bootstrap_runtime(self, start_index: int) -> RuntimeEngine:
        feature_engine = FeatureEngine.from_scaler(self._scaler)
        warmup_start = max(0, int(start_index) - WARMUP_BARS)
        warmup_frame = self._frame.iloc[warmup_start:int(start_index)].copy()
        feature_engine.warm_up(warmup_frame)
        broker = ReplayBroker(
            symbol=self.symbol,
            initial_equity=self.config.initial_equity,
            commission_per_lot=self.config.commission_per_lot,
            slippage_pips=float(self.max_slippage_pips),
            partial_fill_ratio=self.config.partial_fill_ratio,
        )
        snapshot = RuntimeSnapshot(
            last_equity=float(self.config.initial_equity),
            high_water_mark=float(self.config.initial_equity),
            day_start_equity=float(self.config.initial_equity),
        )
        risk_engine = RiskEngine(self.risk_limits, snapshot=snapshot, initial_equity=float(self.config.initial_equity))
        dummy_policy = None
        runtime = RuntimeEngine(
            symbol=self.symbol,
            feature_engine=feature_engine,
            policy=dummy_policy,
            broker=broker,
            action_map=self.action_map,
            risk_engine=risk_engine,
            snapshot=snapshot,
            state_store=None,
            reward_scale=float(self.config.reward_scale),
            reward_drawdown_penalty=float(self._drawdown_penalty),
            reward_transaction_penalty=float(self._transaction_penalty),
            reward_clip_low=float(self.config.reward_clip_low),
            reward_clip_high=float(self.config.reward_clip_high),
            window_size=int(self.config.window_size),
            minimal_post_cost_reward=bool(self.config.minimal_post_cost_reward),
            force_fast_window_benchmark=bool(self.config.force_fast_window_benchmark),
            alpha_gate=self._alpha_gate,
            churn_min_hold_bars=int(self.config.churn_min_hold_bars),
            churn_action_cooldown=int(self.config.churn_action_cooldown),
            entry_spread_z_limit=float(self.config.entry_spread_z_limit),
        )
        runtime.startup_reconcile()
        return runtime

    def _resolve_start_index(self) -> int:
        if not self.config.random_start:
            return WARMUP_BARS
        return int(self.np_random.integers(WARMUP_BARS, len(self._frame) - 1))

    def _apply_runtime_slippage(self) -> None:
        if self._runtime is None:
            return
        broker = self._runtime.broker
        if hasattr(broker, "slippage_pips"):
            setattr(broker, "slippage_pips", float(self.max_slippage_pips))

    def _estimate_slippage_penalty(self, *, turnover_lots: float, current_price: float, equity_base: float) -> float:
        turnover = max(float(turnover_lots), 0.0)
        if turnover <= 0 or equity_base <= 0:
            return 0.0
        pip_value_per_lot = pip_value_for_volume(
            self.symbol,
            price=float(current_price),
            volume_lots=1.0,
            account_currency="USD",
        )
        slippage_cost_usd = turnover * pip_value_per_lot * max(float(self.max_slippage_pips), 0.0)
        slippage_ratio = slippage_cost_usd / max(float(equity_base), 1e-6)
        return float(self._runtime.reward_transaction_penalty * self._runtime.reward_scale * slippage_ratio)

    def _estimate_downside_risk_penalty(self, reward_components: dict[str, float]) -> float:
        coef = max(float(self.config.downside_risk_penalty), 0.0)
        if coef <= 0.0:
            return 0.0
        # Keep penalty in the same units as the base reward. Using log_return * reward_scale
        # again would double-scale and can saturate clipping, starving PPO of gradient.
        reward_raw = float(reward_components.get("reward_raw_unclipped", 0.0))
        downside = max(-reward_raw, 0.0)
        return float(coef * downside)

    def _estimate_turnover_penalty(self, reward_components: dict[str, float]) -> float:
        coef = max(float(self.config.turnover_penalty), 0.0)
        if coef <= 0.0:
            return 0.0
        # Penalize turnover as an additional fraction of the already-scaled transaction penalty.
        transaction_penalty = max(float(reward_components.get("transaction_penalty_applied", 0.0)), 0.0)
        return float(coef * transaction_penalty)

    def _estimate_holding_penalty(self, closed_trades: Sequence[dict[str, Any]] | None) -> float:
        penalty_unit = max(float(self.config.churn_penalty_usd), 0.0)
        min_hold_bars = max(int(self.config.churn_min_hold_bars), 0)
        if penalty_unit <= 0.0 or min_hold_bars <= 0:
            return 0.0
        penalty = 0.0
        for trade in list(closed_trades or []):
            holding_bars = max(int(trade.get("holding_bars", 0) or 0), 0)
            if 0 < holding_bars < min_hold_bars:
                penalty += penalty_unit * float(min_hold_bars - holding_bars + 1)
        return float(penalty)

    def _estimate_rapid_reversal_penalty(self, entry_filled_direction: int) -> float:
        penalty_unit = max(float(self.config.churn_penalty_usd), 0.0)
        if penalty_unit <= 0.0:
            return 0.0
        if not self._training_diagnostics.is_rapid_reversal_candidate(entry_filled_direction):
            return 0.0
        return float(penalty_unit)

    def set_slippage_pips(self, value: float) -> None:
        self.max_slippage_pips = float(value)
        self._apply_runtime_slippage()

    def set_transaction_penalty(self, value: float) -> None:
        self._transaction_penalty = float(value)
        if self._runtime is not None:
            self._runtime.reward_transaction_penalty = float(value)

    def set_drawdown_penalty(self, value: float) -> None:
        self._drawdown_penalty = float(value)
        if self._runtime is not None:
            self._runtime.reward_drawdown_penalty = float(value)

    def set_global_step(self, value: int) -> None:
        self._global_step = int(value)

    def set_recovery_config(self, cfg: dict[str, Any] | None) -> None:
        self._recovery_config = copy.deepcopy(cfg) if cfg is not None else None

    def _alpha_gate_relaxation(self) -> tuple[float | None, float | None]:
        if self._alpha_gate is None:
            return None, None
        warmup_steps = max(int(self.config.alpha_gate_warmup_steps), 0)
        if warmup_steps <= 0 or self._global_step >= warmup_steps:
            return None, None
        progress = min(max(float(self._global_step) / float(warmup_steps), 0.0), 1.0)
        threshold_delta = max(float(self.config.alpha_gate_warmup_threshold_delta), 0.0) * (1.0 - progress)
        margin_scale_floor = min(max(float(self.config.alpha_gate_warmup_margin_scale), 0.0), 1.0)
        margin_scale = margin_scale_floor + ((1.0 - margin_scale_floor) * progress)
        threshold_override = max(float(self._alpha_gate.probability_threshold) - threshold_delta, 0.0)
        margin_override = max(float(self._alpha_gate.probability_margin) * margin_scale, 0.0)
        return float(threshold_override), float(margin_override)

    def get_training_diagnostics(self) -> dict[str, Any]:
        snapshot = self._training_diagnostics.snapshot()
        if self._alpha_gate is not None:
            threshold_override, margin_override = self._alpha_gate_relaxation()
            snapshot["alpha_gate"] = {
                "enabled": True,
                "model_kind": self._alpha_gate.model_kind,
                "probability_threshold": float(self._alpha_gate.probability_threshold),
                "probability_margin": float(self._alpha_gate.probability_margin),
                "effective_probability_threshold": float(
                    self._alpha_gate.probability_threshold if threshold_override is None else threshold_override
                ),
                "effective_probability_margin": float(
                    self._alpha_gate.probability_margin if margin_override is None else margin_override
                ),
                "min_edge_pips": float(self._alpha_gate.min_edge_pips),
                "fit_trade_count": float(self._alpha_gate.fit_trade_count),
                "fit_long_trade_count": float(self._alpha_gate.fit_long_trade_count),
                "fit_short_trade_count": float(self._alpha_gate.fit_short_trade_count),
                "fit_expectancy_usd": float(self._alpha_gate.fit_expectancy_usd),
                "fit_profit_factor": float(self._alpha_gate.fit_profit_factor),
                "fit_quality_passed": bool(self._alpha_gate.fit_quality_passed),
                "last_scores": dict(self._last_alpha_gate_scores or {}),
            }
        snapshot["perf"] = self.perf_snapshot()
        return snapshot

    def perf_snapshot(self) -> dict[str, Any]:
        perf = dict(self._perf)
        snapshot: dict[str, float] = {
            key: float(value) if key.endswith("_ns") else int(value)
            for key, value in perf.items()
        }
        for key in ("action_masks", "step", "step_info_build"):
            calls = int(perf.get(f"{key}_calls", 0))
            total_ns = int(perf.get(f"{key}_total_ns", 0))
            snapshot[f"{key}_mean_ns"] = float(total_ns / calls) if calls else 0.0
        runtime = self._runtime
        if runtime is not None and hasattr(runtime, "perf_snapshot"):
            snapshot["runtime"] = runtime.perf_snapshot()
        return snapshot

    def get_trade_log(self) -> list[dict[str, Any]]:
        """Return closed trade log from the current episode broker.

        Callable via SubprocVecEnv.env_method("get_trade_log") so that
        evaluate_model() can reliably extract trade counts even across process
        boundaries where get_attr("_runtime") may fail silently.
        """
        if self._runtime is None:
            return []
        broker = self._runtime.broker
        trade_log = getattr(broker, "trade_log", None)
        if not isinstance(trade_log, list):
            return []
        return [dict(item) for item in trade_log if isinstance(item, dict)]

    def get_execution_log(self) -> list[dict[str, Any]]:
        """Return execution log from the current episode broker.

        Callable via SubprocVecEnv.env_method("get_execution_log") for
        the same reason as get_trade_log().
        """
        if self._runtime is None:
            return []
        broker = self._runtime.broker
        execution_log = getattr(broker, "execution_log", None)
        if not isinstance(execution_log, list):
            return []
        return [dict(item) for item in execution_log if isinstance(item, dict)]

    def _latest_bar(self) -> BarView:
        if self._bars is None or len(self._bars) == 0:
            raise RuntimeError("No bars available for runtime environment.")
        index = min(max(int(self._bar_index), 0), len(self._bars) - 1)
        item = self._bars[index]
        if isinstance(item, np.void):
            return BarView(item)
        return item

    def _finalize_episode(self, *, reason: str) -> dict[str, Any]:
        if self._episode_finalized or self._runtime is None:
            return {
                "forced_close": False,
                "forced_close_count_delta": 0,
                "executed_order_count_delta": 0,
                "trade_closed_count_delta": 0,
                "gross_pnl_usd_delta": 0.0,
                "net_pnl_usd_delta": 0.0,
                "transaction_cost_usd_delta": 0.0,
                "executed_events": [],
                "closed_trades": [],
                "equity": float(self._runtime.last_equity) if self._runtime is not None else float(self.config.initial_equity),
            }
        prev_position = int(self._runtime.confirmed_position.direction)
        prev_position_duration = int(self._runtime.confirmed_position.time_in_trade_bars)
        broker = self._runtime.broker
        execution_log_before = len(getattr(broker, "execution_log", []))
        trade_log_before = len(getattr(broker, "trade_log", []))
        final_bar = self._latest_bar()
        forced_summary = self._runtime.force_flatten(final_bar, reason=reason)
        executed_events = [dict(item) for item in getattr(broker, "execution_log", [])[execution_log_before:] if isinstance(item, dict)]
        closed_trades = [dict(item) for item in getattr(broker, "trade_log", [])[trade_log_before:] if isinstance(item, dict)]
        if executed_events or closed_trades:
            self._training_diagnostics.record_forced_close(
                prev_position=prev_position,
                prev_position_duration=prev_position_duration,
                executed_events=executed_events,
                closed_trades=closed_trades,
            )
        self._episode_finalized = True
        return {
            "forced_close": bool(forced_summary.get("forced_close", False)),
            "forced_close_count_delta": int(sum(1 for event in executed_events if bool(event.get("forced", False)))),
            "executed_order_count_delta": int(len(executed_events)),
            "trade_closed_count_delta": int(len(closed_trades)),
            "gross_pnl_usd_delta": float(sum(float(trade.get("gross_pnl_usd", 0.0)) for trade in closed_trades)),
            "net_pnl_usd_delta": float(sum(float(trade.get("net_pnl_usd", 0.0)) for trade in closed_trades)),
            "transaction_cost_usd_delta": float(sum(float(trade.get("transaction_cost_usd", 0.0)) for trade in closed_trades)),
            "executed_events": executed_events,
            "closed_trades": closed_trades,
            "equity": float(forced_summary.get("equity", self._runtime.last_equity)),
        }

    def _log_slice(self, attr_name: str, start_index: int) -> list[dict[str, Any]]:
        if self._runtime is None:
            return []
        values = getattr(self._runtime.broker, attr_name, None)
        if not isinstance(values, list):
            return []
        return [dict(item) for item in values[start_index:] if isinstance(item, dict)]

    def _force_close_end_of_path(
        self,
        *,
        bar: BarView | VolumeBar,
        prev_equity: float,
        turnover_lots: float,
        reward_components: dict[str, float],
        result: Any,
    ) -> tuple[dict[str, float], float, bool]:
        forced_close = False
        if self._runtime is None or int(result.position_direction) == 0:
            reward_components.setdefault("forced_close_applied", 0.0)
            return reward_components, float(turnover_lots), forced_close
        flatten_result = self._runtime.force_flatten(bar, reason="FORCED_EPISODE_CLOSE")
        if not bool(flatten_result.get("forced_close", False)):
            reward_components.setdefault("forced_close_applied", 0.0)
            return reward_components, float(turnover_lots), forced_close
        forced_close = True
        total_turnover_lots = float(turnover_lots) + float(flatten_result.get("turnover_lots", 0.0))
        updated_reward_components = dict(
            self._runtime._build_reward_components(
                float(flatten_result.get("equity", result.equity)),
                current_price=float(bar.close),
                turnover_lots=total_turnover_lots,
                avg_spread=float(bar.avg_spread),
            )
        )
        updated_reward_components["pnl_reward"] = float(
            updated_reward_components.get("reward_raw_unclipped", updated_reward_components.get("pnl_reward", 0.0))
        )
        updated_reward_components["slippage_penalty_applied"] = self._estimate_slippage_penalty(
            turnover_lots=total_turnover_lots,
            current_price=float(bar.close),
            equity_base=prev_equity,
        )
        updated_reward_components["forced_close_applied"] = 1.0
        result.equity = float(flatten_result.get("equity", result.equity))
        result.position_direction = 0
        result.reward_components = dict(updated_reward_components)
        return updated_reward_components, total_turnover_lots, forced_close

    def action_masks(self) -> np.ndarray:
        start_ns = time.perf_counter_ns()
        try:
            if self._runtime is None:
                return np.zeros(len(self.action_map), dtype=bool)
            self._last_alpha_gate_scores = None

            feature_engine = self._runtime.feature_engine
            latest_raw = getattr(feature_engine, "latest_features_raw", np.zeros(len(FEATURE_COLS), dtype=np.float32))
            has_current_raw_features = len(latest_raw) == len(FEATURE_COLS)
            if has_current_raw_features:
                spread_z = float(latest_raw[FEATURE_COLS.index("spread_z")])
                latest_row = {col: float(latest_raw[idx]) for idx, col in enumerate(FEATURE_COLS)}
            elif feature_engine._buffer is not None:
                latest_buffer_row = feature_engine._buffer.iloc[-1]
                spread_z = float(latest_buffer_row.get("spread_z", 0.0))
                latest_row = latest_buffer_row
            else:
                spread_z = 0.0
                latest_row = {}

            mask = build_action_mask(
                self.action_map,
                position=self._runtime.confirmed_position,
                spread_z=spread_z
            )
            is_flat = bool(self._runtime.confirmed_position.is_flat)
            base_entry_allowed = bool(np.any(mask[2:])) if mask.size > 2 else False
            spread_blocked = bool(is_flat and not base_entry_allowed)
            min_hold_blocked = bool(
                (not is_flat)
                and int(self.config.churn_min_hold_bars) > 0
                and int(self._runtime.confirmed_position.time_in_trade_bars) < int(self.config.churn_min_hold_bars)
            )
            cooldown_blocked = False
            if is_flat and int(self.config.churn_action_cooldown) > 0 and self._runtime.last_close_bar_index is not None:
                bars_since_close = int(self._runtime.processed_bars_count) - int(self._runtime.last_close_bar_index)
                cooldown_blocked = bool(bars_since_close < int(self.config.churn_action_cooldown))
            mask = apply_execution_action_guards(
                mask,
                position=self._runtime.confirmed_position,
                spread_z=spread_z,
                entry_spread_z_limit=float(self.config.entry_spread_z_limit),
                churn_min_hold_bars=int(self.config.churn_min_hold_bars),
                current_bar_index=int(self._runtime.processed_bars_count),
                last_close_bar_index=self._runtime.last_close_bar_index,
                churn_action_cooldown=int(self.config.churn_action_cooldown),
            )

            alpha_gate_allow_long: bool | None = None
            alpha_gate_allow_short: bool | None = None
            if self._runtime.confirmed_position.is_flat and self._alpha_gate is not None:
                threshold_override, margin_override = self._alpha_gate_relaxation()
                allow_long, allow_short, scores = self._alpha_gate.allowed_directions(
                    latest_row,
                    threshold_override=threshold_override,
                    margin_override=margin_override,
                )
                self._last_alpha_gate_scores = dict(scores)
                alpha_gate_allow_long = bool(allow_long)
                alpha_gate_allow_short = bool(allow_short)
                for idx, action in enumerate(self.action_map):
                    if action.action_type != ActionType.OPEN:
                        continue
                    direction = int(action.direction or 0)
                    if direction > 0 and not allow_long:
                        mask[idx] = False
                    if direction < 0 and not allow_short:
                        mask[idx] = False
            self._training_diagnostics.record_mask_observation(
                spread_blocked=spread_blocked,
                cooldown_blocked=cooldown_blocked,
                min_hold_blocked=min_hold_blocked,
                alpha_gate_enabled=bool(is_flat and self._alpha_gate is not None),
                alpha_gate_allow_long=alpha_gate_allow_long,
                alpha_gate_allow_short=alpha_gate_allow_short,
            )
            return mask
        finally:
            self._perf["action_masks_calls"] += 1
            self._perf["action_masks_total_ns"] += max(time.perf_counter_ns() - start_ns, 0)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_internal()
        self._training_diagnostics.reset_episode_state()
        self._episode_diagnostics_baseline = self._training_diagnostics.snapshot()
        self._episode_start_index = self._resolve_start_index()
        self._runtime = self._bootstrap_runtime(self._episode_start_index)
        self._apply_runtime_slippage()
        self._bar_index = self._episode_start_index
        first_bar = self._latest_bar()

        # Prime the environment by processing the first actionable bar with HOLD.
        result = self._runtime.process_bar(first_bar, action_index_override=0)
        self._last_observation = result.observation
        info = {
            "equity": float(result.equity),
            "total_equity_usd": float(result.equity),
            "reward": float(result.reward),
            "reward_components": dict(result.reward_components),
            "episode_start_index": int(self._episode_start_index),
            "timestamp_utc": pd.Timestamp(first_bar.timestamp).isoformat(),
        }
        return (self._last_observation, info) if _GYM else self._last_observation

    def step(self, action: int):
        step_start_ns = time.perf_counter_ns()
        if self._runtime is None:
            raise RuntimeError("Environment not reset.")
        prev_equity = float(self._runtime.last_equity)
        prev_position = int(self._runtime.confirmed_position.direction)
        prev_position_duration = int(self._runtime.confirmed_position.time_in_trade_bars)
        self._apply_runtime_slippage()
        action = int(action)
        if not (0 <= action < len(self.action_map)):
            raise ValueError(f"Invalid action {action}")

        if self._bar_index >= len(self._bars) - 1:
            finalization = self._finalize_episode(reason="FORCED_EPISODE_CLOSE")
            terminated = True
            truncated = False
            obs = (
                self._last_observation
                if self._last_observation is not None
                else np.zeros(self.observation_space.shape, dtype=np.float32)
            )
            info = {
                "equity": float(finalization["equity"]),
                "total_equity_usd": float(finalization["equity"]),
                "timestamp_utc": None,
                "reward_components": {"forced_close_applied": float(bool(finalization["forced_close"]))},
                "episode_start_index": int(self._episode_start_index),
                "forced_close": bool(finalization["forced_close"]),
                "forced_close_count": int(finalization["forced_close_count_delta"]),
                "executed_order_count": int(finalization["executed_order_count_delta"]),
                "trade_count": int(finalization["trade_closed_count_delta"]),
                "gross_pnl_usd_delta": float(finalization["gross_pnl_usd_delta"]),
                "net_pnl_usd_delta": float(finalization["net_pnl_usd_delta"]),
                "transaction_cost_usd_delta": float(finalization["transaction_cost_usd_delta"]),
                "executed_events_delta": [dict(item) for item in list(finalization.get("executed_events", []) or []) if isinstance(item, dict)],
                "closed_trades_delta": [dict(item) for item in list(finalization.get("closed_trades", []) or []) if isinstance(item, dict)],
                "episode_diagnostics": self._training_diagnostics.snapshot_delta(self._episode_diagnostics_baseline),
            }
            if _GYM:
                self._perf["step_calls"] += 1
                self._perf["step_total_ns"] += max(time.perf_counter_ns() - step_start_ns, 0)
                return obs, 0.0, terminated, truncated, info
            self._perf["step_calls"] += 1
            self._perf["step_total_ns"] += max(time.perf_counter_ns() - step_start_ns, 0)
            return obs, 0.0, True, info

        self._bar_index += 1
        bar = self._latest_bar()
        result = self._runtime.process_bar(bar, action_index_override=action)
        self._last_observation = result.observation
        reward_components = dict(result.reward_components)
        turnover_lots = float(reward_components.get("turnover_lots", 0.0))
        entry_filled_direction = 0
        if int(prev_position) == 0 and self._pending_entry_direction != 0 and turnover_lots > 0.0:
            entry_filled_direction = int(self._pending_entry_direction)
        self._pending_entry_direction = 0

        entry_signal_direction = 0
        if (
            result.action.action_type == ActionType.OPEN
            and int(prev_position) == 0
            and result.action.direction is not None
            and result.submit_result is not None
            and bool(getattr(result.submit_result, "accepted", False))
        ):
            entry_signal_direction = int(result.action.direction)
            self._pending_entry_direction = entry_signal_direction

        executed_events = [dict(item) for item in result.executed_events] if result.executed_events else []
        closed_trades = [dict(item) for item in result.closed_trades] if result.closed_trades else []
        
        # Micro-optimization: Skip expensive penalty lookups if nothing happened and we are flat
        if turnover_lots > 0.0 or closed_trades:
            reward_components["slippage_penalty_applied"] = self._estimate_slippage_penalty(
                turnover_lots=turnover_lots,
                current_price=float(bar.close),
                equity_base=prev_equity,
            )
            reward_components["turnover_penalty_applied"] = self._estimate_turnover_penalty(reward_components)
            reward_components["downside_risk_penalty_applied"] = self._estimate_downside_risk_penalty(reward_components)
            reward_components["rapid_reversal_penalty_applied"] = self._estimate_rapid_reversal_penalty(
                entry_filled_direction
            )
            reward_components["holding_penalty_applied"] = self._estimate_holding_penalty(closed_trades)
        else:
            reward_components["slippage_penalty_applied"] = 0.0
            reward_components["turnover_penalty_applied"] = 0.0
            reward_components["downside_risk_penalty_applied"] = 0.0
            reward_components["rapid_reversal_penalty_applied"] = 0.0
            reward_components["holding_penalty_applied"] = 0.0

        terminated = bool(self._bar_index >= len(self._bars) - 1 or result.kill_switch_active)
        finalization: dict[str, Any] | None = None
        if terminated and not result.kill_switch_active:
            finalization = self._finalize_episode(reason="FORCED_EPISODE_CLOSE")
            reward_components["forced_close_applied"] = float(bool(finalization["forced_close"]))
            if bool(finalization["forced_close"]):
                result.equity = float(finalization["equity"])
                result.position_direction = 0
        else:
            reward_components.setdefault("forced_close_applied", 0.0)
        bonus_direction = entry_signal_direction if entry_signal_direction != 0 else int(result.position_direction)
        participation_bonus = compute_participation_bonus(
            prev_position=prev_position,
            new_position=bonus_direction,
            global_step=self._global_step,
            episode_bonus_count=self._training_diagnostics.episode_bonus_count,
            last_bonus_step=self._training_diagnostics.last_bonus_step,
            cfg=self._recovery_config,
        )
        reward_components["participation_bonus_applied"] = float(participation_bonus)
        base_reward_unclipped = float(reward_components.get("reward_unclipped", result.reward))
        reward_components["base_reward_unclipped"] = float(base_reward_unclipped)
        if self.config.minimal_post_cost_reward:
            final_reward = float(result.equity) - float(prev_equity)
            reward_components["participation_bonus_applied"] = 0.0
            reward_components["turnover_penalty_applied"] = 0.0
            reward_components["downside_risk_penalty_applied"] = 0.0
            reward_components["rapid_reversal_penalty_applied"] = 0.0
            reward_components["holding_penalty_applied"] = 0.0
            reward_components["net_return_adjustment_applied"] = 0.0
            reward_components["pre_bonus_reward"] = float(final_reward)
            reward_components["final_reward_unclipped"] = float(final_reward)
            reward_components["reward_raw_unclipped"] = float(final_reward)
            reward_components["reward_unclipped"] = float(final_reward)
            reward_components["reward_clipped"] = float(final_reward)
            reward_components["base_reward_unclipped"] = float(final_reward)
            reward_components["net_reward_with_bonus"] = float(final_reward)
            reward_components["reward_unclipped_net"] = float(final_reward)
            reward_components["final_reward_clipped_low"] = 0.0
            reward_components["final_reward_clipped_high"] = 0.0
            reward_components["clip_hit_high"] = 0.0
            reward_components["clip_hit_low"] = 0.0
        else:
            final_reward, composed_fields = compose_final_reward(
                base_reward_unclipped=base_reward_unclipped,
                net_return_coef=float(self.config.net_return_coef),
                turnover_penalty=float(reward_components["turnover_penalty_applied"]),
                downside_risk_penalty=float(reward_components["downside_risk_penalty_applied"]),
                rapid_reversal_penalty=float(reward_components["rapid_reversal_penalty_applied"]),
                holding_penalty=float(reward_components["holding_penalty_applied"]),
                participation_bonus=float(participation_bonus),
                clip_low=float(self.config.reward_clip_low),
                clip_high=float(self.config.reward_clip_high),
            )
            reward_components.update(composed_fields)
            reward_components["net_reward_with_bonus"] = float(final_reward)
            reward_components["reward_unclipped_net"] = float(base_reward_unclipped + participation_bonus)
            reward_components["clip_hit_high"] = float(reward_components["reward_unclipped_net"] >= self.config.reward_clip_high)
            reward_components["clip_hit_low"] = float(reward_components["reward_unclipped_net"] <= self.config.reward_clip_low)
            if participation_bonus > 0.0:
                self._training_diagnostics.mark_bonus_awarded(self._global_step)
                self._runtime.confirmed_position.last_reward = float(final_reward)
        result.reward = float(final_reward)
        result.reward_components = reward_components
        self._training_diagnostics.record_step(
            action=result.action,
            submit_result=result.submit_result,
            prev_position=prev_position,
            new_position=int(result.position_direction),
            prev_position_duration=prev_position_duration,
            entry_signal_direction=entry_signal_direction,
            entry_filled_direction=entry_filled_direction,
            executed_events=executed_events,
            closed_trades=closed_trades,
            reward_components=reward_components,
            reward=float(final_reward),
        )

        truncated = False
        forced_close_count = int(finalization["forced_close_count_delta"]) if finalization is not None else 0
        executed_order_count = int(len(executed_events)) + int(finalization["executed_order_count_delta"]) if finalization is not None else int(len(executed_events))
        trade_count = int(len(closed_trades)) + int(finalization["trade_closed_count_delta"]) if finalization is not None else int(len(closed_trades))
        
        if self.config.slim_info:
            is_event = (
                terminated or 
                trade_count > 0 or 
                executed_order_count > 0 or
                bool(result.kill_switch_active) or 
                bool(reward_components.get("forced_close_applied", 0.0) > 0.0) or
                bool(reward_components.get("participation_bonus_applied", 0.0) > 0.0)
            )
            if not is_event:
                if _GYM:
                    self._perf["step_calls"] += 1
                    self._perf["step_total_ns"] += max(time.perf_counter_ns() - step_start_ns, 0)
                    return result.observation, float(final_reward), terminated, truncated, {}
                self._perf["step_calls"] += 1
                self._perf["step_total_ns"] += max(time.perf_counter_ns() - step_start_ns, 0)
                return result.observation, float(final_reward), bool(terminated or truncated), {}

        info_start_ns = time.perf_counter_ns()
        gross_pnl_usd_delta = float(sum(float(trade.get("gross_pnl_usd", 0.0)) for trade in closed_trades))
        net_pnl_usd_delta = float(sum(float(trade.get("net_pnl_usd", 0.0)) for trade in closed_trades))
        transaction_cost_usd_delta = float(sum(float(trade.get("transaction_cost_usd", 0.0)) for trade in closed_trades))
        if finalization is not None:
            gross_pnl_usd_delta += float(finalization["gross_pnl_usd_delta"])
            net_pnl_usd_delta += float(finalization["net_pnl_usd_delta"])
            transaction_cost_usd_delta += float(finalization["transaction_cost_usd_delta"])
        executed_events_delta = [dict(item) for item in executed_events if isinstance(item, dict)]
        closed_trades_delta = [dict(item) for item in closed_trades if isinstance(item, dict)]
        if finalization is not None:
            executed_events_delta.extend(
                dict(item) for item in list(finalization.get("executed_events", []) or []) if isinstance(item, dict)
            )
            closed_trades_delta.extend(
                dict(item) for item in list(finalization.get("closed_trades", []) or []) if isinstance(item, dict)
            )

        info = {
            "equity": float(result.equity),
            "total_equity_usd": float(result.equity),
            "reward": float(final_reward),
            "reward_pnl": float(net_pnl_usd_delta),
            "reward_bonus": float(reward_components.get("participation_bonus_applied", 0.0)),
            "reward_penalty": float(
                transaction_cost_usd_delta +
                reward_components.get("turnover_penalty_applied", 0.0) +
                reward_components.get("downside_risk_penalty_applied", 0.0) +
                reward_components.get("rapid_reversal_penalty_applied", 0.0) +
                reward_components.get("holding_penalty_applied", 0.0)
            ),
            "reward_components": dict(reward_components),
            "reward_unclipped": float(reward_components.get("reward_unclipped_net", final_reward)),
            "clip_hit_rate": float(reward_components.get("clip_hit_high", 0) + reward_components.get("clip_hit_low", 0)),
            "participation_bonus_applied": float(reward_components.get("participation_bonus_applied", 0.0)),
            "selected_action_index": int(result.action_index),
            "selected_action_label": action_label(result.action),
            "selected_action_type": result.action.action_type.value,
            "selected_action_direction": int(result.action.direction or 0),
            "action_accepted": bool(result.submit_result is not None and bool(getattr(result.submit_result, "accepted", False))),
            "kill_switch_active": bool(result.kill_switch_active),
            "kill_switch_reason": result.kill_switch_reason,
            "episode_start_index": int(self._episode_start_index),
            "timestamp_utc": pd.Timestamp(bar.timestamp).isoformat(),
            "forced_close": bool(reward_components.get("forced_close_applied", 0.0) > 0.0),
            "forced_close_count": forced_close_count,
            "executed_order_count": executed_order_count,
            "trade_count": trade_count,
            "gross_pnl_usd_delta": gross_pnl_usd_delta,
            "net_pnl_usd_delta": net_pnl_usd_delta,
            "transaction_cost_usd_delta": transaction_cost_usd_delta,
            "executed_events_delta": executed_events_delta,
            "closed_trades_delta": closed_trades_delta,
        }
        if terminated:
            info["episode_diagnostics"] = self._training_diagnostics.snapshot_delta(self._episode_diagnostics_baseline)
        self._perf["step_info_build_ns"] += max(time.perf_counter_ns() - info_start_ns, 0)
        if _GYM:
            self._perf["step_calls"] += 1
            self._perf["step_total_ns"] += max(time.perf_counter_ns() - step_start_ns, 0)
            return result.observation, float(final_reward), terminated, truncated, info
        self._perf["step_calls"] += 1
        self._perf["step_total_ns"] += max(time.perf_counter_ns() - step_start_ns, 0)
        return result.observation, float(final_reward), bool(terminated or truncated), info

