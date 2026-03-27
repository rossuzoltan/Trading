from __future__ import annotations

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

from event_pipeline import ReplayBroker, RiskEngine, RiskLimits, RuntimeEngine, RuntimeSnapshot, VolumeBar
from feature_engine import FEATURE_COLS, FeatureEngine, WARMUP_BARS
from runtime_common import STATE_FEATURE_COUNT, ActionSpec, build_action_mask, build_observation


@dataclass(frozen=True)
class RuntimeGymConfig:
    initial_equity: float = 1_000.0
    commission_per_lot: float = 7.0
    slippage_pips: float = 0.25
    partial_fill_ratio: float = 1.0
    window_size: int = 1
    random_start: bool = False


class RuntimeGymEnv(gym.Env):
    """
    Gym wrapper around the shared replay/live execution stack:
    FeatureEngine + RuntimeEngine + ReplayBroker.

    Reward semantics are taken from RuntimeEngine. The optimized reward is the
    clipped post-cost log-equity delta, while drawdown and estimated transaction
    costs remain available as telemetry via reward_components.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        symbol: str,
        bars_frame: pd.DataFrame,
        scaler: Any,
        action_map: Sequence[ActionSpec],
        risk_limits: RiskLimits | None = None,
        config: RuntimeGymConfig | None = None,
    ) -> None:
        super().__init__()
        self.symbol = symbol.upper()
        self.action_map = tuple(action_map)
        self.risk_limits = risk_limits or RiskLimits()
        self.config = config or RuntimeGymConfig()

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in bars_frame.columns]
        if missing:
            raise ValueError(f"bars_frame missing required columns: {missing}")

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
        self._episode_start_index = 0

        self._reset_internal()

    def _reset_internal(self) -> None:
        self._bar_index = 0
        self._runtime = None
        self._last_observation = None
        self._episode_start_index = 0

    @staticmethod
    def _frame_to_bars(frame: pd.DataFrame) -> list[VolumeBar]:
        bars: list[VolumeBar] = []
        for timestamp, row in frame.iterrows():
            time_delta_s = float(row.get("time_delta_s", 0.0))
            end_time_msc = int(pd.Timestamp(timestamp).timestamp() * 1000)
            bars.append(
                VolumeBar(
                    timestamp=pd.Timestamp(timestamp).to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    avg_spread=float(row.get("avg_spread", 0.0)),
                    time_delta_s=time_delta_s,
                    start_time_msc=end_time_msc,
                    end_time_msc=end_time_msc + int(max(time_delta_s, 0.0) * 1000),
                )
            )
        return bars

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
            policy=dummy_policy,  # not used when action_index_override is provided
            broker=broker,
            action_map=self.action_map,
            risk_engine=risk_engine,
            snapshot=snapshot,
            state_store=None,
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

    def action_masks(self) -> np.ndarray:
        if self._runtime is None or self._runtime.feature_engine._buffer is None:
            return np.zeros(len(self.action_map), dtype=bool)
        latest_row = self._runtime.feature_engine._buffer.iloc[-1]
        spread_z = float(latest_row.get("spread_z", 0.0))
        return build_action_mask(self.action_map, position=self._runtime.confirmed_position, spread_z=spread_z)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_internal()
        self._episode_start_index = self._resolve_start_index()
        self._runtime = self._bootstrap_runtime(self._episode_start_index)
        self._apply_runtime_slippage()
        self._bar_index = self._episode_start_index
        first_bar = self._bars[self._bar_index]

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
        if self._runtime is None:
            raise RuntimeError("Environment not reset.")
        self._apply_runtime_slippage()
        action = int(action)
        if not (0 <= action < len(self.action_map)):
            raise ValueError(f"Invalid action {action}")

        if self._bar_index >= len(self._bars) - 1:
            terminated = True
            truncated = False
            obs = self._last_observation if self._last_observation is not None else np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {
                "equity": float(self._runtime.last_equity),
                "timestamp_utc": None,
                "reward_components": {},
                "episode_start_index": int(self._episode_start_index),
            }
            if _GYM:
                return obs, 0.0, terminated, truncated, info
            return obs, 0.0, True, info

        self._bar_index += 1
        bar = self._bars[self._bar_index]
        result = self._runtime.process_bar(bar, action_index_override=action)
        self._last_observation = result.observation

        terminated = bool(self._bar_index >= len(self._bars) - 1 or result.kill_switch_active)
        truncated = False
        info = {
            "equity": float(result.equity),
            "total_equity_usd": float(result.equity),
            "reward": float(result.reward),
            "reward_components": dict(result.reward_components),
            "kill_switch_active": bool(result.kill_switch_active),
            "kill_switch_reason": result.kill_switch_reason,
            "episode_start_index": int(self._episode_start_index),
            "timestamp_utc": pd.Timestamp(bar.timestamp).isoformat(),
        }
        if _GYM:
            return result.observation, float(result.reward), terminated, truncated, info
        return result.observation, float(result.reward), bool(terminated or truncated), info
