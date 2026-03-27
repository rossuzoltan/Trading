from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from event_pipeline import RuntimeEngine, RuntimeSnapshot
from feature_engine import FEATURE_COLS, FeatureEngine, WARMUP_BARS, _compute_raw
from runtime_common import ConfirmedPosition, build_action_map, build_state_vector
from runtime_gym_env import RuntimeGymConfig, RuntimeGymEnv, compute_participation_bonus


def _make_history(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(1.1000, 1.1200, rows)
    wave = np.sin(np.arange(rows) / 9.0) * 0.0005
    close = base + wave
    open_ = close - 0.0002
    high = close + 0.0004
    low = close - 0.0004
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(rows, 100.0),
            "avg_spread": np.full(rows, 0.0001),
            "time_delta_s": np.full(rows, 3600.0),
        },
        index=index,
    )


class RewardFeatureRedesignTests(unittest.TestCase):
    def _make_runtime_engine(self) -> RuntimeEngine:
        snapshot = RuntimeSnapshot(
            last_equity=1_000.0,
            high_water_mark=1_000.0,
            day_start_equity=1_000.0,
        )
        broker = SimpleNamespace(commission_per_lot=7.0, slippage_pips=0.25)
        risk_engine = SimpleNamespace(high_water_mark=1_000.0)
        return RuntimeEngine(
            symbol="EURUSD",
            feature_engine=FeatureEngine(),
            policy=SimpleNamespace(),
            broker=broker,
            action_map=build_action_map([0.5], [0.5]),
            risk_engine=risk_engine,
            snapshot=snapshot,
        )

    def test_feature_list_is_restricted_but_raw_compatibility_columns_remain(self) -> None:
        expected = [
            "log_return",
            "body_size",
            "candle_range",
            "ma20_slope",
            "ma50_slope",
            "vol_norm_atr",
            "spread_z",
            "time_delta_z",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
        ]
        history = _make_history()
        raw = _compute_raw(history)
        engine = FeatureEngine()
        engine.warm_up(history)

        self.assertEqual(expected, FEATURE_COLS)
        self.assertEqual(len(expected), len(engine.latest_observation))
        for legacy_col in ("rsi_14", "macd", "macdh", "bb_bw", "bb_pct", "adx", "hurst_exp", "frac_diff_z"):
            self.assertIn(legacy_col, raw.columns)

    def test_reward_is_netted_and_components_log_the_applied_penalties(self) -> None:
        engine = self._make_runtime_engine()

        engine.last_equity = 1_000.0
        reward = engine._calc_reward(1_000.05, current_price=1.10, turnover_lots=0.05, avg_spread=0.0001)
        components = engine._build_reward_components(
            1_000.05,
            current_price=1.10,
            turnover_lots=0.05,
            avg_spread=0.0001,
        )

        expected_raw = 10_000.0 * np.log(1_000.05 / 1_000.0)
        expected_transaction_penalty = engine.reward_transaction_penalty * engine.reward_scale * components["estimated_cost_ratio"]
        expected_net = expected_raw - expected_transaction_penalty
        expected_clipped = float(np.clip(expected_net, -5.0, 5.0))

        self.assertAlmostEqual(expected_clipped, reward)
        self.assertAlmostEqual(expected_clipped, components["reward_clipped"])
        self.assertAlmostEqual(expected_raw, components["reward_raw_unclipped"])
        self.assertAlmostEqual(expected_net, components["reward_unclipped"])
        self.assertGreater(components["estimated_cost_ratio"], 0.0)
        self.assertEqual(0.0, components["drawdown_penalty_applied"])
        self.assertGreater(components["transaction_penalty_applied"], 0.0)
        self.assertAlmostEqual(expected_transaction_penalty, components["transaction_penalty_applied"])

    def test_short_pnl_sign_matches_actual_short_pnl(self) -> None:
        winning_short = ConfirmedPosition(direction=-1, entry_price=1.1000, volume=0.05)
        losing_short = ConfirmedPosition(direction=-1, entry_price=1.1000, volume=0.05)

        winning_state = build_state_vector(winning_short, current_price=1.0990, symbol="EURUSD")
        losing_state = build_state_vector(losing_short, current_price=1.1010, symbol="EURUSD")

        self.assertGreater(winning_state[2], 0.0)
        self.assertLess(losing_state[2], 0.0)

    def test_persisted_scaler_changes_live_observation(self) -> None:
        history = _make_history(rows=320)
        raw = _compute_raw(history).dropna(subset=FEATURE_COLS)
        scaler = StandardScaler().fit(raw.loc[:, FEATURE_COLS])

        unscaled_engine = FeatureEngine()
        unscaled_engine.warm_up(history)
        scaled_engine = FeatureEngine.from_scaler(scaler)
        scaled_engine.warm_up(history)

        unscaled_obs = unscaled_engine.latest_observation
        scaled_obs = scaled_engine.latest_observation

        self.assertEqual(len(FEATURE_COLS), len(unscaled_obs))
        self.assertEqual(len(FEATURE_COLS), len(scaled_obs))
        self.assertFalse(np.allclose(unscaled_obs, scaled_obs))

    def test_runtime_env_random_start_is_seeded_and_respects_warmup(self) -> None:
        history = _make_history(rows=320)
        raw = _compute_raw(history).dropna(subset=FEATURE_COLS)
        scaler = StandardScaler().fit(raw.loc[:, FEATURE_COLS])
        action_map = build_action_map([0.5], [0.5])

        def build_env(random_start: bool) -> RuntimeGymEnv:
            return RuntimeGymEnv(
                symbol="EURUSD",
                bars_frame=history,
                scaler=scaler,
                action_map=action_map,
                config=RuntimeGymConfig(random_start=random_start),
            )

        deterministic_env = build_env(random_start=False)
        _, deterministic_info = deterministic_env.reset(seed=7)
        self.assertEqual(WARMUP_BARS, deterministic_info["episode_start_index"])

        env_a = build_env(random_start=True)
        _, info_a = env_a.reset(seed=11)
        env_b = build_env(random_start=True)
        _, info_b = env_b.reset(seed=11)
        env_c = build_env(random_start=True)
        _, info_c = env_c.reset(seed=19)

        self.assertEqual(info_a["episode_start_index"], info_b["episode_start_index"])
        self.assertNotEqual(info_a["episode_start_index"], info_c["episode_start_index"])
        self.assertGreaterEqual(info_a["episode_start_index"], WARMUP_BARS)
        self.assertLessEqual(info_a["episode_start_index"], len(history) - 2)

    def test_participation_bonus_helper_and_runtime_diagnostics(self) -> None:
        bonus = compute_participation_bonus(
            prev_position=0,
            new_position=1,
            global_step=10,
            episode_bonus_count=0,
            last_bonus_step=-10_000,
            cfg={
                "participation_bonus": {
                    "enabled": True,
                    "bonus_value": 0.01,
                    "active_until_step": 500_000,
                    "cooldown_steps": 8,
                    "only_from_flat": True,
                    "max_bonus_per_episode": 50,
                }
            },
        )
        self.assertAlmostEqual(0.01, bonus)

        history = _make_history(rows=320)
        raw = _compute_raw(history).dropna(subset=FEATURE_COLS)
        scaler = StandardScaler().fit(raw.loc[:, FEATURE_COLS])
        action_map = build_action_map([0.5], [0.5])
        env = RuntimeGymEnv(
            symbol="EURUSD",
            bars_frame=history,
            scaler=scaler,
            action_map=action_map,
            config=RuntimeGymConfig(random_start=False, slippage_pips=0.1),
            recovery_config={
                "participation_bonus": {
                    "enabled": True,
                    "bonus_value": 0.01,
                    "active_until_step": 500_000,
                    "cooldown_steps": 8,
                    "only_from_flat": True,
                    "max_bonus_per_episode": 50,
                }
            },
        )

        env.reset(seed=7)
        env.set_global_step(10)
        env.set_slippage_pips(0.5)
        self.assertAlmostEqual(0.5, env._runtime.broker.slippage_pips)

        _, reward, terminated, truncated, info = env.step(2)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertGreaterEqual(reward, 0.0)
        self.assertAlmostEqual(0.01, info["reward_components"]["participation_bonus_applied"])
        diagnostics = env.get_training_diagnostics()
        self.assertEqual(1, diagnostics["action_counts"]["long"])
        self.assertEqual(1, diagnostics["trade_stats"]["entry_signal_long_count"])
        self.assertEqual(1, diagnostics["trade_stats"]["trade_attempt_count"])
        self.assertEqual(1, diagnostics["trade_stats"]["flat_steps"])
        self.assertAlmostEqual(0.01, diagnostics["reward_components"]["participation_bonus_sum"])


if __name__ == "__main__":
    unittest.main()
