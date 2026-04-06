from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd

from feature_engine import FEATURE_COLS
from train_agent import (
    TrainingDiagnosticsCallback,
    TrainingHeartbeatCallback,
    _apply_profile_override,
    _deployment_candidate_rank,
    _holdout_deployment_blockers,
    _resolve_training_experiment_profile,
    evaluate_model,
    find_optimal_env_workers,
    get_current_ent_coef,
    get_current_phase,
    get_current_slippage_pips,
    run_baseline_research_gate,
)


TEST_TMP_ROOT = Path("tests/tmp")


def make_test_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_supervised_frame(*, rows: int, predictive: bool) -> pd.DataFrame:
    rng = np.random.default_rng(42 if predictive else 123)
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    latent = np.sin(np.arange(rows) / 11.0) + 0.35 * np.cos(np.arange(rows) / 19.0)
    if predictive:
        returns = 0.00035 * latent + rng.normal(0.0, 0.00004, rows)
        feature_driver = latent
    else:
        returns = rng.normal(0.0, 0.00035, rows)
        feature_driver = rng.normal(0.0, 1.0, rows)

    close = 1.1000 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame(
        {
            "Close": close,
            "log_return": np.concatenate([[0.0], returns[:-1]]),
            "body_size": feature_driver + rng.normal(0.0, 0.02, rows),
            "candle_range": np.abs(feature_driver) + 0.1,
            "ma20_slope": pd.Series(feature_driver).rolling(20, min_periods=1).mean().to_numpy(),
            "ma50_slope": pd.Series(feature_driver).rolling(50, min_periods=1).mean().to_numpy(),
            "vol_norm_atr": np.abs(returns) * 100.0 + 0.01,
            "spread_z": rng.normal(0.0, 0.5, rows),
            "time_delta_z": rng.normal(0.0, 0.5, rows),
            "hour_sin": np.sin(2 * np.pi * index.hour / 24),
            "hour_cos": np.cos(2 * np.pi * index.hour / 24),
            "day_sin": np.sin(2 * np.pi * index.dayofweek / 5),
            "day_cos": np.cos(2 * np.pi * index.dayofweek / 5),
        },
        index=index,
    )
    return frame


class DummyModel:
    def predict(self, obs, action_masks=None, deterministic=True):
        return np.array([0], dtype=np.int64), None


class DummyEvalEnv:
    def __init__(self, equities: list[float], rewards: list[float], timestamps: list[pd.Timestamp]) -> None:
        self._equities = list(equities)
        self._rewards = list(rewards)
        self._timestamps = list(timestamps)
        self._index = 0

    def reset(self):
        self._index = 0
        return np.zeros((1, 1), dtype=np.float32)

    def env_method(self, name: str):
        if name != "action_masks":
            raise AssertionError(f"Unexpected env_method {name}")
        return [np.array([True], dtype=bool)]

    def step(self, action):
        equity = float(self._equities[self._index])
        reward = float(self._rewards[self._index])
        timestamp = self._timestamps[self._index]
        self._index += 1
        done = self._index >= len(self._equities)
        return (
            np.zeros((1, 1), dtype=np.float32),
            np.array([reward], dtype=np.float32),
            np.array([done], dtype=bool),
            [{"equity": equity, "timestamp_utc": timestamp.isoformat()}],
        )

    def get_attr(self, name: str):
        if name != "equity_usd":
            raise AssertionError(f"Unexpected attr {name}")
        return [float(self._equities[max(self._index - 1, 0)])]


class DummyTrainingEnv:
    def __init__(self, snapshots: list[dict[str, object]]) -> None:
        self._snapshots = snapshots

    def env_method(self, name: str):
        if name != "get_training_diagnostics":
            raise AssertionError(f"Unexpected env_method {name}")
        return self._snapshots


class DummyCurriculum:
    def snapshot(self) -> dict[str, object]:
        return {
            "slippage_mode": "staircase",
            "current_slippage_pips": 0.1,
            "current_phase": 1,
            "entropy_coef": 0.02,
            "participation_bonus_enabled": True,
            "participation_bonus_active": True,
        }


class DummyHeartbeatModel:
    def __init__(self, env: DummyTrainingEnv) -> None:
        self._env = env
        self.saved_paths: list[str] = []
        self._vecnormalize = None

    def get_env(self) -> DummyTrainingEnv:
        return self._env

    def save(self, path: str) -> None:
        self.saved_paths.append(path)
        Path(f"{path}.zip").write_bytes(b"model")

    def get_vec_normalize_env(self):
        return self._vecnormalize


class TrainRehabTests(unittest.TestCase):
    def test_finalboss_reward_strip_profiles_resolve_expected_overrides(self):
        reward_strip = _resolve_training_experiment_profile("reward_strip")
        hard_churn = _resolve_training_experiment_profile("reward_strip_hard_churn")
        hard_churn_alpha = _resolve_training_experiment_profile("reward_strip_hard_churn_alpha_gate")
        rehab_safer = _resolve_training_experiment_profile("reward_strip_rehab_safer_alpha_gate")

        self.assertEqual(0.0, reward_strip["reward_downside_risk_coef"])
        self.assertEqual(0.0, reward_strip["reward_turnover_coef"])
        self.assertEqual(0.0, reward_strip["churn_penalty_usd"])
        self.assertEqual(1000.0, reward_strip["reward_scale"])
        self.assertEqual(-10.0, reward_strip["reward_clip_low"])
        self.assertEqual(10.0, reward_strip["reward_clip_high"])
        self.assertFalse(reward_strip["participation_bonus_enabled"])
        self.assertEqual(5, hard_churn["churn_min_hold_bars"])
        self.assertEqual(3, hard_churn["churn_action_cooldown"])
        self.assertTrue(hard_churn_alpha["alpha_gate_enabled"])
        self.assertEqual("auto", hard_churn_alpha["alpha_gate_model"])
        self.assertEqual(8, rehab_safer["churn_min_hold_bars"])
        self.assertEqual(5, rehab_safer["churn_action_cooldown"])
        self.assertAlmostEqual(0.75, rehab_safer["entry_spread_z_limit"])
        self.assertTrue(rehab_safer["alpha_gate_enabled"])
        self.assertEqual("auto", rehab_safer["alpha_gate_model"])

    def test_explicit_env_var_values_beat_profile_defaults(self):
        with patch.dict("os.environ", {"TRAIN_REWARD_SCALE": "2500"}, clear=False):
            self.assertEqual(5, _apply_profile_override(5, "TRAIN_REWARD_SCALE", 1000.0))
        self.assertEqual(1000.0, _apply_profile_override(5, "TRAIN_EXPERIMENT_PROFILE_FAKE", 1000.0))

    def test_find_optimal_env_workers_stops_on_sps_regression_without_crashing(self):
        class FakeVecEnv:
            def __init__(self, env_fns):
                self.worker_count = len(env_fns)

            def close(self):
                return None

        class FakeModel:
            def __init__(self, *args, **kwargs):
                return None

            def learn(self, total_timesteps: int):
                return self

        class FakeMonitor:
            def start(self):
                return None

            def get_latest(self):
                class Snapshot:
                    cpu_pct = 50.0

                return Snapshot()

        with patch("train_agent.SubprocVecEnv", FakeVecEnv), patch(
            "sb3_contrib.MaskablePPO", FakeModel
        ), patch("train_agent.resource_monitor", FakeMonitor()), patch(
            "train_agent.time.time",
            side_effect=[0.0, 2.0, 10.0, 14.0],
        ):
            best_n = find_optimal_env_workers(
                lambda: object(),
                starting_n=8,
                max_n=12,
                step_size=4,
                target_cpu_pct=95.0,
            )

        self.assertEqual(8, best_n)

    def test_recovery_helpers_follow_staircase_and_entropy_schedule(self):
        cfg = {
            "slippage_curriculum": {
                "enabled": True,
                "mode": "staircase",
                "phases": [
                    {"until_step": 750_000, "slippage_pips": 0.1},
                    {"until_step": 1_750_000, "slippage_pips": 0.5},
                    {"until_step": 3_000_000, "slippage_pips": 1.0},
                ],
                "linear_start_pips": 0.1,
                "linear_end_pips": 1.0,
                "default_slippage_pips": 1.0,
            },
            "entropy_schedule": {
                "enabled": True,
                "initial_ent_coef": 0.02,
                "mid_ent_coef": 0.01,
                "final_ent_coef": 0.001,
                "phase_1_until": 750_000,
                "phase_2_until": 1_750_000,
            },
        }

        self.assertEqual(0.1, get_current_slippage_pips(100_000, cfg))
        self.assertEqual(1, get_current_phase(100_000, cfg))
        self.assertEqual(0.5, get_current_slippage_pips(900_000, cfg))
        self.assertEqual(2, get_current_phase(900_000, cfg))
        self.assertEqual(1.0, get_current_slippage_pips(2_000_000, cfg))
        self.assertEqual(3, get_current_phase(2_000_000, cfg))
        self.assertEqual(0.02, get_current_ent_coef(100_000, cfg))
        self.assertEqual(0.01, get_current_ent_coef(900_000, cfg))
        self.assertEqual(0.001, get_current_ent_coef(2_000_000, cfg))

    def test_baseline_gate_passes_on_predictive_synthetic_data(self):
        tmpdir = make_test_dir("baseline_pass")
        frame = make_supervised_frame(rows=900, predictive=True)
        trainable = frame.iloc[:700].copy()
        holdout = frame.iloc[700:].copy()
        folds = [
            (trainable.iloc[:450].copy(), trainable.iloc[450:550].copy()),
            (trainable.iloc[:550].copy(), trainable.iloc[550:650].copy()),
        ]
        out_path = tmpdir / "baseline_pass.json"

        try:
            report = run_baseline_research_gate(
                symbol="EURUSD",
                trainable_frame=trainable,
                holdout_frame=holdout,
                folds=folds,
                feature_cols=FEATURE_COLS,
                out_path=out_path,
            )
            self.assertTrue(report["gate_passed"])
            self.assertTrue(report["passing_models"])
            self.assertTrue(out_path.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_baseline_gate_fails_on_noise(self):
        tmpdir = make_test_dir("baseline_fail")
        frame = make_supervised_frame(rows=900, predictive=False)
        trainable = frame.iloc[:700].copy()
        holdout = frame.iloc[700:].copy()
        folds = [
            (trainable.iloc[:450].copy(), trainable.iloc[450:550].copy()),
            (trainable.iloc[:550].copy(), trainable.iloc[550:650].copy()),
        ]
        out_path = tmpdir / "baseline_fail.json"

        try:
            report = run_baseline_research_gate(
                symbol="EURUSD",
                trainable_frame=trainable,
                holdout_frame=holdout,
                folds=folds,
                feature_cols=FEATURE_COLS,
                out_path=out_path,
            )
            self.assertFalse(report["gate_passed"])
            self.assertEqual([], report["passing_models"])
            self.assertTrue(out_path.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_heartbeat_v2_writes_required_fields(self):
        tmpdir = make_test_dir("heartbeat_v2")
        out_path = tmpdir / "training_heartbeat.json"
        callback = TrainingDiagnosticsCallback(verbose=0)
        callback.metrics["train/approx_kl"] = [0.02]
        callback.metrics["train/explained_variance"] = [0.35]
        callback.metrics["train/value_loss"] = [1.0]
        callback._last_update_index = 7

        run_context_path = tmpdir / "current_training_run.json"
        heartbeat = TrainingHeartbeatCallback(
            out_path=out_path,
            diagnostics_cb=callback,
            curriculum_cb=DummyCurriculum(),
            eval_cb=None,
            run_id="run-123",
            symbol="USDJPY",
            checkpoints_root=tmpdir,
            fold_index=1,
            current_run_path=run_context_path,
            every_steps=10,
        )
        heartbeat.model = DummyHeartbeatModel(
            DummyTrainingEnv(
                [
                    {
                        "total_steps": 10,
                        "action_counts": {"hold": 7, "close": 1, "long": 1, "short": 1},
                        "trade_stats": {
                            "entered_long_count": 1,
                            "entered_short_count": 1,
                            "closed_trade_count": 1,
                            "trade_attempt_count": 3,
                            "trade_reject_count": 0,
                            "flat_steps": 7,
                            "long_steps": 2,
                            "short_steps": 1,
                            "position_duration_sum": 4.0,
                            "position_duration_count": 1,
                            "rapid_reversals": 0,
                            "position_durations_sample": [4],
                        },
                        "reward_components": {
                            "pnl_reward_sum": 1.5,
                            "slippage_penalty_sum": -0.2,
                            "participation_bonus_sum": 0.05,
                            "holding_penalty_sum": 0.0,
                            "drawdown_penalty_sum": -0.1,
                            "net_reward_sum": 1.25,
                        },
                    }
                ]
            )
        )
        heartbeat.num_timesteps = 10
        heartbeat._next_write = 10

        try:
            self.assertTrue(heartbeat._on_step())
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(2, payload["schema_version"])
            self.assertEqual("run-123", payload["run_id"])
            self.assertEqual("USDJPY", payload["symbol"])
            self.assertEqual(1, payload["fold_index"])
            self.assertIn("process_started_utc", payload)
            self.assertEqual(10, payload["num_timesteps"])
            self.assertEqual(7, payload["n_updates"])
            self.assertEqual(1, payload["diagnostic_sample_count"])
            self.assertTrue(payload["ppo_diagnostics"]["metrics_fresh"])
            self.assertEqual(7, payload["ppo_diagnostics"]["last_distinct_update_seen"])
            self.assertEqual("stage_a_unlock", payload["training_stage"])
            self.assertEqual("staircase", payload["curriculum_state"]["slippage_mode"])
            self.assertEqual(7, payload["action_distribution"]["hold"])
            self.assertEqual(1, payload["trade_diagnostics"]["entered_long_count"])
            self.assertAlmostEqual(1.25, payload["reward_components"]["net_reward_sum"])
            run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
            self.assertEqual("run-123", run_context["run_id"])
            self.assertEqual("training", run_context["state"])
            self.assertEqual(str(out_path), run_context["heartbeat_path"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_heartbeat_writes_resume_checkpoints_when_configured(self):
        tmpdir = make_test_dir("heartbeat_resume_checkpoint")
        out_path = tmpdir / "training_heartbeat.json"
        callback = TrainingDiagnosticsCallback(verbose=0)
        callback.metrics["train/approx_kl"] = [0.02]
        callback.metrics["train/explained_variance"] = [0.35]
        callback.metrics["train/value_loss"] = [1.0]
        callback._last_update_index = 7

        class DummyVecNormalize:
            def __init__(self) -> None:
                self.saved_paths: list[str] = []

            def save(self, path: str) -> None:
                self.saved_paths.append(path)
                Path(path).write_bytes(b"vec")

        resume_model_path = tmpdir / "resume_model.zip"
        resume_vecnormalize_path = tmpdir / "resume_vecnormalize.pkl"
        heartbeat = TrainingHeartbeatCallback(
            out_path=out_path,
            diagnostics_cb=callback,
            curriculum_cb=DummyCurriculum(),
            eval_cb=None,
            run_id="run-456",
            symbol="EURUSD",
            checkpoints_root=tmpdir,
            fold_index=0,
            current_run_path=tmpdir / "current_training_run.json",
            every_steps=10,
            resume_model_path=resume_model_path,
            resume_vecnormalize_path=resume_vecnormalize_path,
        )
        model = DummyHeartbeatModel(DummyTrainingEnv([]))
        model._vecnormalize = DummyVecNormalize()
        heartbeat.model = model
        heartbeat.num_timesteps = 10
        heartbeat._next_write = 10

        try:
            self.assertTrue(heartbeat._on_step())
            self.assertEqual([str(resume_model_path.with_suffix(""))], model.saved_paths)
            self.assertTrue(resume_model_path.exists())
            self.assertTrue(resume_vecnormalize_path.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_full_path_eval_reports_segment_metrics_without_fake_stddev(self):
        timestamps = list(pd.date_range("2024-01-01", periods=9, freq="h", tz="UTC"))
        env = DummyEvalEnv(
            equities=[1000.0, 1001.0, 1002.0, 1001.5, 1003.0, 1004.0, 1005.0, 1004.5, 1006.0],
            rewards=[0.1, 0.2, 0.1, -0.1, 0.3, 0.1, 0.2, -0.1, 0.4],
            timestamps=timestamps,
        )

        curve, metrics = evaluate_model(DummyModel(), env)

        self.assertEqual(9, len(curve))
        self.assertEqual(9, metrics["steps"])
        self.assertIn("segment_metrics", metrics)
        self.assertEqual({"first", "middle", "last"}, set(metrics["segment_metrics"].keys()))
        self.assertNotIn("reward_std", metrics)
        self.assertNotIn("std_reward", metrics)

    def test_candidate_ranking_prefers_holdout_eligible_fold_over_better_validation_only(self):
        rejected = {
            "deploy_ready": False,
            "val_sharpe": 1.20,
            "holdout_sharpe": 0.10,
        }
        promoted = {
            "deploy_ready": True,
            "val_sharpe": 0.45,
            "holdout_sharpe": 0.55,
            "holdout_profit_factor": 1.20,
        }

        self.assertIsNone(_deployment_candidate_rank(rejected))
        self.assertEqual((0.55, 1.20, 0.45), _deployment_candidate_rank(promoted))

    def test_holdout_gate_requires_sharpe_drawdown_and_nonnegative_equity(self):
        blockers = _holdout_deployment_blockers(
            holdout_sharpe=0.10,
            holdout_max_drawdown=0.35,
            holdout_final_equity=999.0,
            holdout_profit_factor=1.50,
            holdout_expectancy=0.05,
            holdout_trade_count=50,
        )

        self.assertEqual(3, len(blockers))
        self.assertTrue(any("Holdout timed Sharpe" in blocker for blocker in blockers))
        self.assertTrue(any("Holdout max drawdown" in blocker for blocker in blockers))
        self.assertTrue(any("Holdout final equity" in blocker for blocker in blockers))


if __name__ == "__main__":
    unittest.main()
