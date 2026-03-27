from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from feature_engine import FEATURE_COLS
from train_agent import (
    TrainingDiagnosticsCallback,
    TrainingHeartbeatCallback,
    _deployment_candidate_rank,
    _holdout_deployment_blockers,
    evaluate_model,
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


class TrainRehabTests(unittest.TestCase):
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
            run_id="run-123",
            symbol="USDJPY",
            checkpoints_root=tmpdir,
            fold_index=1,
            current_run_path=run_context_path,
            every_steps=10,
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
            run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
            self.assertEqual("run-123", run_context["run_id"])
            self.assertEqual("training", run_context["state"])
            self.assertEqual(str(out_path), run_context["heartbeat_path"])
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
        }

        self.assertIsNone(_deployment_candidate_rank(rejected))
        self.assertEqual((0.55, 0.45), _deployment_candidate_rank(promoted))

    def test_holdout_gate_requires_sharpe_drawdown_and_nonnegative_equity(self):
        blockers = _holdout_deployment_blockers(
            holdout_sharpe=0.10,
            holdout_max_drawdown=0.35,
            holdout_final_equity=999.0,
        )

        self.assertEqual(3, len(blockers))
        self.assertTrue(any("Holdout timed Sharpe" in blocker for blocker in blockers))
        self.assertTrue(any("Holdout max drawdown" in blocker for blocker in blockers))
        self.assertTrue(any("Holdout final equity" in blocker for blocker in blockers))


if __name__ == "__main__":
    unittest.main()
