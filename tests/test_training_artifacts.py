from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from artifact_manifest import create_manifest, load_manifest, save_manifest
from evaluate_oos import _resolve_execution_cost_profile, _resolve_reward_profile
from runtime_common import ActionSpec, ActionType, build_trade_metric_reconciliation, runtime_options_from_training_payload
from train_agent import (
    AdaptiveKLLearningRateCallback,
    TrainingDiagnosticsCallback,
    _archive_paths,
    _baseline_competition_blockers,
    _build_promoted_training_diagnostics,
    _candidate_scaler_artifact_path,
    _clear_legacy_checkpoint_artifacts,
    _deployment_candidate_rank,
    _holdout_deployment_blockers,
    _load_training_resume_state,
    _publish_primary_candidate_artifacts,
    _recover_completed_fold_state,
    _resume_model_checkpoint_path,
    _resume_vecnormalize_checkpoint_path,
    _write_fold_training_diagnostics,
)


TEST_TMP_ROOT = Path("tests/tmp")


def make_test_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class TrainingArtifactTests(unittest.TestCase):
    def test_write_fold_training_diagnostics_persists_expected_path(self):
        tmpdir = make_test_dir("fold_diagnostics")
        try:
            payload = {"deploy_ready": False, "val_sharpe": -0.1}
            out_path = _write_fold_training_diagnostics(tmpdir / "fold_0", payload)

            self.assertEqual(tmpdir / "fold_0" / "training_diagnostics.json", out_path)
            self.assertEqual(payload, json.loads(out_path.read_text(encoding="utf-8")))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_promoted_diagnostics_always_include_run_metadata(self):
        diagnostics = {
            "train_bars": 5000,
            "holdout_bars": 500,
            "point_in_time_verified": True,
            "dataset_integrity_verified": True,
        }
        payload = _build_promoted_training_diagnostics(
            diagnostics,
            run_id="run-123",
            artifact_candidate_selected=False,
            artifact_candidate_reason="Economics gate failed.",
        )
        self.assertEqual("run-123", payload["run_id"])
        self.assertFalse(payload["artifact_candidate_selected"])
        self.assertEqual("Economics gate failed.", payload["artifact_candidate_reason"])
        self.assertEqual(2000, payload["bar_construction_ticks_per_bar"])
        self.assertEqual(2000, payload["ticks_per_bar"])

    def test_archive_paths_moves_stale_files_under_archive_root(self):
        tmpdir = make_test_dir("train_artifacts")
        archive_root = tmpdir / "archive"
        created_paths = [
            tmpdir / "model.zip",
            tmpdir / "vecnormalize.pkl",
            tmpdir / "manifest.json",
            tmpdir / "diagnostics.json",
        ]
        for path in created_paths:
            path.write_text("stale", encoding="utf-8")

        try:
            archived = _archive_paths(created_paths, archive_root=archive_root)
            self.assertGreaterEqual(len(archived), len(created_paths))
            for path in created_paths:
                self.assertFalse(path.exists())
            self.assertTrue(archive_root.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_clear_legacy_checkpoint_artifacts_archives_fold_dirs(self):
        tmpdir = make_test_dir("legacy_checkpoint_archive")
        checkpoints_dir = tmpdir / "checkpoints"
        legacy_fold = checkpoints_dir / "fold_legacy_test"
        legacy_file = checkpoints_dir / "best_model.zip"
        legacy_fold.mkdir(parents=True, exist_ok=True)
        (legacy_fold / "training_heartbeat.json").write_text("{}", encoding="utf-8")
        legacy_file.write_text("old-model", encoding="utf-8")

        try:
            archived = _clear_legacy_checkpoint_artifacts(run_id="test-run", checkpoints_root=checkpoints_dir)
            self.assertTrue(any("fold_legacy_test" in item for item in archived))
            self.assertTrue(any("best_model.zip" in item for item in archived))
            self.assertFalse(legacy_fold.exists())
            self.assertFalse(legacy_file.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_training_diagnostics_summary_is_json_serializable(self):
        callback = TrainingDiagnosticsCallback(verbose=0)
        callback.metrics["train/approx_kl"] = [0.02]
        callback.metrics["train/explained_variance"] = [0.35]
        callback.metrics["train/value_loss"] = [1.0, 1.1, 0.9]

        summary = callback.summary()

        self.assertIsInstance(summary["value_loss_stable"], bool)
        json.dumps(summary)

    def test_training_diagnostics_records_each_logger_update_once(self):
        callback = TrainingDiagnosticsCallback(verbose=0)
        callback.model = SimpleNamespace(
            logger=SimpleNamespace(
                name_to_value={
                    "train/approx_kl": 0.02,
                    "train/explained_variance": 0.15,
                    "train/value_loss": 1.5,
                    "train/n_updates": 10,
                }
            )
        )

        callback._on_rollout_end()
        callback._on_rollout_end()

        self.assertEqual([0.02], callback.metrics["train/approx_kl"])
        self.assertEqual([0.15], callback.metrics["train/explained_variance"])
        self.assertEqual([1.5], callback.metrics["train/value_loss"])

    def test_adaptive_kl_callback_applies_constant_lr_schedule_after_adjustment(self):
        optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])

        class _DummyModel:
            def __init__(self):
                self.learning_rate = 0.0
                self.lr_schedule = lambda _progress_remaining: 0.0
                self.policy = SimpleNamespace(optimizer=optimizer)
                self.logger = SimpleNamespace(
                    name_to_value={
                        "train/approx_kl": 1e-6,
                        "train/n_updates": 1,
                    }
                )

            def _update_learning_rate(self, optimizer_obj):
                lr = float(self.lr_schedule(0.25))
                for param_group in optimizer_obj.param_groups:
                    param_group["lr"] = lr

        callback = AdaptiveKLLearningRateCallback(
            min_lr=1e-5,
            max_lr=1e-3,
            low_kl=0.005,
            high_kl=0.05,
            up_multiplier=2.0,
            down_multiplier=0.5,
            verbose=0,
        )
        callback.model = _DummyModel()
        callback.num_timesteps = 4096

        callback._on_training_start()
        callback._on_rollout_end()

        expected_lr = min(4e-4 * 2.0, 1e-3)
        self.assertAlmostEqual(expected_lr, callback.current_base_lr)
        self.assertAlmostEqual(expected_lr, callback.model.learning_rate)
        self.assertAlmostEqual(expected_lr, callback.model.lr_schedule(0.9))
        self.assertAlmostEqual(expected_lr, callback.model.lr_schedule(0.1))
        self.assertAlmostEqual(expected_lr, optimizer.param_groups[0]["lr"])

    def test_baseline_competition_blockers_require_rl_to_match_positive_baseline(self):
        blockers = _baseline_competition_blockers(
            current_metrics={"net_pnl_usd": -10.0, "expectancy_usd": -0.5},
            baseline_metrics={"trade_count": 12, "net_pnl_usd": 25.0, "expectancy_usd": 1.2},
        )
        self.assertTrue(any("net pnl" in blocker.lower() for blocker in blockers))
        self.assertTrue(any("expectancy" in blocker.lower() for blocker in blockers))

    def test_baseline_competition_blockers_ignore_non_profitable_baseline(self):
        blockers = _baseline_competition_blockers(
            current_metrics={"net_pnl_usd": -10.0, "expectancy_usd": -0.5},
            baseline_metrics={"trade_count": 12, "net_pnl_usd": -25.0, "expectancy_usd": -1.2},
        )
        self.assertEqual([], blockers)

    def test_holdout_blockers_include_trade_quality_checks(self):
        blockers = _holdout_deployment_blockers(
            holdout_sharpe=0.35,
            holdout_max_drawdown=0.10,
            holdout_final_equity=1005.0,
            holdout_profit_factor=1.02,
            holdout_expectancy=-0.01,
            holdout_trade_count=3,
        )
        self.assertTrue(any("profit factor" in blocker.lower() for blocker in blockers))
        self.assertTrue(any("expectancy" in blocker.lower() for blocker in blockers))
        self.assertTrue(any("trades" in blocker.lower() for blocker in blockers))

    def test_candidate_rank_prefers_holdout_profit_factor_after_sharpe(self):
        strong = _deployment_candidate_rank(
            {
                "deploy_ready": True,
                "holdout_sharpe": 0.40,
                "holdout_profit_factor": 1.25,
                "val_sharpe": 0.10,
            }
        )
        weak = _deployment_candidate_rank(
            {
                "deploy_ready": True,
                "holdout_sharpe": 0.40,
                "holdout_profit_factor": 1.05,
                "val_sharpe": 0.20,
            }
        )
        self.assertGreater(strong, weak)

    def test_load_training_resume_state_uses_current_run_checkpoint_paths(self):
        tmpdir = make_test_dir("resume_state")
        try:
            checkpoints_root = tmpdir / "run_resume"
            ckpt_dir = checkpoints_root / "fold_1"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            resume_model_path = _resume_model_checkpoint_path(ckpt_dir)
            resume_vecnormalize_path = _resume_vecnormalize_checkpoint_path(ckpt_dir)
            resume_model_path.write_bytes(b"model")
            resume_vecnormalize_path.write_bytes(b"vec")
            current_run_path = tmpdir / "current_training_run.json"
            current_run_path.write_text(
                json.dumps(
                    {
                        "run_id": "run-resume",
                        "symbol": "GBPUSD",
                        "checkpoints_root": str(checkpoints_root),
                        "fold_index": 1,
                        "state": "training",
                        "num_timesteps": 12345,
                    }
                ),
                encoding="utf-8",
            )

            state = _load_training_resume_state(
                symbol="GBPUSD",
                total_timesteps=50000,
                current_run_path=current_run_path,
            )

            self.assertIsNotNone(state)
            self.assertEqual("run-resume", state["run_id"])
            self.assertEqual(1, state["fold_index"])
            self.assertEqual(resume_model_path, state["model_path"])
            self.assertEqual(resume_vecnormalize_path, state["vecnormalize_path"])
            self.assertEqual(50000 - 12345, state["remaining_timesteps_hint"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_publish_primary_candidate_artifacts_archives_existing_canonical_files(self):
        tmpdir = make_test_dir("publish_candidate")
        model_dir = tmpdir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = tmpdir / "dataset.csv"
        dataset_path.write_text("timestamp,Close\n", encoding="utf-8")

        model_artifact_path = model_dir / "model_eurusd_best.zip"
        vecnormalize_artifact_path = model_dir / "model_eurusd_best_vecnormalize.pkl"
        canonical_scaler_path = model_dir / "scaler_EURUSD.pkl"
        canonical_manifest_path = model_dir / "artifact_manifest_EURUSD.json"
        default_manifest_path = model_dir / "artifact_manifest.json"
        canonical_diagnostics_path = model_dir / "training_diagnostics_eurusd.json"
        canonical_gate_path = model_dir / "gate_report_eurusd.json"
        canonical_live_path = model_dir / "live_preflight_eurusd.json"
        canonical_ops_path = model_dir / "ops_attestation_eurusd.json"
        for path, content in (
            (model_artifact_path, "old-model"),
            (vecnormalize_artifact_path, "old-vec"),
            (canonical_scaler_path, "old-scaler"),
            (canonical_manifest_path, "{}"),
            (default_manifest_path, "{}"),
            (canonical_diagnostics_path, "{\"canonical\": true}"),
            (canonical_gate_path, "{}"),
            (canonical_live_path, "{}"),
            (canonical_ops_path, "{}"),
        ):
            path.write_text(content, encoding="utf-8")

        candidate_model_source = tmpdir / "candidate_model.zip"
        candidate_vecnormalize_source = tmpdir / "candidate_vec.pkl"
        candidate_scaler_source = tmpdir / "candidate_scaler.pkl"
        candidate_model_source.write_text("new-model", encoding="utf-8")
        candidate_vecnormalize_source.write_text("new-vec", encoding="utf-8")
        candidate_scaler_source.write_text("new-scaler", encoding="utf-8")

        try:
            with patch("train_agent.TRAIN_MODEL_DIR", model_dir):
                archived_paths = _publish_primary_candidate_artifacts(
                    primary_symbol="EURUSD",
                    model_artifact_path=model_artifact_path,
                    vecnormalize_artifact_path=vecnormalize_artifact_path,
                    candidate_model_source=candidate_model_source,
                    candidate_vecnormalize_source=candidate_vecnormalize_source,
                    candidate_scaler_source=candidate_scaler_source,
                    holdout_start_utc="2024-01-01T00:00:00+00:00",
                    dataset_path=dataset_path,
                    run_id="run-archive-test",
                    execution_cost_profile={"slippage_pips": 1.0, "commission_per_lot": 7.0},
                    reward_profile={"minimal_post_cost_reward": True},
                )

            archive_root = model_dir / "archive" / "eurusd" / "run-archive-test"
            self.assertTrue(archived_paths)
            self.assertTrue(archive_root.exists())
            self.assertEqual("new-model", model_artifact_path.read_text(encoding="utf-8"))
            self.assertEqual("new-vec", vecnormalize_artifact_path.read_text(encoding="utf-8"))
            self.assertEqual("new-scaler", canonical_scaler_path.read_text(encoding="utf-8"))
            self.assertTrue((archive_root / "training_diagnostics_eurusd.json").exists())
            self.assertTrue((archive_root / "model_eurusd_best.zip").exists())
            self.assertTrue(canonical_manifest_path.exists())
            self.assertTrue(default_manifest_path.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_runtime_options_from_training_payload_preserves_truth_flags(self):
        options = runtime_options_from_training_payload(
            {
                "training_window_size": 8,
                "training_alpha_gate_enabled": True,
                "training_minimal_post_cost_reward": True,
                "training_force_fast_window_benchmark": True,
            }
        )

        self.assertEqual(8, options["window_size"])
        self.assertTrue(options["alpha_gate_enabled"])
        self.assertTrue(options["minimal_post_cost_reward"])
        self.assertTrue(options["force_fast_window_benchmark"])

    def test_load_training_resume_state_ignores_terminal_stopped_and_collapsed_runs(self):
        tmpdir = make_test_dir("resume_state_terminal")
        try:
            checkpoints_root = tmpdir / "run_resume"
            ckpt_dir = checkpoints_root / "fold_1"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            _resume_model_checkpoint_path(ckpt_dir).write_bytes(b"model")

            for state_name in ("stopped", "collapsed"):
                current_run_path = tmpdir / f"current_training_run_{state_name}.json"
                current_run_path.write_text(
                    json.dumps(
                        {
                            "run_id": f"run-{state_name}",
                            "symbol": "GBPUSD",
                            "checkpoints_root": str(checkpoints_root),
                            "fold_index": 1,
                            "state": state_name,
                            "num_timesteps": 12345,
                        }
                    ),
                    encoding="utf-8",
                )

                state = _load_training_resume_state(
                    symbol="GBPUSD",
                    total_timesteps=50000,
                    current_run_path=current_run_path,
                )

                self.assertIsNone(state)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_recover_completed_fold_state_reuses_prior_candidate_artifacts(self):
        tmpdir = make_test_dir("recover_fold_state")
        try:
            checkpoints_root = tmpdir / "run_resume"
            fold_dir = checkpoints_root / "fold_0"
            fold_dir.mkdir(parents=True, exist_ok=True)
            diagnostics = {
                "deploy_ready": True,
                "val_sharpe": 0.45,
                "holdout_sharpe": 0.55,
                "holdout_profit_factor": 1.20,
                "bar_construction_ticks_per_bar": 2000,
            }
            (fold_dir / "training_diagnostics.json").write_text(
                json.dumps(diagnostics),
                encoding="utf-8",
            )
            candidate_model_path = fold_dir / "deployment_candidate_model.zip"
            candidate_model_path.write_bytes(b"model")
            candidate_scaler_path = _candidate_scaler_artifact_path(fold_dir, "EURUSD")
            candidate_scaler_path.write_bytes(b"scaler")

            (
                best_observed_sharpe,
                best_observed_summary,
                candidate_rank,
                candidate_summary,
                candidate_model_source,
                _candidate_vecnormalize_source,
                candidate_scaler_source,
            ) = _recover_completed_fold_state(
                run_id="run-resume",
                checkpoints_root=checkpoints_root,
                primary_symbol="EURUSD",
            )

            self.assertAlmostEqual(0.45, best_observed_sharpe)
            self.assertIsNotNone(best_observed_summary)
            self.assertEqual((0.55, 1.20, 0.45), candidate_rank)
            self.assertIsNotNone(candidate_summary)
            self.assertEqual(candidate_model_path, candidate_model_source)
            self.assertEqual(candidate_scaler_path, candidate_scaler_source)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_manifest_persists_execution_and_reward_profiles(self):
        tmpdir = make_test_dir("manifest_profiles")
        try:
            model_path = tmpdir / "model.zip"
            scaler_path = tmpdir / "scaler.pkl"
            dataset_path = tmpdir / "dataset.csv"
            model_path.write_bytes(b"model")
            scaler_path.write_bytes(b"scaler")
            dataset_path.write_text("Gmt time,Symbol,Open,High,Low,Close,Volume\n", encoding="utf-8")

            manifest = create_manifest(
                strategy_symbol="EURUSD",
                model_path=model_path,
                scaler_path=scaler_path,
                model_version="test-v1",
                feature_columns=["feature"],
                observation_shape=[1, 1],
                action_map=[ActionSpec(ActionType.HOLD)],
                dataset_path=dataset_path,
                execution_cost_profile={
                    "commission_per_lot": 7.0,
                    "slippage_pips": 0.5,
                    "partial_fill_ratio": 1.0,
                },
                reward_profile={
                    "reward_scale": 10000.0,
                    "drawdown_penalty": 2.0,
                    "transaction_penalty": 1.0,
                    "reward_clip_low": -5.0,
                    "reward_clip_high": 5.0,
                },
            )
            manifest_path = tmpdir / "artifact_manifest.json"
            save_manifest(manifest, manifest_path)

            reloaded = load_manifest(manifest_path)

            self.assertEqual(manifest.execution_cost_profile, reloaded.execution_cost_profile)
            self.assertEqual(manifest.reward_profile, reloaded.reward_profile)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_evaluate_oos_profile_helpers_fall_back_and_read_manifest_overrides(self):
        manifest = SimpleNamespace(
            execution_cost_profile={"commission_per_lot": 6.5, "slippage_pips": 0.5, "partial_fill_ratio": 0.75},
            reward_profile={
                "reward_scale": 9000.0,
                "drawdown_penalty": 1.5,
                "transaction_penalty": 0.8,
                "reward_clip_low": -4.0,
                "reward_clip_high": 4.0,
            },
        )
        self.assertEqual(
            {
                "commission_per_lot": 6.5,
                "slippage_pips": 0.5,
                "partial_fill_ratio": 0.75,
            },
            _resolve_execution_cost_profile(manifest),
        )
        self.assertEqual(
            {
                "reward_scale": 9000.0,
                "drawdown_penalty": 1.5,
                "transaction_penalty": 0.8,
                "reward_clip_low": -4.0,
                "reward_clip_high": 4.0,
                "minimal_post_cost_reward": False,
            },
            _resolve_reward_profile(manifest),
        )
        self.assertEqual(
            {
                "commission_per_lot": 7.0,
                "slippage_pips": 0.25,
                "partial_fill_ratio": 1.0,
            },
            _resolve_execution_cost_profile(SimpleNamespace(execution_cost_profile=None)),
        )

    def test_trade_metric_reconciliation_detects_mismatch(self):
        trade_metrics = {
            "trade_count": 3.0,
            "gross_pnl_usd": 12.0,
            "net_pnl_usd": 7.0,
            "total_transaction_cost_usd": 5.0,
            "total_commission_usd": 2.0,
            "total_spread_slippage_cost_usd": 3.0,
            "total_spread_cost_usd": 1.0,
            "total_slippage_cost_usd": 2.0,
            "forced_close_count": 1.0,
            "avg_holding_bars": 4.0,
        }

        matched = build_trade_metric_reconciliation(
            trade_metrics=trade_metrics,
            trade_diagnostics={
                "closed_trade_count": 3,
                "forced_close_count": 1,
                "order_executed_count": 6,
                "position_duration_sum": 12.0,
                "position_duration_count": 3,
            },
            economics={
                "gross_pnl_usd": 12.0,
                "net_pnl_usd": 7.0,
                "transaction_cost_usd": 5.0,
                "commission_usd": 2.0,
                "spread_slippage_cost_usd": 3.0,
                "spread_cost_usd": 1.0,
                "slippage_cost_usd": 2.0,
            },
            trade_log_count=3,
            execution_log_count=6,
        )
        self.assertTrue(matched["passed"])

        mismatched = build_trade_metric_reconciliation(
            trade_metrics=trade_metrics,
            trade_diagnostics={
                "closed_trade_count": 2,
                "forced_close_count": 0,
                "order_executed_count": 5,
                "position_duration_sum": 9.0,
                "position_duration_count": 2,
            },
            economics={
                "gross_pnl_usd": 12.0,
                "net_pnl_usd": 6.5,
                "transaction_cost_usd": 5.5,
                "commission_usd": 2.0,
                "spread_slippage_cost_usd": 3.0,
                "spread_cost_usd": 1.0,
                "slippage_cost_usd": 2.0,
            },
            trade_log_count=3,
            execution_log_count=6,
        )
        self.assertFalse(mismatched["passed"])
        self.assertIn("trade_count_vs_diagnostics", mismatched["mismatch_fields"])
        self.assertIn("executed_order_count_vs_execution_log", mismatched["mismatch_fields"])
        self.assertIn("net_pnl_usd_vs_diagnostics", mismatched["mismatch_fields"])


if __name__ == "__main__":
    unittest.main()
