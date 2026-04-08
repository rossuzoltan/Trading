from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from build_h1_dataset import _coverage_summary
from dataset_validation import validate_symbol_bar_spec
from project_paths import validate_dataset_bar_spec
from trading_config import resolve_bar_construction_ticks_per_bar
from validation_metrics import assess_training_data_sufficiency, build_deployment_gate


class DataQualityGateTests(unittest.TestCase):
    def _deployment_ready_training_diagnostics(self, **overrides):
        payload = {
            "blockers": [],
            "gate_passed": True,
            "baseline_gate_passed": True,
            "eval_protocol_valid": True,
            "full_path_eval_used": True,
            "train_bars": 6000,
            "val_bars": 250,
            "holdout_bars": 600,
            "point_in_time_verified": True,
            "dataset_integrity_verified": True,
        }
        payload.update(overrides)
        return payload

    def test_coverage_summary_detects_large_gap(self):
        index = pd.to_datetime(
            [
                "2020-01-01T00:00:00Z",
                "2020-01-01T01:00:00Z",
                "2020-01-05T12:00:00Z",
            ],
            utc=True,
        )
        bars = pd.DataFrame({"Close": [1.0, 1.1, 1.2]}, index=index)
        summary = _coverage_summary(
            bars,
            requested_start=pd.Timestamp("2020-01-01T00:00:00Z"),
            requested_end=pd.Timestamp("2020-01-10T00:00:00Z"),
            min_years=0.01,
            max_gap_hours=24,
        )
        self.assertEqual(summary["gap_count_over_threshold"], 1)
        self.assertFalse(summary["meets_gap_threshold"])

    def test_deployment_gate_hard_blocks_low_data_and_unverified_integrity(self):
        gate = build_deployment_gate(
            symbol="EURUSD",
            replay_metrics={
                "timed_sharpe": 0.5,
                "max_drawdown": 0.1,
                "profit_factor": 1.5,
                "expectancy": 1.0,
            },
            training_diagnostics={
                "blockers": [],
                "gate_passed": True,
                "train_bars": 100,
                "holdout_bars": 50,
                "point_in_time_verified": False,
                "dataset_integrity_verified": False,
            },
        )
        self.assertIn("Train bars 100 < required 5000", gate["blockers"])
        self.assertIn("Holdout bars 50 < required 500", gate["blockers"])
        self.assertIn("Point-in-time integrity is not verified.", gate["blockers"])
        self.assertIn("Dataset integrity is not verified.", gate["blockers"])

    def test_deployment_gate_blocks_missing_or_false_eval_schema_fields(self):
        replay_metrics = {
            "timed_sharpe": 0.5,
            "max_drawdown": 0.1,
            "profit_factor": 1.5,
            "expectancy": 1.0,
        }
        cases = [
            (
                "missing_baseline_gate_passed",
                lambda: self._deployment_ready_training_diagnostics(),
                "Training diagnostics missing baseline_gate_passed.",
            ),
            (
                "missing_eval_protocol_valid",
                lambda: self._deployment_ready_training_diagnostics(eval_protocol_valid=None),
                "Training diagnostics missing eval_protocol_valid.",
            ),
            (
                "missing_full_path_eval_used",
                lambda: self._deployment_ready_training_diagnostics(full_path_eval_used=None),
                "Training diagnostics missing full_path_eval_used.",
            ),
            (
                "baseline_gate_passed_false",
                lambda: self._deployment_ready_training_diagnostics(baseline_gate_passed=False),
                "Training diagnostics baseline_gate_passed must be true.",
            ),
            (
                "eval_protocol_valid_false",
                lambda: self._deployment_ready_training_diagnostics(eval_protocol_valid=False),
                "Training diagnostics eval_protocol_valid must be true.",
            ),
            (
                "full_path_eval_used_false",
                lambda: self._deployment_ready_training_diagnostics(full_path_eval_used=False),
                "Training diagnostics full_path_eval_used must be true.",
            ),
            (
                "legacy_schema_missing_all_new_fields",
                lambda: {
                    "blockers": [],
                    "gate_passed": True,
                    "train_bars": 6000,
                    "val_bars": 250,
                    "holdout_bars": 600,
                    "point_in_time_verified": True,
                    "dataset_integrity_verified": True,
                },
                "Training diagnostics missing baseline_gate_passed.",
            ),
        ]

        with patch.dict("os.environ", {"LIVE_REQUIRE_OPS_ATTESTATION": "0"}, clear=False):
            for case_name, payload_factory, expected_blocker in cases:
                with self.subTest(case_name=case_name):
                    training_diagnostics = payload_factory()
                    if case_name == "missing_baseline_gate_passed":
                        training_diagnostics.pop("baseline_gate_passed")
                    elif case_name == "missing_eval_protocol_valid":
                        training_diagnostics.pop("eval_protocol_valid")
                    elif case_name == "missing_full_path_eval_used":
                        training_diagnostics.pop("full_path_eval_used")
                    gate = build_deployment_gate(
                        symbol="EURUSD",
                        replay_metrics=replay_metrics,
                        training_diagnostics=training_diagnostics,
                    )
                    self.assertFalse(gate["approved_for_live"])
                    self.assertIn(expected_blocker, gate["blockers"])

    def test_deployment_gate_approves_when_all_required_conditions_pass(self):
        with patch.dict("os.environ", {"LIVE_REQUIRE_OPS_ATTESTATION": "0"}, clear=False):
            gate = build_deployment_gate(
                symbol="EURUSD",
                replay_metrics={
                    "timed_sharpe": 0.5,
                    "max_drawdown": 0.1,
                    "profit_factor": 1.5,
                    "expectancy": 1.0,
                },
                training_diagnostics=self._deployment_ready_training_diagnostics(),
            )
        self.assertTrue(gate["approved_for_live"])
        self.assertEqual([], gate["blockers"])

    def test_deployment_gate_blocks_replay_embedded_parity_mismatch(self):
        replay_metrics = {
            "timed_sharpe": 0.5,
            "max_drawdown": 0.1,
            "profit_factor": 1.5,
            "expectancy": 1.0,
            "runtime_parity_verdict": {
                "research_vs_runtime_parity_aligned": False,
                "fragile_under_cost_stress": False,
                "research_baseline_summary": {"research_baseline_viable": True},
            },
        }
        with patch.dict("os.environ", {"LIVE_REQUIRE_OPS_ATTESTATION": "0"}, clear=False):
            gate = build_deployment_gate(
                symbol="EURUSD",
                replay_metrics=replay_metrics,
                training_diagnostics=self._deployment_ready_training_diagnostics(),
            )
        self.assertFalse(gate["approved_for_live"])
        self.assertTrue(any("runtime-parity baselines do not" in blocker for blocker in gate["blockers"]))

    def test_deployment_gate_blocks_cost_stress_fragility_only_when_flagged(self):
        replay_metrics = {
            "timed_sharpe": 0.5,
            "max_drawdown": 0.1,
            "profit_factor": 1.5,
            "expectancy": 1.0,
            "runtime_parity_verdict": {
                "research_vs_runtime_parity_aligned": True,
                "fragile_under_cost_stress": True,
                "research_baseline_summary": {"research_baseline_viable": False},
            },
        }
        with patch.dict("os.environ", {"LIVE_REQUIRE_OPS_ATTESTATION": "0"}, clear=False):
            gate = build_deployment_gate(
                symbol="EURUSD",
                replay_metrics=replay_metrics,
                training_diagnostics=self._deployment_ready_training_diagnostics(),
            )
        self.assertFalse(gate["approved_for_live"])
        self.assertIn("Replay is profitable only under base costs and fails slippage stress.", gate["blockers"])

    def test_assess_training_data_sufficiency_includes_validation_floor(self):
        with patch.dict(
            "os.environ",
            {
                "TRAIN_MIN_TRAIN_BARS": "5000",
                "TRAIN_MIN_VAL_BARS": "200",
                "TRAIN_MIN_HOLDOUT_BARS": "500",
            },
            clear=False,
        ):
            blockers = assess_training_data_sufficiency(train_bars=4999, val_bars=199, holdout_bars=499)
        self.assertIn("Train bars 4999 < required 5000", blockers)
        self.assertIn("Validation bars 199 < required 200", blockers)
        self.assertIn("Holdout bars 499 < required 500", blockers)

    def test_purged_walk_forward_skips_small_fold_windows(self):
        try:
            from train_agent import purged_walk_forward_splits
        except Exception as exc:  # pragma: no cover - workspace import guard
            self.skipTest(f"train_agent import unavailable: {exc}")
        frame = pd.DataFrame({"Close": range(800)})
        with patch.dict(
            "os.environ",
            {
                "TRAIN_MIN_TRAIN_BARS": "5000",
                "TRAIN_MIN_VAL_BARS": "200",
            },
            clear=False,
        ):
            folds = purged_walk_forward_splits(frame, n_folds=1, test_frac=0.1, purge_gap=0)
        self.assertEqual([], folds)

    def test_single_worker_or_forced_dummy_is_marked_debug_only(self):
        try:
            from train_agent import resolve_train_vec_env_type
        except Exception as exc:  # pragma: no cover - workspace import guard
            self.skipTest(f"train_agent import unavailable: {exc}")
        vec_env_type, warnings = resolve_train_vec_env_type(
            requested_envs=1,
            effective_envs=1,
            force_dummy=True,
        )
        self.assertEqual("dummy", vec_env_type)
        self.assertTrue(any("TRAIN_NUM_ENVS=1" in warning for warning in warnings))
        self.assertTrue(any("TRAIN_FORCE_DUMMY_VEC=1" in warning for warning in warnings))

    def test_windows_subproc_disables_shared_memmap_buffers(self):
        try:
            from train_agent import resolve_shared_dataset_buffers
        except Exception as exc:  # pragma: no cover - workspace import guard
            self.skipTest(f"train_agent import unavailable: {exc}")
        enabled, note = resolve_shared_dataset_buffers(
            enabled=True,
            vec_env_type="subproc",
            platform_name="win32",
        )
        self.assertFalse(enabled)
        self.assertIsNotNone(note)
        self.assertIn("Windows SubprocVecEnv", str(note))

    def test_non_windows_or_dummy_keeps_shared_dataset_buffers(self):
        try:
            from train_agent import resolve_shared_dataset_buffers
        except Exception as exc:  # pragma: no cover - workspace import guard
            self.skipTest(f"train_agent import unavailable: {exc}")
        enabled, note = resolve_shared_dataset_buffers(
            enabled=True,
            vec_env_type="dummy",
            platform_name="win32",
        )
        self.assertTrue(enabled)
        self.assertIsNone(note)

    def test_bar_spec_default_is_2000(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(2000, resolve_bar_construction_ticks_per_bar("BAR_SPEC_TICKS_PER_BAR", "TRADING_TICKS_PER_BAR"))

    def test_validate_dataset_bar_spec_accepts_matching_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "DATA_CLEAN_VOLUME.csv"
            dataset_path.write_text("Gmt time,Symbol\n", encoding="utf-8")
            metadata_path = Path(tmp) / "dataset_build_info.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "bar_construction_ticks_per_bar": 2000,
                        "ticks_per_bar": 2000,
                    }
                ),
                encoding="utf-8",
            )
            build_info = validate_dataset_bar_spec(
                dataset_path=dataset_path,
                expected_ticks_per_bar=2000,
                metadata_path=metadata_path,
                metadata_required=True,
            )
        self.assertEqual(2000, build_info["bar_construction_ticks_per_bar"])

    def test_validate_dataset_bar_spec_rejects_mismatched_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "DATA_CLEAN_VOLUME.csv"
            dataset_path.write_text("Gmt time,Symbol\n", encoding="utf-8")
            metadata_path = Path(tmp) / "dataset_build_info.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "bar_construction_ticks_per_bar": 2000,
                        "ticks_per_bar": 2000,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError) as ctx:
                validate_dataset_bar_spec(
                    dataset_path=dataset_path,
                    expected_ticks_per_bar=250,
                    metadata_path=metadata_path,
                    metadata_required=True,
                )
        self.assertIn("built with bar_construction_ticks_per_bar=2000", str(ctx.exception))

    def test_validate_symbol_bar_spec_allows_single_tail_partial_row(self):
        frame = pd.DataFrame(
            {
                "Gmt time": pd.to_datetime(
                    ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"],
                    utc=True,
                ),
                "Volume": [2000, 2000, 1556],
            }
        )
        summary = validate_symbol_bar_spec(frame, expected_ticks_per_bar=2000, symbol="EURUSD")
        self.assertEqual(1, summary["partial_rows"])
        self.assertTrue(summary["partial_only_at_tail"])

    def test_validate_symbol_bar_spec_rejects_mixed_bar_sizes(self):
        frame = pd.DataFrame(
            {
                "Gmt time": pd.to_datetime(
                    ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"],
                    utc=True,
                ),
                "Volume": [250, 250, 250],
            }
        )
        with self.assertRaises(RuntimeError) as ctx:
            validate_symbol_bar_spec(frame, expected_ticks_per_bar=2000, symbol="USDJPY")
        self.assertIn("mix bar sizes", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
