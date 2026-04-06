from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import training_status


TEST_TMP_ROOT = Path("tests/tmp")


def make_test_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class TrainingStatusToolTests(unittest.TestCase):
    def test_build_status_summary_falls_back_to_checkpoint_diagnostics(self) -> None:
        tmpdir = make_test_dir("training_status_fallback")
        try:
            checkpoints_dir = tmpdir / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            current_run_path = checkpoints_dir / "current_training_run.json"
            run_root = checkpoints_dir / "run_123"
            fold_dir = run_root / "fold_0"
            fold_dir.mkdir(parents=True, exist_ok=True)
            diagnostics_path = fold_dir / "training_diagnostics.json"
            diagnostics_path.write_text(
                json.dumps({"gate_passed": False, "blockers": ["economics failed"], "val_sharpe": -0.1}),
                encoding="utf-8",
            )
            current_run_path.write_text(
                json.dumps(
                    {
                        "run_id": "run-123",
                        "symbol": "EURUSD",
                        "checkpoints_root": str(run_root),
                        "state": "completed",
                    }
                ),
                encoding="utf-8",
            )
            fake_paths = SimpleNamespace(
                diagnostics_path=tmpdir / "models" / "training_diagnostics_eurusd.json",
                gate_path=tmpdir / "models" / "deployment_gate_eurusd.json",
                ops_attestation_path=tmpdir / "models" / "ops_attestation_eurusd.json",
            )

            with patch.object(training_status, "deployment_paths", return_value=fake_paths):
                summary = training_status.build_status_summary("EURUSD", checkpoints_dir)

            self.assertEqual(str(diagnostics_path), summary["training_diagnostics_path"])
            self.assertIsNotNone(summary["training_diagnostics"])
            self.assertFalse(summary["training_diagnostics"]["gate_passed"])
            self.assertIn("economics failed", summary["failures"]["training_blockers"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_profitability_summary_surfaces_cost_drag_and_baseline_gap(self) -> None:
        heartbeat = {
            "latest_eval": {
                "trade_count": 12,
                "gross_pnl_usd": 15.0,
                "net_pnl_usd": -5.0,
                "total_transaction_cost_usd": 20.0,
                "execution_diagnostics": {
                    "reward_components": {
                        "net_reward_sum": -50.0,
                        "downside_risk_penalty_sum": -12.5,
                    }
                },
            }
        }
        baseline_report = {
            "holdout_metrics": {
                "models": {
                    "mean_reversion": {"metrics": {"net_pnl_usd": 10.0, "profit_factor": 1.1, "expectancy_usd": 1.0}},
                    "flat": {"metrics": {"net_pnl_usd": 0.0, "profit_factor": 0.0, "expectancy_usd": 0.0}},
                }
            }
        }

        summary = training_status.build_profitability_summary(heartbeat, baseline_report)

        self.assertTrue(summary["latest_eval_cost_exceeds_gross_profit"])
        self.assertEqual("mean_reversion", summary["best_holdout_baseline"])
        self.assertAlmostEqual(-15.0, summary["latest_eval_vs_best_holdout_baseline_net_pnl_gap_usd"])


if __name__ == "__main__":
    unittest.main()
