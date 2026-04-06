from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from artifact_manifest import create_manifest, save_manifest
from feature_engine import FEATURE_COLS
from research import schema as research_schema
from runtime_common import STATE_FEATURE_COUNT, build_action_map
from tools import research_runner
from trading_config import ACTION_SL_MULTS, ACTION_TP_MULTS


TEST_TMP_ROOT = Path("tests/tmp")


def make_test_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def make_proposal(path: Path, payload: dict | None = None) -> Path:
    base_payload = {
        "experiment_name": "eurusd_reward_strip_window8",
        "symbol": "EURUSD",
        "timesteps": 120000,
        "fast_mode": True,
        "rationale": "Unit-test proposal.",
        "overrides": {
            "TRAIN_EXPERIMENT_PROFILE": "reward_strip_rehab_safer_alpha_gate",
            "TRAIN_WINDOW_SIZE": 8,
        },
        "tags": ["unit_test"],
    }
    if payload:
        base_payload.update(payload)
    return write_json(path, base_payload)


def make_manifest_bundle(artifacts_dir: Path, *, symbol: str) -> Path:
    dataset_path = artifacts_dir / "dataset.csv"
    dataset_path.write_text(
        "Gmt time,Open,High,Low,Close,Volume,Symbol\n2024-01-01T00:00:00Z,1,1,1,1,2000,"
        f"{symbol}\n",
        encoding="utf-8",
    )
    model_path = artifacts_dir / f"model_{symbol.lower()}_best.zip"
    scaler_path = artifacts_dir / f"scaler_{symbol}.pkl"
    model_path.write_bytes(b"model")
    scaler_path.write_bytes(b"scaler")
    manifest = create_manifest(
        strategy_symbol=symbol,
        model_path=model_path,
        scaler_path=scaler_path,
        model_version=f"{symbol.lower()}-test-v1",
        feature_columns=FEATURE_COLS,
        observation_shape=[1, len(FEATURE_COLS) + STATE_FEATURE_COUNT],
        action_map=build_action_map(list(ACTION_SL_MULTS), list(ACTION_TP_MULTS)),
        dataset_path=dataset_path,
        bar_construction_ticks_per_bar=2000,
    )
    manifest_path = artifacts_dir / f"artifact_manifest_{symbol}.json"
    save_manifest(manifest, manifest_path)
    save_manifest(manifest, artifacts_dir / "artifact_manifest.json")
    return manifest_path


class ResearchSchemaTests(unittest.TestCase):
    def test_invalid_override_is_rejected(self) -> None:
        tmpdir = make_test_dir("research_invalid_override")
        try:
            proposal_path = make_proposal(
                tmpdir / "proposal.json",
                payload={"overrides": {"TRAIN_NOT_ALLOWED": 1}},
            )
            with self.assertRaises(research_schema.ProposalValidationError):
                research_schema.load_proposal(proposal_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_safe_env_is_pinned(self) -> None:
        tmpdir = make_test_dir("research_env_pins")
        try:
            proposal_path = make_proposal(tmpdir / "proposal.json")
            proposal = research_schema.load_proposal(proposal_path)
            env = research_schema.build_research_env_overrides(proposal, tmpdir / "artifacts")

            self.assertEqual("runtime", env["train"]["TRAIN_ENV_MODE"])
            self.assertEqual("EURUSD", env["train"]["TRAIN_SYMBOL"])
            self.assertEqual("120000", env["train"]["TRAIN_TOTAL_TIMESTEPS"])
            self.assertEqual("1", env["train"]["TRAIN_EXPORT_BEST_FOLD"])
            self.assertEqual("0", env["train"]["TRAIN_DEBUG_ALLOW_BASELINE_BYPASS"])
            self.assertEqual("1", env["eval"]["EVAL_SKIP_PLOT"])
            self.assertIn("TRAIN_WINDOW_SIZE", env["train"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_explicit_baseline_prefers_result_id_match(self) -> None:
        tmpdir = make_test_dir("research_baseline_lookup")
        try:
            proposal_path = make_proposal(
                tmpdir / "proposal.json",
                payload={"baseline_reference": "run-002", "fast_mode": False, "timesteps": 300000},
            )
            proposal = research_schema.load_proposal(proposal_path)
            rows = [
                {
                    "result_id": "run-001",
                    "experiment_name": "eurusd_reward_strip_window8",
                    "symbol": "EURUSD",
                    "fast_mode": False,
                    "status": "completed",
                    "composite_score": 1.2,
                    "timed_sharpe": 0.4,
                    "profit_factor": 1.1,
                    "expectancy_usd": 0.4,
                    "trade_count": 20,
                    "net_pnl_usd": 12.0,
                    "max_drawdown": 0.2,
                    "dataset_id": "dataset-1",
                    "bar_construction_ticks_per_bar": 2000,
                    "completed_at_utc": "2026-04-01T00:00:00+00:00",
                },
                {
                    "result_id": "run-002",
                    "experiment_name": "eurusd_baseline_alt",
                    "symbol": "EURUSD",
                    "fast_mode": False,
                    "status": "completed",
                    "composite_score": 2.1,
                    "timed_sharpe": 0.8,
                    "profit_factor": 1.3,
                    "expectancy_usd": 0.7,
                    "trade_count": 24,
                    "net_pnl_usd": 18.0,
                    "max_drawdown": 0.12,
                    "dataset_id": "dataset-1",
                    "bar_construction_ticks_per_bar": 2000,
                    "completed_at_utc": "2026-04-02T00:00:00+00:00",
                },
            ]

            baseline = research_schema.resolve_research_baseline(
                proposal,
                rows,
                dataset_id="dataset-1",
                bar_construction_ticks_per_bar=2000,
            )

            assert baseline is not None
            self.assertEqual("run-002", baseline["reference"])
            self.assertEqual("research_result", baseline["source"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class ResearchRunnerTests(unittest.TestCase):
    def _fake_subprocess(self, repo_root: Path):
        def _runner(command, cwd, env, stdout, stderr, text, check):
            script = Path(command[-1]).name.lower()
            if stdout is not None:
                stdout.write(f"running {script}\n")
            if script == "train_agent.py":
                artifacts_dir = Path(env["TRAIN_MODEL_DIR"])
                symbol = env["TRAIN_SYMBOL"].upper()
                slug = symbol.lower()
                baseline_report_path = artifacts_dir / f"baseline_diagnostics_{slug}.json"
                make_manifest_bundle(artifacts_dir, symbol=symbol)
                write_json(
                    baseline_report_path,
                    {
                        "symbol": symbol,
                        "gate_passed": True,
                        "passing_models": ["logistic_pair"],
                        "target_definition": {"type": "cost_adjusted_tradability"},
                        "holdout_metrics": {
                            "models": {
                                "logistic_pair": {
                                    "metrics": {
                                        "profit_factor": 1.10,
                                        "expectancy_usd": 0.50,
                                        "trade_count": 20.0,
                                        "net_pnl_usd": 10.0,
                                    }
                                }
                            }
                        },
                    },
                )
                write_json(
                    artifacts_dir / f"training_diagnostics_{slug}.json",
                    {
                        "baseline_gate_passed": True,
                        "deploy_ready": True,
                        "blockers": [],
                        "baseline_report_path": str(baseline_report_path),
                        "full_path_validation_metrics": {"timed_sharpe": 0.7, "profit_factor": 1.2},
                        "holdout_metrics": {
                            "timed_sharpe": 0.8,
                            "profit_factor": 1.4,
                            "expectancy_usd": 1.4,
                            "trade_count": 28,
                            "max_drawdown": 0.11,
                        },
                        "overtrade_negative_edge_triggered": False,
                        "explained_variance": 0.42,
                        "approx_kl": 0.02,
                        "value_loss_mean": 0.9,
                    },
                )
            elif script == "evaluate_oos.py":
                artifacts_dir = Path(env["EVAL_OUTPUT_DIR"])
                symbol = env["EVAL_SYMBOL"].upper()
                slug = symbol.lower()
                write_json(
                    artifacts_dir / f"replay_report_{slug}.json",
                    {
                        "replay_metrics": {
                            "timed_sharpe": 1.15,
                            "profit_factor": 1.45,
                            "expectancy_usd": 1.6,
                            "trade_count": 28,
                            "net_pnl_usd": 22.0,
                            "gross_pnl_usd": 28.0,
                            "total_transaction_cost_usd": 7.0,
                            "max_drawdown": 0.10,
                            "metric_reconciliation": {"passed": True},
                            "runtime_parity_verdict": {
                                "research_vs_runtime_parity_aligned": True,
                                "fragile_under_cost_stress": False,
                                "best_runtime_baseline": "runtime_mean_reversion",
                            },
                        },
                        "runtime_parity_verdict": {
                            "research_vs_runtime_parity_aligned": True,
                            "fragile_under_cost_stress": False,
                            "best_runtime_baseline": "runtime_mean_reversion",
                        },
                        "decision_summary": {"verdict": "no_immediate_reject_flag"},
                    },
                )
                write_json(
                    artifacts_dir / f"deployment_gate_{slug}.json",
                    {
                        "approved_for_live": False,
                        "blockers": ["Ops attestation missing (expected in research mode)."],
                    },
                )
            return subprocess.CompletedProcess(command, 0)

        return _runner

    def test_validate_only_does_not_launch_subprocesses(self) -> None:
        tmpdir = make_test_dir("research_validate_only")
        try:
            proposal_path = make_proposal(tmpdir / "proposal.json")
            with patch("tools.research_runner.subprocess.run") as run_mock:
                result = research_runner.run_research_proposal(
                    proposal_path,
                    validate_only=True,
                    repo_root=tmpdir,
                )
            run_mock.assert_not_called()
            self.assertEqual("EURUSD", result["resolved_proposal"]["symbol"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_active_training_context_is_rejected(self) -> None:
        tmpdir = make_test_dir("research_active_context")
        try:
            proposal_path = make_proposal(tmpdir / "proposal.json")
            current_run_path = tmpdir / "checkpoints" / "current_training_run.json"
            write_json(
                current_run_path,
                {
                    "state": "training",
                    "symbol": "EURUSD",
                    "checkpoints_root": str(tmpdir / "checkpoints" / "run_live"),
                },
            )
            with self.assertRaises(research_schema.ProposalValidationError):
                research_runner.run_research_proposal(
                    proposal_path,
                    repo_root=tmpdir,
                    current_run_path=current_run_path,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dead_pid_active_training_context_does_not_block_runner(self) -> None:
        tmpdir = make_test_dir("research_dead_pid_context")
        try:
            proposal_path = make_proposal(tmpdir / "proposal.json")
            current_run_path = tmpdir / "checkpoints" / "current_training_run.json"
            write_json(
                current_run_path,
                {
                    "state": "training",
                    "symbol": "EURUSD",
                    "pid": 42424242,
                    "updated_at_utc": "2026-04-06T17:00:01+00:00",
                    "heartbeat_path": str(tmpdir / "checkpoints" / "run_live" / "fold_0" / "training_heartbeat.json"),
                    "checkpoints_root": str(tmpdir / "checkpoints" / "run_live"),
                },
            )
            with patch("research.schema._pid_is_running", return_value=False), patch(
                "tools.research_runner.subprocess.run",
                side_effect=self._fake_subprocess(tmpdir),
            ):
                result = research_runner.run_research_proposal(
                    proposal_path,
                    repo_root=tmpdir,
                    current_run_path=current_run_path,
                )

            self.assertEqual("completed", result["run_status"])
            self.assertEqual("promote_candidate", result["decision"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_runner_writes_result_and_appends_ledger_without_touching_models(self) -> None:
        tmpdir = make_test_dir("research_runner_e2e")
        try:
            proposal_path = make_proposal(
                tmpdir / "research" / "proposals" / "proposal.json",
                payload={"fast_mode": False, "timesteps": 300000},
            )
            canonical_models = tmpdir / "models"
            canonical_models.mkdir(parents=True, exist_ok=True)
            sentinel_model = canonical_models / "model_eurusd_best.zip"
            sentinel_model.write_bytes(b"canonical-model")

            with patch("tools.research_runner.subprocess.run", side_effect=self._fake_subprocess(tmpdir)):
                result = research_runner.run_research_proposal(
                    proposal_path,
                    repo_root=tmpdir,
                    current_run_path=tmpdir / "checkpoints" / "current_training_run.json",
                )

            self.assertEqual("completed", result["run_status"])
            self.assertIn("research/results", result["result_json_path"].replace("\\", "/"))
            self.assertEqual("promote_candidate", result["decision"])
            self.assertEqual("baseline_gate", result["baseline_resolution"]["source"])
            self.assertTrue(Path(result["result_json_path"]).exists())
            self.assertTrue((Path(result["artifact_pointers"]["artifacts_dir"]) / "replay_report_eurusd.json").exists())
            self.assertEqual(b"canonical-model", sentinel_model.read_bytes())
            self.assertIn("baseline_distance_summary", result)
            self.assertAlmostEqual(12.0, result["baseline_distance_summary"]["net_pnl_delta_vs_baseline"])
            self.assertAlmostEqual(1.4, result["baseline_distance_summary"]["trade_count_vs_baseline_ratio"])
            self.assertAlmostEqual(1.1, result["baseline_distance_summary"]["expectancy_delta_vs_baseline"])
            self.assertAlmostEqual(0.25, result["baseline_distance_summary"]["cost_share"])
            self.assertFalse(result["baseline_distance_summary"]["overtrade_negative_edge_triggered"])
            self.assertAlmostEqual(12.0, result["training_diagnostics_summary"]["net_pnl_delta_vs_baseline"])
            self.assertAlmostEqual(0.25, result["training_diagnostics_summary"]["cost_share"])

            ledger_path = tmpdir / "research" / "ledger" / "experiments.jsonl"
            ledger_rows = research_schema.read_jsonl_rows(ledger_path)
            self.assertEqual(1, len(ledger_rows))
            self.assertEqual(result["result_id"], ledger_rows[0]["result_id"])
            self.assertEqual(result["decision"], ledger_rows[0]["decision"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
