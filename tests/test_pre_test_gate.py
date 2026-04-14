from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from selector_manifest import (
    AlphaGateSpec,
    CostModel,
    RuntimeConstraints,
    ThresholdPolicy,
    _file_sha256,
    create_rule_manifest,
    save_selector_manifest,
)
from tools.pre_test_gate import build_pre_test_gate


class PreTestGateTests(unittest.TestCase):
    def _write_bundle(
        self,
        root: Path,
        *,
        rc_candidate: dict[str, float | int],
        historical_payload: dict[str, object],
        with_alpha_gate: bool = False,
    ) -> Path:
        pack_dir = root / "models" / "rc1" / "eurusd_5k_v1_mr_rc1"
        pack_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = root / "data" / "DATA_CLEAN_VOLUME_5000.csv"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text("Gmt time,Symbol,Volume\n2024-01-01T00:00:00Z,EURUSD,5000\n", encoding="utf-8")

        alpha_gate_spec = AlphaGateSpec()
        if with_alpha_gate:
            alpha_path = pack_dir / "alpha_gate.joblib"
            alpha_path.write_text("stub", encoding="utf-8")
            alpha_gate_spec = AlphaGateSpec(
                enabled=True,
                model_path=str(alpha_path),
                model_sha256=_file_sha256(alpha_path),
                probability_threshold=0.55,
                probability_margin=0.05,
                min_edge_pips=0.0,
            )

        manifest = create_rule_manifest(
            strategy_symbol="EURUSD",
            rule_family="mean_reversion",
            rule_params={"threshold": 1.0, "sl_value": 1.5, "tp_value": 3.0},
            dataset_path=dataset_path,
            ticks_per_bar=5000,
            cost_model=CostModel(commission_per_lot=7.0, slippage_pips=0.25),
            threshold_policy=ThresholdPolicy(min_edge_pips=0.0, reject_ambiguous=True),
            runtime_constraints=RuntimeConstraints(
                session_filter_active=True,
                spread_sanity_max_pips=1.5,
                max_concurrent_positions=1,
                daily_loss_stop_usd=100.0,
            ),
            alpha_gate=alpha_gate_spec,
            release_stage="paper_live_candidate",
            evaluator_hash="eval-hash",
            logic_hash="logic-hash",
        )
        manifest_path = pack_dir / "manifest.json"
        save_selector_manifest(manifest, manifest_path)
        manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
        scoreboard = {
            "name": "eurusd_5k_v1_mr_rc1",
            "symbol": "EURUSD",
            "ticks_per_bar": 5000,
            "release_stage": "paper_live_candidate",
            "live_trading_approved": False,
            "manifest_hash": manifest_json["manifest_hash"],
            "evaluator_hash": manifest_json["evaluator_hash"],
            "logic_hash": manifest_json["logic_hash"],
            "rc_candidate": rc_candidate,
            "baselines": {
                "runtime_flat": {
                    "net_pnl_usd": 0.0,
                    "profit_factor": 0.0,
                    "expectancy_usd": 0.0,
                    "trade_count": 0,
                    "long_count": 0,
                    "short_count": 0,
                },
                "runtime_always_short": {
                    "net_pnl_usd": -10.0,
                    "profit_factor": 0.8,
                    "expectancy_usd": -1.0,
                    "trade_count": 5,
                    "long_count": 0,
                    "short_count": 5,
                },
                "runtime_trend": {
                    "net_pnl_usd": -12.0,
                    "profit_factor": 0.7,
                    "expectancy_usd": -1.2,
                    "trade_count": 6,
                    "long_count": 3,
                    "short_count": 3,
                },
            },
        }
        (pack_dir / "baseline_scoreboard_rc1.json").write_text(json.dumps(scoreboard), encoding="utf-8")
        historical_report = {
            "manifest_hash": manifest_json["manifest_hash"],
            "logic_hash": manifest_json["logic_hash"],
            "evaluator_hash": manifest_json["evaluator_hash"],
            **historical_payload,
        }
        (pack_dir / "mt5_historical_replay_report.json").write_text(json.dumps(historical_report), encoding="utf-8")
        return manifest_path

    def test_pre_test_gate_passes_balanced_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = self._write_bundle(
                root,
                rc_candidate={
                    "net_pnl_usd": 38.67,
                    "profit_factor": 1.17,
                    "expectancy_usd": 0.34,
                    "trade_count": 111,
                    "long_count": 60,
                    "short_count": 51,
                },
                historical_payload={
                    "overall_verdict": "WATCH",
                    "signal_density_ratio": 1.12,
                    "session_opens": {"Asia": 0, "Rollover": 0, "London": 4},
                    "live_trades_per_bar": 0.014,
                    "replay_trades_per_bar": 0.013,
                },
                with_alpha_gate=True,
            )
            payload = build_pre_test_gate(manifest_path=manifest_path)
            self.assertTrue(payload["ready_for_test"])
            self.assertEqual([], payload["blockers"])
            self.assertTrue((manifest_path.parent / "pre_test_gate.json").exists())

    def test_pre_test_gate_blocks_one_sided_and_drifted_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = self._write_bundle(
                root,
                rc_candidate={
                    "net_pnl_usd": -27.54,
                    "profit_factor": 0.79,
                    "expectancy_usd": -1.53,
                    "trade_count": 18,
                    "long_count": 0,
                    "short_count": 18,
                },
                historical_payload={
                    "overall_verdict": "DRIFT_CRITICAL",
                    "signal_density_ratio": 4.45,
                    "session_opens": {"Asia": 1, "Rollover": 1, "London": 4},
                    "live_trades_per_bar": 0.044,
                    "replay_trades_per_bar": 0.009,
                },
            )
            payload = build_pre_test_gate(manifest_path=manifest_path)
            self.assertFalse(payload["ready_for_test"])
            self.assertTrue(any(str(item).startswith("non_positive_net_pnl") for item in payload["blockers"]))
            self.assertTrue(any("historical_replay:asia_session_opens" in str(item) for item in payload["blockers"]))
            self.assertTrue(any("historical_replay:rollover_session_opens" in str(item) for item in payload["blockers"]))

    def test_pre_test_gate_blocks_stale_historical_report_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = self._write_bundle(
                root,
                rc_candidate={
                    "net_pnl_usd": 15.0,
                    "profit_factor": 1.2,
                    "expectancy_usd": 0.5,
                    "trade_count": 24,
                    "long_count": 12,
                    "short_count": 12,
                },
                historical_payload={
                    "manifest_hash": "stale-manifest",
                    "overall_verdict": "WATCH",
                    "signal_density_ratio": 1.02,
                    "session_opens": {"Asia": 0, "Rollover": 0},
                    "live_trades_per_bar": 0.012,
                    "replay_trades_per_bar": 0.011,
                },
            )
            payload = build_pre_test_gate(manifest_path=manifest_path)
            self.assertFalse(payload["ready_for_test"])
            self.assertTrue(any("historical_replay:manifest_hash_mismatch" in str(item) for item in payload["blockers"]))


if __name__ == "__main__":
    unittest.main()
