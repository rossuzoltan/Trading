from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from domain.models import VolumeBar
from runtime.shadow_broker import ShadowBroker
from selector_manifest import (
    CostModel,
    RuntimeConstraints,
    ThresholdPolicy,
    create_rule_manifest,
    save_selector_manifest,
)


class ShadowBrokerTests(unittest.TestCase):
    def _build_manifest(self, tmp_path: Path) -> Path:
        dataset_path = tmp_path / "DATA_CLEAN_VOLUME_5000.csv"
        dataset_path.write_text("stub dataset", encoding="utf-8")
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
            release_stage="paper_live_candidate",
            evaluator_hash="eval-hash",
            logic_hash="logic-hash",
        )
        manifest_path = tmp_path / "manifest.json"
        save_selector_manifest(manifest, manifest_path)
        return manifest_path

    def test_shadow_broker_logs_open_and_reversal_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_path = self._build_manifest(tmp_path)
            audit_path = tmp_path / "shadow_audit.jsonl"
            broker = ShadowBroker(manifest_path, audit_path=audit_path)

            open_record = broker.evaluate(
                bar_ts="2026-04-08T10:00:00+00:00",
                features={
                    "price_z": -1.6,
                    "spread_z": 0.1,
                    "ma20_slope": 0.05,
                    "ma50_slope": 0.02,
                    "Close": 1.1005,
                    "avg_spread": 0.0001,
                    "time_delta_s": 300.0,
                },
                current_spread_pips=0.7,
                is_session_open=True,
            )
            reverse_record = broker.evaluate(
                bar_ts="2026-04-08T10:05:00+00:00",
                features={
                    "price_z": 1.8,
                    "spread_z": 0.1,
                    "ma20_slope": -0.04,
                    "ma50_slope": -0.02,
                    "Close": 1.1010,
                    "avg_spread": 0.0001,
                    "time_delta_s": 300.0,
                },
                current_spread_pips=0.8,
                is_session_open=True,
            )

            self.assertTrue(open_record.would_open)
            self.assertFalse(open_record.would_close)
            self.assertEqual(1, open_record.position_after)

            self.assertTrue(reverse_record.would_open)
            self.assertTrue(reverse_record.would_close)
            self.assertEqual(-1, reverse_record.position_after)

            audit_rows = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(2, len(audit_rows))
            self.assertEqual("authorized", audit_rows[0]["reason"])
            self.assertEqual("reverse", audit_rows[1]["action_state"])
            self.assertEqual(-1, audit_rows[1]["position_after"])
            self.assertEqual(1, audit_rows[1]["exit_snapshot"]["transition_sequence"])
            self.assertEqual(2, audit_rows[1]["entry_snapshot"]["transition_sequence"])
            self.assertEqual("sell", audit_rows[1]["exit_snapshot"]["fill_side"])
            self.assertEqual("sell", audit_rows[1]["entry_snapshot"]["fill_side"])
            self.assertIsNotNone(audit_rows[1]["exit_snapshot"]["fill_price"])
            self.assertIsNotNone(audit_rows[1]["entry_snapshot"]["fill_price"])
            self.assertTrue((audit_path.parent / "run_meta.json").exists())

    def test_shadow_broker_logs_session_block_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_path = self._build_manifest(tmp_path)
            broker = ShadowBroker(manifest_path, audit_path=tmp_path / "shadow_audit.jsonl")

            record = broker.evaluate(
                bar_ts="2026-04-08T22:30:00+00:00",
                features={"price_z": -2.0, "spread_z": 0.0, "ma20_slope": 0.0, "ma50_slope": 0.0},
                current_spread_pips=0.6,
                is_session_open=False,
            )

            self.assertFalse(record.session_ok)
            self.assertFalse(record.allow_execution)
            self.assertFalse(record.would_open)
            self.assertFalse(record.would_close)
            self.assertEqual("session blocked", record.reason)
            self.assertEqual("runtime_gate", record.block_stage)
            self.assertEqual("session", record.runtime_gate_block_reason)

    def test_shadow_broker_writes_run_meta_and_rich_event_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_path = self._build_manifest(tmp_path)
            audit_path = tmp_path / "shadow_audit.jsonl"
            broker = ShadowBroker(manifest_path, audit_path=audit_path, profile_name="test_profile")

            bar = VolumeBar(
                timestamp=datetime(2026, 4, 8, 10, 0, tzinfo=timezone.utc),
                open=1.1000,
                high=1.1010,
                low=1.0990,
                close=1.1005,
                volume=5000.0,
                avg_spread=0.00008,
                time_delta_s=320.0,
                start_time_msc=int(datetime(2026, 4, 8, 10, 0, tzinfo=timezone.utc).timestamp() * 1000),
                end_time_msc=int(datetime(2026, 4, 8, 10, 5, tzinfo=timezone.utc).timestamp() * 1000),
            )
            record = broker.evaluate(
                bar_ts=bar.timestamp,
                bar=bar,
                features={
                    "price_z": 0.3,
                    "price_roll_std": 0.00042,
                    "spread_z": 0.1,
                    "time_delta_z": 0.0,
                    "ma20": 1.1001,
                    "ma50": 1.0998,
                    "ma20_slope": 0.01,
                    "ma50_slope": 0.01,
                },
                current_spread_pips=0.8,
                is_session_open=True,
            )

            meta_payload = json.loads((audit_path.parent / "run_meta.json").read_text(encoding="utf-8"))
            event_payload = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual("rule", record.block_stage)
            self.assertEqual("price_z_threshold", record.rule_block_reason)
            self.assertEqual(1.1005, event_payload["bar_close"])
            self.assertEqual(0.00042, event_payload["price_roll_std"])
            self.assertEqual("test_profile", event_payload["profile_name"])
            self.assertEqual("test_profile", meta_payload["profile_name"])
            self.assertTrue(meta_payload["run_id"])
            self.assertEqual(meta_payload["run_id"], event_payload["run_id"])
            self.assertIn("python_executable", meta_payload["process"])
            self.assertEqual(1.0, meta_payload["resolved_execution_cost_profile"]["partial_fill_ratio"])
            self.assertTrue(meta_payload["resolved_execution_cost_profile_hash"])
            self.assertEqual(meta_payload["resolved_execution_cost_profile_hash"], event_payload["execution_cost_profile_hash"])


if __name__ == "__main__":
    unittest.main()
