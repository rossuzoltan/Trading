from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import verify_v1_rc
from runtime.shadow_broker import ShadowBroker
from selector_manifest import (
    CostModel,
    LabelDefinition,
    RuntimeConstraints,
    ThresholdPolicy,
    create_rule_manifest,
    create_selector_manifest,
    load_selector_manifest,
    load_validated_selector_model,
    save_selector_manifest,
)


class Rc1ShadowTests(unittest.TestCase):
    def _write_dataset_stub(self, root: Path) -> Path:
        dataset_path = root / "dataset.csv"
        dataset_path.write_text("stub-dataset\n", encoding="utf-8")
        return dataset_path

    def _rule_manifest(self, root: Path):
        dataset_path = self._write_dataset_stub(root)
        return create_rule_manifest(
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
            replay_parity_reference="baseline_scoreboard_rc1.json",
        )

    def test_rule_manifest_round_trip_preserves_v4_truth_fields(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._rule_manifest(root)
            manifest_path = root / "manifest.json"
            save_selector_manifest(manifest, manifest_path)
            loaded = load_selector_manifest(manifest_path, verify_manifest_hash=True)
            self.assertEqual("4", loaded.manifest_version)
            self.assertEqual("paper_live_candidate", loaded.release_stage)
            self.assertFalse(loaded.live_trading_approved)
            self.assertEqual("eval-hash", loaded.evaluator_hash)
            self.assertEqual("logic-hash", loaded.logic_hash)
            self.assertEqual("baseline_scoreboard_rc1.json", loaded.replay_parity_reference)
            self.assertTrue(loaded.manifest_hash)

    def test_load_validated_selector_model_rejects_checksum_drift(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = self._write_dataset_stub(root)
            model_path = root / "selector.joblib"
            model = DecisionTreeRegressor(max_depth=1, random_state=7)
            model.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 1.0])
            joblib.dump(model, model_path)
            manifest = create_selector_manifest(
                strategy_symbol="EURUSD",
                model_path=model_path,
                model_version="1.0.0",
                feature_schema=["feature_a"],
                dataset_path=dataset_path,
                ticks_per_bar=10000,
                label_definition=LabelDefinition(
                    path="A",
                    target_column="signed_target",
                    horizon_bars=10,
                    is_classification=False,
                ),
                cost_model=CostModel(commission_per_lot=7.0, slippage_pips=0.25),
                threshold_policy=ThresholdPolicy(min_edge_pips=0.5, reject_ambiguous=True),
                runtime_constraints=RuntimeConstraints(
                    session_filter_active=True,
                    spread_sanity_max_pips=2.0,
                    max_concurrent_positions=1,
                    daily_loss_stop_usd=50.0,
                ),
            )
            loaded_model = load_validated_selector_model(manifest, expected_symbol="EURUSD")
            self.assertTrue(hasattr(loaded_model, "predict"))
            model_path.write_text("checksum-drift", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_validated_selector_model(manifest, expected_symbol="EURUSD")

    def test_verify_rc1_certification_flags_hash_drift(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._rule_manifest(root)
            drifted_manifest = manifest.__class__(
                **{
                    **manifest.__dict__,
                    "evaluator_hash": "old-eval",
                    "logic_hash": "old-logic",
                }
            )
            manifest_path = root / "manifest.json"
            save_selector_manifest(drifted_manifest, manifest_path)
            with self.assertRaises(RuntimeError) as ctx:
                verify_v1_rc.validate_manifest_truth_requirements(manifest_path)
            message = str(ctx.exception)
            self.assertIn("Truth-engine drift detected", message)

    def test_shadow_broker_audits_open_then_close_without_execution(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._rule_manifest(root)
            manifest_path = root / "manifest.json"
            audit_path = root / "shadow_audit.jsonl"
            save_selector_manifest(manifest, manifest_path)

            broker = ShadowBroker(manifest_path, audit_path=audit_path)
            open_record = broker.evaluate(
                bar_ts=datetime(2026, 4, 8, 9, 0, tzinfo=timezone.utc),
                features={"spread_z": -1.5},
                current_spread_pips=0.5,
                is_session_open=True,
            )
            close_record = broker.evaluate(
                bar_ts=datetime(2026, 4, 8, 9, 5, tzinfo=timezone.utc),
                features={"spread_z": 0.0},
                current_spread_pips=0.4,
                is_session_open=True,
            )

            rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(2, len(rows))
            self.assertTrue(open_record.would_open)
            self.assertFalse(open_record.would_close)
            self.assertTrue(open_record.session_ok)
            self.assertTrue(open_record.risk_ok)
            self.assertFalse(close_record.would_open)
            self.assertTrue(close_record.would_close)
            self.assertEqual(1, close_record.current_position_direction)
            self.assertEqual(True, rows[0]["would_open"])
            self.assertEqual(True, rows[1]["would_close"])


if __name__ == "__main__":
    unittest.main()
