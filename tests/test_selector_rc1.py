from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.dummy import DummyRegressor

from rule_selector import RuleSelector
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
from tools.verify_v1_rc import validate_manifest_truth_requirements


class SelectorRc1Tests(unittest.TestCase):
    def test_create_selector_manifest_round_trip_and_load_validated_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset.csv"
            dataset_path.write_text("x\n1\n", encoding="utf-8")

            model = DummyRegressor(strategy="mean")
            model.fit([[0.0], [1.0]], [0.0, 1.0])
            model_path = root / "selector.joblib"
            joblib.dump(model, model_path)

            manifest = create_selector_manifest(
                strategy_symbol="EURUSD",
                model_path=model_path,
                model_version="1.2.3",
                feature_schema=["spread_z"],
                dataset_path=dataset_path,
                ticks_per_bar=10000,
                holdout_start_utc="2026-01-01T00:00:00+00:00",
                label_definition=LabelDefinition(
                    path="A",
                    target_column="signed_target",
                    horizon_bars=10,
                    is_classification=False,
                ),
                cost_model=CostModel(commission_per_lot=7.0, slippage_pips=0.25),
                threshold_policy=ThresholdPolicy(min_edge_pips=1.0, reject_ambiguous=True),
                runtime_constraints=RuntimeConstraints(
                    session_filter_active=True,
                    spread_sanity_max_pips=2.0,
                    max_concurrent_positions=1,
                    daily_loss_stop_usd=50.0,
                ),
            )

            manifest_path = root / "selector_manifest.json"
            save_selector_manifest(manifest, manifest_path)
            loaded = load_selector_manifest(manifest_path, verify_manifest_hash=True)
            validated_model = load_validated_selector_model(loaded, expected_symbol="EURUSD")

            self.assertEqual(loaded.engine_type, "ML")
            self.assertFalse(loaded.live_trading_approved)
            self.assertTrue(loaded.manifest_hash)
            self.assertEqual(validated_model.__class__.__name__, "DummyRegressor")

    def test_rule_selector_and_shadow_broker_emit_open_then_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset.csv"
            dataset_path.write_text("x\n1\n", encoding="utf-8")

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
                evaluator_hash="eval",
                logic_hash="logic",
            )
            manifest_path = root / "manifest.json"
            save_selector_manifest(manifest, manifest_path)

            selector = RuleSelector(manifest_path)
            gate_status = selector.gate_status(
                signal=1,
                current_spread_pips=0.5,
                is_session_open=True,
                portfolio_state={"current_positions": 0, "daily_pnl_usd": 0.0},
            )
            self.assertTrue(gate_status["allow_execution"])

            audit_path = root / "shadow_audit.jsonl"
            broker = ShadowBroker(manifest_path, audit_path=audit_path)
            open_record = broker.evaluate(
                bar_ts="2026-04-08T09:00:00+00:00",
                features={"spread_z": -2.0},
                current_spread_pips=0.5,
                is_session_open=True,
            )
            close_record = broker.evaluate(
                bar_ts="2026-04-08T09:05:00+00:00",
                features={"spread_z": 0.0},
                current_spread_pips=0.4,
                is_session_open=True,
            )

            rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(open_record.would_open)
            self.assertFalse(open_record.would_close)
            self.assertTrue(close_record.would_close)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["signal"], 1)
            self.assertEqual(rows[1]["signal"], 0)

    def test_validate_manifest_truth_requirements_detects_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset.csv"
            dataset_path.write_text("x\n1\n", encoding="utf-8")

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
                evaluator_hash="stale-evaluator-hash",
                logic_hash="stale-logic-hash",
            )
            manifest_path = root / "manifest.json"
            save_selector_manifest(manifest, manifest_path)

            with self.assertRaises(RuntimeError):
                validate_manifest_truth_requirements(manifest_path)


if __name__ == "__main__":
    unittest.main()
