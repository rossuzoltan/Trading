from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor

from edge_research import BaselineAlphaGate, save_baseline_alpha_gate
from rule_selector import RuleSelector
from runtime.shadow_broker import ShadowBroker
from selector_manifest import (
    CostModel,
    LabelDefinition,
    RuntimeConstraints,
    ThresholdPolicy,
    _file_sha256,
    create_rule_manifest,
    create_selector_manifest,
    load_selector_manifest,
    load_validated_selector_model,
    save_selector_manifest,
)
from tools.verify_v1_rc import validate_manifest_truth_requirements


class ConstantProbaModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, features):
        n_rows = int(len(features))
        return np.tile(np.asarray([[1.0 - self.probability, self.probability]], dtype=np.float64), (n_rows, 1))


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
                current_hour_utc=10,
            )
            self.assertTrue(gate_status["allow_execution"])

            audit_path = root / "shadow_audit.jsonl"
            broker = ShadowBroker(manifest_path, audit_path=audit_path)
            open_record = broker.evaluate(
                bar_ts="2026-04-08T09:00:00+00:00",
                features={"price_z": -2.0, "spread_z": 0.0, "ma20_slope": 0.0, "ma50_slope": 0.0},
                current_spread_pips=0.5,
                is_session_open=True,
            )
            close_record = broker.evaluate(
                bar_ts="2026-04-08T09:05:00+00:00",
                features={"price_z": 0.0, "spread_z": 0.0, "ma20_slope": 0.0, "ma50_slope": 0.0},
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

    def test_load_selector_manifest_strict_mode_requires_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset.csv"
            dataset_path.write_text("x\n1\n", encoding="utf-8")
            manifest = create_rule_manifest(
                strategy_symbol="EURUSD",
                rule_family="mean_reversion",
                rule_params={"threshold": 1.0},
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
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["manifest_hash"] = ""
            manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_selector_manifest(manifest_path, strict_manifest_hash=True)

    def test_rule_selector_applies_alpha_gate_veto(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset.csv"
            dataset_path.write_text("x\n1\n", encoding="utf-8")

            gate = BaselineAlphaGate(
                symbol="EURUSD",
                feature_cols=("price_z", "spread_z", "ma20_slope", "ma50_slope"),
                model_kind="logistic_pair",
                probability_threshold=0.55,
                probability_margin=0.05,
                long_model=ConstantProbaModel(0.40),
                short_model=ConstantProbaModel(0.20),
                fit_quality_passed=True,
            )
            gate_path = root / "alpha_gate.joblib"
            save_baseline_alpha_gate(gate, gate_path)

            manifest = create_rule_manifest(
                strategy_symbol="EURUSD",
                rule_family="mean_reversion",
                rule_params={"threshold": 1.0},
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
                alpha_gate={
                    "enabled": True,
                    "model_path": str(gate_path),
                    "model_sha256": _file_sha256(gate_path),
                    "probability_threshold": 0.55,
                    "probability_margin": 0.05,
                },
                release_stage="paper_live_candidate",
                evaluator_hash="eval",
                logic_hash="logic",
            )
            manifest_path = root / "manifest.json"
            save_selector_manifest(manifest, manifest_path)
            selector = RuleSelector(manifest_path)
            decision = selector.decide(
                features={"price_z": -2.0, "spread_z": 0.0, "ma20_slope": 0.0, "ma50_slope": 0.0},
                current_spread_pips=0.5,
                is_session_open=True,
                portfolio_state={"current_positions": 0, "current_direction": 0, "daily_pnl_usd": 0.0},
                current_hour_utc=10,
            )
            self.assertEqual(0, decision.signal)
            self.assertIn("alpha gate veto", decision.reason)


if __name__ == "__main__":
    unittest.main()
