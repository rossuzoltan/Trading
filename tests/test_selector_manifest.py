from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor

from feature_engine import FEATURE_COLS
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
    validate_paper_live_candidate_manifest,
)


class SelectorManifestTests(unittest.TestCase):
    def test_rule_manifest_round_trip_preserves_rc1_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "DATA_CLEAN_VOLUME_5000.csv"
            dataset_path.write_text("stub dataset", encoding="utf-8")
            manifest_path = tmp_path / "manifest.json"

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
            save_selector_manifest(manifest, manifest_path)

            loaded = load_selector_manifest(manifest_path)
            validate_paper_live_candidate_manifest(loaded)

            self.assertEqual("RULE", loaded.engine_type)
            self.assertFalse(loaded.live_trading_approved)
            self.assertTrue(bool(loaded.manifest_hash))
            self.assertEqual(5000, loaded.ticks_per_bar)
            self.assertEqual("mean_reversion", loaded.rule_family)

    def test_selector_model_loader_accepts_manifest_created_for_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "DATA_CLEAN_VOLUME_10000.csv"
            dataset_path.write_text("stub dataset", encoding="utf-8")
            model_path = tmp_path / "model.joblib"
            manifest_path = tmp_path / "selector_manifest.json"

            x = np.arange(len(FEATURE_COLS) * 4, dtype=float).reshape(4, len(FEATURE_COLS))
            y = np.arange(4, dtype=float)
            model = DummyRegressor(strategy="mean")
            model.fit(x, y)
            joblib.dump(model, model_path)

            manifest = create_selector_manifest(
                strategy_symbol="EURUSD",
                model_path=model_path,
                model_version="1.0.0",
                feature_schema=list(FEATURE_COLS),
                dataset_path=dataset_path,
                ticks_per_bar=10000,
                bar_construction_ticks_per_bar=10000,
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
            save_selector_manifest(manifest, manifest_path)

            loaded_manifest = load_selector_manifest(manifest_path)
            loaded_model = load_validated_selector_model(loaded_manifest, expected_symbol="EURUSD")

            self.assertTrue(hasattr(loaded_model, "predict"))
            self.assertEqual("ML", loaded_manifest.engine_type)
            self.assertEqual(model_path.as_posix(), Path(loaded_manifest.model_path).as_posix())


if __name__ == "__main__":
    unittest.main()
