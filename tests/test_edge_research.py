from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd

import edge_research
from edge_research import fit_baseline_alpha_gate, run_edge_baseline_research
from feature_engine import FEATURE_COLS


TEST_TMP_ROOT = Path("tests/tmp")


def make_test_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_supervised_frame(*, rows: int, predictive: bool) -> pd.DataFrame:
    rng = np.random.default_rng(42 if predictive else 123)
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    latent = np.sin(np.arange(rows) / 11.0) + 0.35 * np.cos(np.arange(rows) / 19.0)
    if predictive:
        returns = 0.00035 * latent + rng.normal(0.0, 0.00004, rows)
        feature_driver = latent
    else:
        returns = rng.normal(0.0, 0.00035, rows)
        feature_driver = rng.normal(0.0, 1.0, rows)

    close = 1.1000 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "Close": close,
            "log_return": np.concatenate([[0.0], returns[:-1]]),
            "body_size": feature_driver + rng.normal(0.0, 0.02, rows),
            "candle_range": np.abs(feature_driver) + 0.1,
            "ma20_slope": pd.Series(feature_driver).rolling(20, min_periods=1).mean().to_numpy(),
            "ma50_slope": pd.Series(feature_driver).rolling(50, min_periods=1).mean().to_numpy(),
            "vol_norm_atr": np.abs(returns) * 100.0 + 0.01,
            "spread_z": rng.normal(0.0, 0.5, rows),
            "time_delta_z": rng.normal(0.0, 0.5, rows),
            "hour_sin": np.sin(2 * np.pi * index.hour / 24),
            "hour_cos": np.cos(2 * np.pi * index.hour / 24),
            "day_sin": np.sin(2 * np.pi * index.dayofweek / 5),
            "day_cos": np.cos(2 * np.pi * index.dayofweek / 5),
        },
        index=index,
    )


class EdgeResearchTests(unittest.TestCase):
    def test_allowed_directions_respects_relaxed_threshold_override(self):
        class DummyProbModel:
            def __init__(self, positive_prob: float) -> None:
                self.positive_prob = positive_prob

            def predict_proba(self, x):
                return np.tile(np.array([[1.0 - self.positive_prob, self.positive_prob]], dtype=np.float64), (len(x), 1))

        gate = edge_research.BaselineAlphaGate(
            symbol="EURUSD",
            feature_cols=tuple(FEATURE_COLS),
            model_kind="logistic_pair",
            probability_threshold=0.55,
            probability_margin=0.05,
            long_model=DummyProbModel(0.48),
            short_model=DummyProbModel(0.46),
        )
        row = {feature: 0.0 for feature in FEATURE_COLS}

        allow_long, allow_short, _ = gate.allowed_directions(row)
        self.assertFalse(allow_long)
        self.assertFalse(allow_short)

        relaxed_long, relaxed_short, _ = gate.allowed_directions(
            row,
            threshold_override=0.45,
            margin_override=0.0,
        )
        self.assertTrue(relaxed_long)
        self.assertFalse(relaxed_short)

    def test_fit_baseline_alpha_gate_prefers_logistic_pair_when_available(self):
        frame = make_supervised_frame(rows=900, predictive=True)
        gate = fit_baseline_alpha_gate(
            symbol="EURUSD",
            train_frame=frame.iloc[:700].copy(),
            feature_cols=FEATURE_COLS,
            horizon_bars=10,
            commission_per_lot=0.0,
            slippage_pips=0.0,
            min_edge_pips=0.0,
            probability_threshold=0.55,
            probability_margin=0.05,
            model_preference="auto",
        )

        self.assertIsNotNone(gate)
        assert gate is not None
        self.assertEqual("logistic_pair", gate.model_kind)
        allow_long, allow_short, scores = gate.allowed_directions(frame.iloc[650])
        self.assertIn("long_score", scores)
        self.assertIn("short_score", scores)
        self.assertIsInstance(allow_long, bool)
        self.assertIsInstance(allow_short, bool)

    def test_fit_baseline_alpha_gate_uses_selected_logistic_threshold(self):
        frame = make_supervised_frame(rows=900, predictive=True)

        class DummyProbModel:
            def predict_proba(self, x):
                return np.tile(np.array([[0.2, 0.8]], dtype=np.float64), (len(x), 1))

        with patch.object(edge_research, "_fit_probability_pair", return_value=(DummyProbModel(), DummyProbModel())), patch.object(
            edge_research,
            "_choose_probability_threshold",
            return_value={
                "threshold": 0.65,
                "metrics": {
                    "trade_count": 48.0,
                    "long_trade_count": 24.0,
                    "short_trade_count": 24.0,
                    "expectancy_usd": 3.5,
                    "profit_factor": 1.3,
                },
            },
        ):
            gate = fit_baseline_alpha_gate(
                symbol="EURUSD",
                train_frame=frame.iloc[:700].copy(),
                feature_cols=FEATURE_COLS,
                horizon_bars=10,
                commission_per_lot=0.0,
                slippage_pips=0.0,
                min_edge_pips=0.0,
                probability_threshold=0.55,
                probability_margin=0.05,
                model_preference="auto",
            )

        self.assertIsNotNone(gate)
        assert gate is not None
        self.assertEqual("logistic_pair", gate.model_kind)
        self.assertAlmostEqual(0.65, gate.probability_threshold)
        self.assertTrue(gate.fit_quality_passed)
        self.assertEqual(24.0, gate.fit_long_trade_count)
        self.assertEqual(24.0, gate.fit_short_trade_count)

    def test_fit_baseline_alpha_gate_auto_falls_back_from_sparse_one_sided_logistic(self):
        frame = make_supervised_frame(rows=900, predictive=True)

        class DummyProbModel:
            def predict_proba(self, x):
                return np.tile(np.array([[0.1, 0.9]], dtype=np.float64), (len(x), 1))

        with patch.object(edge_research, "_fit_probability_pair", return_value=(DummyProbModel(), DummyProbModel())), patch.object(
            edge_research,
            "_choose_probability_threshold",
            return_value={
                "threshold": 0.65,
                "metrics": {
                    "trade_count": 3.0,
                    "long_trade_count": 0.0,
                    "short_trade_count": 3.0,
                    "expectancy_usd": 25.0,
                    "profit_factor": 2.0,
                },
            },
        ):
            gate = fit_baseline_alpha_gate(
                symbol="EURUSD",
                train_frame=frame.iloc[:700].copy(),
                feature_cols=FEATURE_COLS,
                horizon_bars=10,
                commission_per_lot=0.0,
                slippage_pips=0.0,
                min_edge_pips=0.0,
                probability_threshold=0.55,
                probability_margin=0.05,
                model_preference="auto",
            )

        self.assertIsNotNone(gate)
        assert gate is not None
        self.assertEqual("ridge_signed_target", gate.model_kind)
        self.assertTrue(gate.fit_quality_passed)

    def test_cost_adjusted_gate_passes_on_predictive_data(self):
        frame = make_supervised_frame(rows=900, predictive=True)
        trainable = frame.iloc[:700].copy()
        holdout = frame.iloc[700:].copy()
        folds = [
            (trainable.iloc[:450].copy(), trainable.iloc[450:550].copy()),
            (trainable.iloc[:550].copy(), trainable.iloc[550:650].copy()),
        ]
        tmpdir = make_test_dir("edge_gate_pass")
        out_path = tmpdir / "baseline.json"
        try:
            report = run_edge_baseline_research(
                symbol="EURUSD",
                trainable_frame=trainable,
                holdout_frame=holdout,
                folds=folds,
                feature_cols=FEATURE_COLS,
                out_path=out_path,
                horizon_bars=10,
                commission_per_lot=0.0,
                slippage_pips=0.0,
                min_edge_pips=0.0,
                probability_threshold=0.55,
                probability_margin=0.05,
                min_trade_count=20,
            )
            self.assertTrue(report["gate_passed"])
            self.assertTrue(report["passing_models"])
            self.assertIn("profit_factor", report["holdout_metrics"]["models"]["logistic_pair"]["metrics"])
            self.assertIn("sharpe_like", report["holdout_metrics"]["models"]["logistic_pair"]["metrics"])
            self.assertIn("tree_signed_target", report["holdout_metrics"]["models"])
            self.assertIn("trend_rule", report["holdout_metrics"]["models"])
            self.assertTrue(out_path.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_cost_adjusted_gate_fails_on_noise(self):
        frame = make_supervised_frame(rows=900, predictive=False)
        trainable = frame.iloc[:700].copy()
        holdout = frame.iloc[700:].copy()
        folds = [
            (trainable.iloc[:450].copy(), trainable.iloc[450:550].copy()),
            (trainable.iloc[:550].copy(), trainable.iloc[550:650].copy()),
        ]
        tmpdir = make_test_dir("edge_gate_fail")
        out_path = tmpdir / "baseline.json"
        try:
            report = run_edge_baseline_research(
                symbol="EURUSD",
                trainable_frame=trainable,
                holdout_frame=holdout,
                folds=folds,
                feature_cols=FEATURE_COLS,
                out_path=out_path,
                horizon_bars=10,
                commission_per_lot=7.0,
                slippage_pips=0.25,
                min_edge_pips=0.0,
                probability_threshold=0.55,
                probability_margin=0.05,
                min_trade_count=20,
            )
            self.assertFalse(report["gate_passed"])
            self.assertEqual([], report["passing_models"])
            self.assertIn("max_drawdown_usd", report["holdout_metrics"]["models"]["ridge_signed_target"]["metrics"])
            self.assertTrue(out_path.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
