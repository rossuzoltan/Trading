from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import compare_oos_baselines as baseline_tool
import evaluate_oos
from feature_engine import FEATURE_COLS, _compute_raw
from runtime_common import build_action_map
from train_agent import _prepend_runtime_warmup_context


def _make_history(rows: int = 320) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(1.1000, 1.1200, rows)
    wave = np.sin(np.arange(rows) / 9.0) * 0.0005
    close = base + wave
    open_ = close - 0.0002
    high = close + 0.0004
    low = close - 0.0004
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(rows, 100.0),
            "avg_spread": np.full(rows, 0.0001),
            "time_delta_s": np.full(rows, 3600.0),
        },
        index=index,
    )


class CompareOosBaselinesTests(unittest.TestCase):
    def _build_context(self) -> evaluate_oos.ReplayContext:
        history = _make_history()
        featured = _compute_raw(history).dropna(subset=list(FEATURE_COLS))
        trainable = featured.iloc[:-80].copy()
        holdout = featured.iloc[-80:].copy()
        warmup_source = _prepend_runtime_warmup_context(featured, holdout)
        warmup_count = max(len(warmup_source) - len(holdout), 0)
        scaler = StandardScaler().fit(trainable.loc[:, FEATURE_COLS])
        return evaluate_oos.ReplayContext(
            symbol="EURUSD",
            source="unit_test",
            dataset_path=Path("tests/tmp/unit_dataset.csv"),
            action_map=build_action_map([1.0], [1.0]),
            model=SimpleNamespace(),
            obs_normalizer=None,
            scaler=scaler,
            execution_cost_profile={
                "commission_per_lot": 7.0,
                "slippage_pips": 0.25,
                "partial_fill_ratio": 1.0,
            },
            reward_profile={
                "reward_scale": 10_000.0,
                "drawdown_penalty": 2.0,
                "transaction_penalty": 1.0,
                "reward_clip_low": -5.0,
                "reward_clip_high": 5.0,
            },
            warmup_frame=warmup_source.iloc[:warmup_count].copy(),
            replay_frame=warmup_source.iloc[warmup_count:].copy(),
            replay_feature_frame=holdout.copy(),
            full_feature_frame=featured.copy(),
            trainable_feature_frame=trainable.copy(),
            holdout_feature_frame=holdout.copy(),
            holdout_start_utc=pd.Timestamp(holdout.index[0]).isoformat(),
            diagnostics_path=None,
            manifest_path=None,
            artifact_metadata={},
        )

    def test_flat_baseline_zero_trade_accounting(self) -> None:
        context = self._build_context()

        payload = baseline_tool._evaluate_policy(
            replay_context=context,
            action_index_provider=baseline_tool._flat_provider,
        )
        metrics = payload["metrics"]

        self.assertEqual(0, int(metrics["trade_count"]))
        self.assertEqual(0, int(metrics["executed_order_count"]))
        self.assertAlmostEqual(0.0, float(metrics["gross_pnl_usd"]))
        self.assertAlmostEqual(0.0, float(metrics["net_pnl_usd"]))
        self.assertAlmostEqual(0.0, float(metrics["total_transaction_cost_usd"]))

    def test_segment_context_preserves_replay_start(self) -> None:
        context = self._build_context()
        segment = context.holdout_feature_frame.iloc[-20:].copy()

        segment_context = baseline_tool._clone_context_with_segment(context, segment)

        self.assertEqual(segment.index[0], segment_context.replay_feature_frame.index[0])
        self.assertEqual(segment.index[0], segment_context.replay_frame.index[0])
        self.assertGreater(len(segment_context.warmup_frame), 0)

    def test_cost_stress_scales_slippage_without_changing_commission(self) -> None:
        context = self._build_context()

        stressed = baseline_tool._with_cost_stress(context, slippage_multiplier=2.0)

        self.assertEqual(
            float(context.execution_cost_profile["commission_per_lot"]),
            float(stressed.execution_cost_profile["commission_per_lot"]),
        )
        self.assertAlmostEqual(
            float(context.execution_cost_profile["slippage_pips"]) * 2.0,
            float(stressed.execution_cost_profile["slippage_pips"]),
        )


if __name__ == "__main__":
    unittest.main()
