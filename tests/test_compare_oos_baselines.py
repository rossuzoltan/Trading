from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

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
            runtime_options={"window_size": 1, "alpha_gate_enabled": False},
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

    def test_mean_reversion_provider_handles_entry_and_exit(self) -> None:
        action_map = build_action_map([1.0], [1.0])

        open_short = baseline_tool._mean_reversion_provider(
            feature_row=pd.Series({"spread_z": 1.2}),
            position_direction=0,
            action_map=action_map,
        )
        hold_flat = baseline_tool._mean_reversion_provider(
            feature_row=pd.Series({"spread_z": 0.0}),
            position_direction=0,
            action_map=action_map,
        )
        close_short = baseline_tool._mean_reversion_provider(
            feature_row=pd.Series({"spread_z": 0.0}),
            position_direction=-1,
            action_map=action_map,
        )

        self.assertEqual(3, open_short)
        self.assertEqual(0, hold_flat)
        self.assertEqual(1, close_short)

    def test_runtime_baselines_include_flat_zero_trade_floor(self) -> None:
        context = self._build_context()

        results = baseline_tool._evaluate_runtime_baselines(replay_context=context)

        self.assertIn("runtime_flat", results)
        self.assertIn("runtime_always_long", results)
        self.assertIn("runtime_always_short", results)
        self.assertIn("runtime_mean_reversion", results)
        self.assertIn("runtime_trend", results)
        self.assertEqual(0, int(results["runtime_flat"]["metrics"]["trade_count"]))

    def test_runtime_parity_verdict_marks_fragility_only_after_positive_base(self) -> None:
        context = self._build_context()
        replay_metrics = {
            "trade_count": 10,
            "net_pnl_usd": 25.0,
            "validation_status": {"passed": True},
        }
        stress_side_effects = [
            {"metrics": {"trade_count": 12, "net_pnl_usd": 7.5, "validation_status": {"passed": True}}},
            {"metrics": {"trade_count": 10, "net_pnl_usd": 25.0, "validation_status": {"passed": True}}},
            {"metrics": {"trade_count": 10, "net_pnl_usd": 5.0, "validation_status": {"passed": True}}},
            {"metrics": {"trade_count": 10, "net_pnl_usd": -1.0, "validation_status": {"passed": True}}},
        ]
        with (
            patch.object(evaluate_oos, "_evaluate_runtime_baselines", return_value={"runtime_flat": {"metrics": {"trade_count": 0, "net_pnl_usd": 0.0, "validation_status": {"passed": True}}}}),
            patch.object(evaluate_oos, "_load_research_baseline_summary", return_value={"research_baseline_viable": True, "best_baseline": "mean_reversion"}),
            patch.object(evaluate_oos, "_evaluate_policy", side_effect=stress_side_effects),
        ):
            verdict = evaluate_oos._build_runtime_parity_verdict(
                context=context,
                replay_metrics=replay_metrics,
                training_diagnostics={"baseline_report_path": "unused.json"},
            )

        self.assertFalse(verdict["research_vs_runtime_parity_aligned"])
        self.assertTrue(verdict["fragile_under_cost_stress"])
        self.assertIn("1.5x", verdict["slippage_stress"])
        self.assertIn("2.0x", verdict["slippage_stress"])

    def test_decision_summary_flags_no_trade_and_concentration_risks(self) -> None:
        summary = evaluate_oos._build_decision_summary(
            reject_fast_diagnostics={
                "cost_share_of_gross_pnl": {"cost_share_of_abs_gross_pnl": 1.2},
                "expectancy_by_direction": {
                    "long": {"trade_count": 0.0, "expectancy_usd": 0.0},
                    "short": {"trade_count": 10.0, "expectancy_usd": 2.0},
                },
                "pnl_concentration": {"top_3_share_of_abs_net_pnl": 0.9},
            },
            runtime_parity_verdict={
                "best_runtime_baseline": "runtime_flat",
                "fragile_under_cost_stress": True,
            },
        )

        self.assertEqual("reject_trade_deployment", summary["verdict"])
        self.assertIn("overtrading_or_weak_entry_quality", summary["flags"])
        self.assertIn("direction_concentration", summary["flags"])
        self.assertIn("pnl_concentrated_in_few_trades", summary["flags"])
        self.assertIn("fragile_under_slippage_stress", summary["flags"])
        self.assertIn("no_trade_baseline_preferred", summary["flags"])

    def test_build_baseline_comparison_uses_replay_context_fallback(self) -> None:
        context = self._build_context()
        baseline_report = {
            "target_definition": {"type": "unit_test"},
            "holdout_metrics": {
                "models": {
                    "mean_reversion": {
                        "metrics": {
                            "trade_count": 12.0,
                            "net_pnl_usd": 45.0,
                            "profit_factor": 1.2,
                            "expectancy_usd": 3.75,
                        }
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "baseline_comparison.json"
            with (
                patch.object(baseline_tool, "load_replay_context", return_value=context),
                patch.object(baseline_tool, "run_edge_baseline_research", return_value=baseline_report),
                patch.object(
                    baseline_tool,
                    "_evaluate_policy",
                    return_value={"metrics": {"trade_count": 9.0, "net_pnl_usd": 12.5, "profit_factor": 1.3}},
                ),
                patch.object(
                    baseline_tool,
                    "load_json_report",
                    return_value={"replay_metrics": {"trade_count": 9.0, "net_pnl_usd": 12.5, "profit_factor": 1.3}},
                ),
                patch.object(
                    baseline_tool,
                    "_shared_build_runtime_parity_verdict",
                    return_value={
                        "best_runtime_baseline": "runtime_flat",
                        "best_runtime_baseline_metrics": {"trade_count": 0.0, "net_pnl_usd": 0.0},
                        "runtime_holdout_models": {"runtime_flat": {"metrics": {"trade_count": 0.0, "net_pnl_usd": 0.0}}},
                        "research_vs_runtime_parity_aligned": False,
                        "fragile_under_cost_stress": False,
                    },
                ),
            ):
                report = baseline_tool.build_baseline_comparison(symbol="EURUSD", report_path=report_path)
            self.assertTrue(report_path.exists())

        self.assertEqual("mean_reversion", report["best_baseline"])
        self.assertIn(
            report["best_runtime_baseline"],
            {
                "runtime_flat",
                "runtime_always_long",
                "runtime_always_short",
                "runtime_mean_reversion",
                "runtime_trend",
            },
        )
        self.assertIn("runtime_holdout_models", report)
        self.assertIn("runtime_parity_verdict", report)
        self.assertAlmostEqual(7.0, float(report["cost_profile"]["commission_per_lot"]))
        self.assertEqual(9.0, float(report["rl_replay_metrics"]["trade_count"]))


if __name__ == "__main__":
    unittest.main()
