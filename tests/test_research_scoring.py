from __future__ import annotations

import unittest

from research import scoring


class ResearchScoringTests(unittest.TestCase):
    def test_strong_holdout_metrics_score_above_promotion_floor(self) -> None:
        summary = scoring.compute_composite_score(
            replay_metrics={
                "timed_sharpe": 1.4,
                "profit_factor": 1.6,
                "expectancy_usd": 2.4,
                "trade_count": 36,
                "max_drawdown": 0.08,
                "metric_reconciliation": {"passed": True},
            },
            training_diagnostics={"deploy_ready": True},
            runtime_parity_verdict={
                "research_vs_runtime_parity_aligned": True,
                "fragile_under_cost_stress": False,
            },
            baseline_gate_passed=True,
        )

        self.assertGreaterEqual(summary["score"], scoring.PROMOTE_SCORE_MIN)
        self.assertEqual([], summary["critical_failures"])

    def test_low_trade_count_gets_penalized(self) -> None:
        strong = scoring.compute_composite_score(
            replay_metrics={
                "timed_sharpe": 0.9,
                "profit_factor": 1.3,
                "expectancy_usd": 1.0,
                "trade_count": 30,
                "max_drawdown": 0.12,
                "metric_reconciliation": {"passed": True},
            },
            training_diagnostics=None,
            runtime_parity_verdict={"research_vs_runtime_parity_aligned": True},
            baseline_gate_passed=True,
        )
        weak = scoring.compute_composite_score(
            replay_metrics={
                "timed_sharpe": 0.9,
                "profit_factor": 1.3,
                "expectancy_usd": 1.0,
                "trade_count": 4,
                "max_drawdown": 0.12,
                "metric_reconciliation": {"passed": True},
            },
            training_diagnostics=None,
            runtime_parity_verdict={"research_vs_runtime_parity_aligned": True},
            baseline_gate_passed=True,
        )

        self.assertLess(weak["score"], strong["score"])
        self.assertLess(weak["penalties"]["low_trade_penalty"], 0.0)

    def test_reconciliation_failure_is_critical(self) -> None:
        summary = scoring.compute_composite_score(
            replay_metrics={
                "timed_sharpe": 1.0,
                "profit_factor": 1.5,
                "expectancy_usd": 1.8,
                "trade_count": 24,
                "max_drawdown": 0.10,
                "metric_reconciliation": {"passed": False},
            },
            training_diagnostics=None,
            runtime_parity_verdict={"research_vs_runtime_parity_aligned": True},
            baseline_gate_passed=True,
        )

        self.assertIn("Replay accounting reconciliation failed.", summary["critical_failures"])
        self.assertLess(summary["score"], 0.0)

    def test_runtime_fragility_penalizes_without_being_critical_by_itself(self) -> None:
        stable = scoring.compute_composite_score(
            replay_metrics={
                "timed_sharpe": 0.8,
                "profit_factor": 1.2,
                "expectancy_usd": 0.8,
                "trade_count": 26,
                "max_drawdown": 0.11,
                "metric_reconciliation": {"passed": True},
            },
            training_diagnostics=None,
            runtime_parity_verdict={
                "research_vs_runtime_parity_aligned": True,
                "fragile_under_cost_stress": False,
            },
            baseline_gate_passed=True,
        )
        fragile = scoring.compute_composite_score(
            replay_metrics={
                "timed_sharpe": 0.8,
                "profit_factor": 1.2,
                "expectancy_usd": 0.8,
                "trade_count": 26,
                "max_drawdown": 0.11,
                "metric_reconciliation": {"passed": True},
            },
            training_diagnostics=None,
            runtime_parity_verdict={
                "research_vs_runtime_parity_aligned": True,
                "fragile_under_cost_stress": True,
            },
            baseline_gate_passed=True,
        )

        self.assertEqual([], fragile["critical_failures"])
        self.assertLess(fragile["score"], stable["score"])

    def test_baseline_delta_controls_promote_vs_keep(self) -> None:
        score_summary = {"score": 3.0, "critical_failures": []}

        promote = scoring.build_research_decision(
            run_status="completed",
            score_summary=score_summary,
            baseline_comparison={
                "materially_better": True,
                "reason": "Composite score delta 0.700 meets the promotion threshold.",
            },
        )
        keep = scoring.build_research_decision(
            run_status="completed",
            score_summary=score_summary,
            baseline_comparison={
                "materially_better": False,
                "reason": "Composite score delta 0.100 does not meet the promotion threshold.",
            },
        )

        self.assertEqual("promote_candidate", promote["decision"])
        self.assertEqual("keep", keep["decision"])


if __name__ == "__main__":
    unittest.main()
