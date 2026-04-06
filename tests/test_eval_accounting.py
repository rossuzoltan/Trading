"""test_eval_accounting.py
=========================
Phase 1 regression tests for the evaluation accounting fix.

Spec: a trade stream with non-zero trades MUST NOT serialize as zero
through any reporting path. Mismatches between trade_log and
execution_diagnostics MUST be detected and flagged, not silently ignored.

Tests:
  - compute_trade_metrics with known trade stream → all fields non-zero
  - _extract_eval_trade_log strategy1 (env_method) → correct count
  - _extract_eval_trade_log strategy2 (get_attr _runtime) → correct count
  - _extract_eval_trade_log all-fail → returns [] with warning (not exception)
  - evaluate_model with trade-producing env → trade_count == len(trade_log)
  - evaluate_model: accounting_gap_detected == False when trade_log matches diag
  - evaluate_model: accounting_gap_detected == True when trade_log empty but
    diagnostics claim closed trades (regression for the original bug)
  - build_trade_metric_reconciliation: mismatch → passed == False
  - build_trade_metric_reconciliation: match → passed == True
"""
from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from runtime_common import build_trade_metric_reconciliation, compute_trade_metrics
from train_agent import _extract_eval_trade_log, evaluate_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(*, gross_pnl: float, cost: float, holding_bars: int = 5, forced: bool = False) -> dict[str, Any]:
    """Helper to build a well-formed trade dict matching broker.trade_log schema."""
    net_pnl = gross_pnl - cost
    return {
        "reason": "MANUAL",
        "direction": 1,
        "volume": 0.01,
        "entry_price": 1.1000,
        "exit_price": 1.1010,
        "gross_pnl_usd": float(gross_pnl),
        "net_pnl_usd": float(net_pnl),
        "transaction_cost_usd": float(cost),
        "commission_usd": float(cost * 0.5),
        "spread_slippage_cost_usd": float(cost * 0.5),
        "spread_cost_usd": float(cost * 0.25),
        "slippage_cost_usd": float(cost * 0.25),
        "holding_bars": int(holding_bars),
        "forced_close": bool(forced),
        "equity": 1000.0 + float(net_pnl),
        "net_pips": float(gross_pnl * 10),
        "gross_pips": float(gross_pnl * 10),
    }


def _make_trade_log(n_wins: int = 5, n_losses: int = 5) -> list[dict[str, Any]]:
    """Build a synthetic trade log with known economics."""
    trades = []
    equity = 1000.0
    for _ in range(n_wins):
        trade = _make_trade(gross_pnl=0.50, cost=0.14)
        trade["equity"] = equity + trade["net_pnl_usd"]
        equity = trade["equity"]
        trades.append(trade)
    for _ in range(n_losses):
        trade = _make_trade(gross_pnl=-0.30, cost=0.14)
        trade["equity"] = equity + trade["net_pnl_usd"]
        equity = trade["equity"]
        trades.append(trade)
    return trades


# ---------------------------------------------------------------------------
# ENV stubs
# ---------------------------------------------------------------------------

class _EnvMethodEnv:
    """Stub env where env_method("get_trade_log") returns a real trade log."""

    def __init__(self, trade_log: list[dict]) -> None:
        self._trade_log = list(trade_log)

    def reset(self):
        return np.zeros((1, 1), dtype=np.float32)

    def env_method(self, name: str, *args, **kwargs):
        if name == "get_trade_log":
            return [list(self._trade_log)]
        if name == "get_execution_log":
            return [[]]
        if name == "action_masks":
            return [np.array([True], dtype=bool)]
        if name == "get_training_diagnostics":
            return [{"total_steps": 0, "action_counts": {}, "trade_stats": {}, "economics": {}, "reward_components": {}}]
        raise AssertionError(f"Unexpected env_method: {name}")

    def step(self, action):
        return (
            np.zeros((1, 1), dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([True], dtype=bool),
            [{"equity": 1000.0, "timestamp_utc": "2024-01-01T00:00:00+00:00"}],
        )

    def get_attr(self, name: str):
        raise RuntimeError(f"No get_attr for {name}")


class _GetAttrRuntimeEnv:
    """Stub env where only Strategy 2 (get_attr _runtime) works."""

    def __init__(self, trade_log: list[dict]) -> None:
        broker = MagicMock()
        broker.trade_log = list(trade_log)
        broker.execution_log = []
        runtime = MagicMock()
        runtime.broker = broker
        self._runtime = runtime

    def reset(self):
        return np.zeros((1, 1), dtype=np.float32)

    def env_method(self, name: str, *args, **kwargs):
        if name == "get_trade_log":
            # Strategy 1 fails
            raise RuntimeError("env_method not supported")
        if name == "get_execution_log":
            raise RuntimeError("env_method not supported")
        if name == "action_masks":
            return [np.array([True], dtype=bool)]
        if name == "get_training_diagnostics":
            return [{"total_steps": 0, "action_counts": {}, "trade_stats": {}, "economics": {}, "reward_components": {}}]
        raise AssertionError(f"Unexpected env_method: {name}")

    def get_attr(self, name: str):
        if name == "_runtime":
            return [self._runtime]
        raise AttributeError(name)

    def step(self, action):
        return (
            np.zeros((1, 1), dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([True], dtype=bool),
            [{"equity": 1000.0, "timestamp_utc": "2024-01-01T00:00:00+00:00"}],
        )


class _AllFailEnv:
    """Stub env where all trade-log extraction strategies fail."""

    def reset(self):
        return np.zeros((1, 1), dtype=np.float32)

    def env_method(self, name: str, *args, **kwargs):
        if name == "get_trade_log":
            raise RuntimeError("not supported")
        if name == "get_execution_log":
            raise RuntimeError("not supported")
        if name == "action_masks":
            return [np.array([True], dtype=bool)]
        if name == "get_training_diagnostics":
            return [{"total_steps": 0, "action_counts": {}, "trade_stats": {}, "economics": {}, "reward_components": {}}]
        raise AssertionError(f"Unexpected env_method: {name}")

    def get_attr(self, name: str):
        raise RuntimeError("get_attr not supported")

    def step(self, action):
        return (
            np.zeros((1, 1), dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([True], dtype=bool),
            [{"equity": 1000.0, "timestamp_utc": "2024-01-01T00:00:00+00:00"}],
        )


class _DummyModel:
    def predict(self, obs, action_masks=None, deterministic=True):
        return np.array([0], dtype=np.int64), None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeTradeMetrics(unittest.TestCase):
    """Regression: non-zero trade stream cannot produce all-zero metrics."""

    def test_non_zero_trades_produce_non_zero_metrics(self):
        """REGRESSION: 10 trades must not serialize as trade_count=0 etc."""
        trade_log = _make_trade_log(n_wins=5, n_losses=5)
        metrics = compute_trade_metrics(trade_log, initial_equity=1_000.0)

        self.assertEqual(10, int(metrics["trade_count"]), "trade_count must be 10")
        self.assertNotEqual(0.0, metrics["gross_pnl_usd"], "gross_pnl_usd must be non-zero")
        self.assertNotEqual(0.0, metrics["total_transaction_cost_usd"], "total_transaction_cost_usd must be non-zero")
        self.assertGreater(metrics["win_rate"], 0.0, "win_rate must be > 0")

    def test_empty_trade_log_returns_zero_sentinel(self):
        """Empty log is legitimately zero — must not raise."""
        metrics = compute_trade_metrics([], initial_equity=1_000.0)
        self.assertEqual(0.0, metrics["trade_count"])
        self.assertEqual(0.0, metrics["profit_factor"])

    def test_all_winning_trades_infinite_profit_factor(self):
        trade_log = [_make_trade(gross_pnl=1.0, cost=0.10)] * 5
        metrics = compute_trade_metrics(trade_log, initial_equity=1_000.0)
        self.assertTrue(
            metrics["profit_factor"] == float("inf") or metrics["profit_factor"] > 1.0,
            "all-win => profit_factor should be inf or very high",
        )

    def test_forced_close_counted(self):
        trades = [
            _make_trade(gross_pnl=0.5, cost=0.1, forced=True),
            _make_trade(gross_pnl=-0.3, cost=0.1, forced=False),
        ]
        metrics = compute_trade_metrics(trades, initial_equity=1_000.0)
        self.assertEqual(1, int(metrics["forced_close_count"]))

    def test_avg_holding_bars(self):
        trades = [
            _make_trade(gross_pnl=0.2, cost=0.1, holding_bars=10),
            _make_trade(gross_pnl=0.2, cost=0.1, holding_bars=20),
        ]
        metrics = compute_trade_metrics(trades, initial_equity=1_000.0)
        self.assertAlmostEqual(15.0, metrics["avg_holding_bars"], places=1)


class TestExtractEvalTradeLog(unittest.TestCase):
    """Tests for the 3-strategy fallback extraction chain."""

    def test_strategy1_env_method_returns_correct_trades(self):
        trade_log = _make_trade_log(n_wins=3, n_losses=2)
        env = _EnvMethodEnv(trade_log)
        result = _extract_eval_trade_log(env)
        self.assertEqual(5, len(result))

    def test_strategy2_get_attr_runtime_returns_correct_trades(self):
        trade_log = _make_trade_log(n_wins=2, n_losses=3)
        env = _GetAttrRuntimeEnv(trade_log)
        result = _extract_eval_trade_log(env)
        self.assertEqual(5, len(result))

    def test_all_strategies_fail_returns_empty_and_warns(self):
        """REGRESSION: if all strategies fail, return [] — do NOT raise, DO warn."""
        import logging
        env = _AllFailEnv()
        with self.assertLogs("train_agent", level=logging.WARNING):
            result = _extract_eval_trade_log(env)
        self.assertEqual([], result, "should return empty list, not raise")

    def test_strategy1_takes_priority_over_strategy2(self):
        """Strategy 1 returns data; strategy 2 should not be reached."""
        trade_log_s1 = _make_trade_log(n_wins=4, n_losses=0)

        class PriorityEnv(_EnvMethodEnv):
            def get_attr(self, name: str):
                raise AssertionError("Strategy 2 should not be reached if Strategy 1 works")

        env = PriorityEnv(trade_log_s1)
        result = _extract_eval_trade_log(env)
        self.assertEqual(4, len(result))


class TestEvaluateModelAccounting(unittest.TestCase):
    """Tests for evaluate_model() accounting correctness."""

    def test_trade_count_matches_trade_log_length(self):
        """CORE REGRESSION: trade_count in metrics == len(trade_log)."""
        trade_log = _make_trade_log(n_wins=6, n_losses=4)
        env = _EnvMethodEnv(trade_log)
        model = _DummyModel()
        _, metrics = evaluate_model(model, env)
        self.assertEqual(10, metrics["trade_count"], "trade_count must match trade_log length")

    def test_accounting_gap_not_detected_when_log_matches(self):
        """When trade_log is non-empty, accounting_gap_detected must be False."""
        trade_log = _make_trade_log(n_wins=3, n_losses=2)
        env = _EnvMethodEnv(trade_log)
        model = _DummyModel()
        _, metrics = evaluate_model(model, env)
        self.assertFalse(metrics["accounting_gap_detected"])

    def test_metric_reconciliation_in_output(self):
        """evaluate_model must always include metric_reconciliation in output."""
        env = _EnvMethodEnv(_make_trade_log(n_wins=2, n_losses=1))
        model = _DummyModel()
        _, metrics = evaluate_model(model, env)
        self.assertIn("metric_reconciliation", metrics)
        self.assertIn("passed", metrics["metric_reconciliation"])

    def test_accounting_gap_detected_when_trade_log_empty_but_diag_non_zero(self):
        """REGRESSION: if trade_log extraction fails silently but diagnostics
        show closed trades, accounting_gap_detected must be True."""
        import logging

        class EmptyTradeLogDiagnosticEnv(_AllFailEnv):
            """All strategies fail (empty trade_log) but diagnostics show 50 trades."""

            def env_method(self, name: str, *args, **kwargs):
                if name == "get_trade_log":
                    raise RuntimeError("not supported")
                if name == "get_execution_log":
                    raise RuntimeError("not supported")
                if name == "action_masks":
                    return [np.array([True], dtype=bool)]
                if name == "get_training_diagnostics":
                    return [{
                        "total_steps": 500,
                        "action_counts": {"hold": 400, "close": 50, "long": 25, "short": 25},
                        "trade_stats": {
                            "action_selected_count": 500,
                            "closed_trade_count": 50,
                            "action_accepted_count": 100,
                            "accepted_open_count": 50,
                            "accepted_close_count": 50,
                            "order_executed_count": 100,
                            "executed_open_count": 50,
                            "executed_close_count": 50,
                            "entered_long_count": 25,
                            "entered_short_count": 25,
                            "entry_signal_long_count": 25,
                            "entry_signal_short_count": 25,
                            "trade_attempt_count": 100,
                            "trade_reject_count": 0,
                            "forced_close_count": 5,
                            "flat_steps": 400,
                            "long_steps": 60,
                            "short_steps": 40,
                            "position_duration_sum": 250.0,
                            "position_duration_count": 50,
                            "rapid_reversals": 2,
                            "position_durations_sample": [5] * 50,
                        },
                        "economics": {
                            "gross_pnl_usd": 12.5,
                            "net_pnl_usd": -5.2,
                            "transaction_cost_usd": 17.7,
                            "commission_usd": 7.0,
                            "spread_slippage_cost_usd": 10.7,
                            "spread_cost_usd": 8.0,
                            "slippage_cost_usd": 2.7,
                        },
                        "reward_components": {},
                    }]
                raise AssertionError(f"Unexpected env_method: {name}")

        env = EmptyTradeLogDiagnosticEnv()
        model = _DummyModel()
        with self.assertLogs("train_agent", level=logging.WARNING) as cm:
            _, metrics = evaluate_model(model, env)

        self.assertTrue(
            metrics["accounting_gap_detected"],
            "accounting_gap_detected must be True when trade_log is empty but diag shows trades",
        )
        self.assertEqual(0, metrics["trade_count"], "trade_count should be 0 (from empty log)")
        self.assertTrue(
            any("accounting gap" in msg.lower() for msg in cm.output),
            "A warning about the accounting gap must be logged",
        )

    def test_economic_metrics_non_zero_with_real_trades(self):
        """Economics fields must all be non-zero when trades exist."""
        trade_log = _make_trade_log(n_wins=5, n_losses=5)
        env = _EnvMethodEnv(trade_log)
        model = _DummyModel()
        _, metrics = evaluate_model(model, env)
        self.assertNotEqual(0.0, metrics["gross_pnl_usd"])
        self.assertNotEqual(0.0, metrics["total_transaction_cost_usd"])
        self.assertGreater(metrics["win_rate"], 0.0)
        self.assertGreater(metrics["profit_factor"], 0.0)


class TestTradeMetricReconciliation(unittest.TestCase):
    """Tests for build_trade_metric_reconciliation()."""

    def test_count_match_passes(self):
        trade_log = _make_trade_log(n_wins=3, n_losses=2)
        metrics = compute_trade_metrics(trade_log, initial_equity=1_000.0)
        result = build_trade_metric_reconciliation(
            trade_metrics=metrics,
            trade_diagnostics={"closed_trade_count": 5},
            trade_log_count=5,
        )
        self.assertTrue(result["passed"])
        self.assertEqual([], result["mismatch_fields"])

    def test_count_mismatch_fails(self):
        """REGRESSION: if reported trade_count != actual trade_log count → mismatch."""
        trade_log = _make_trade_log(n_wins=3, n_losses=2)
        metrics = compute_trade_metrics(trade_log, initial_equity=1_000.0)
        result = build_trade_metric_reconciliation(
            trade_metrics=metrics,
            trade_diagnostics={"closed_trade_count": 99},  # deliberate mismatch
            trade_log_count=5,
        )
        self.assertFalse(result["passed"])
        self.assertIn("trade_count_vs_diagnostics", result["mismatch_fields"])

    def test_trade_log_count_mismatch_fails(self):
        trade_log = _make_trade_log(n_wins=2, n_losses=2)
        metrics = compute_trade_metrics(trade_log, initial_equity=1_000.0)
        result = build_trade_metric_reconciliation(
            trade_metrics=metrics,
            trade_log_count=99,  # wrong number
        )
        self.assertFalse(result["passed"])
        self.assertIn("trade_count_vs_trade_log", result["mismatch_fields"])

    def test_no_diagnostics_no_crash(self):
        """Missing diagnostics → no crash, may pass or have no checks."""
        metrics = compute_trade_metrics([], initial_equity=1_000.0)
        result = build_trade_metric_reconciliation(trade_metrics=metrics)
        self.assertIn("passed", result)
        self.assertIn("mismatch_fields", result)


if __name__ == "__main__":
    unittest.main()
