from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from train_agent import evaluate_model


def _make_trade(*, gross_pnl: float, cost: float, holding_bars: int = 5, forced: bool = False, equity: float = 1000.0) -> dict[str, Any]:
    net_pnl = gross_pnl - cost
    return {
        "gross_pnl_usd": float(gross_pnl),
        "net_pnl_usd": float(net_pnl),
        "transaction_cost_usd": float(cost),
        "commission_usd": float(cost * 0.5),
        "spread_slippage_cost_usd": float(cost * 0.5),
        "spread_cost_usd": float(cost * 0.25),
        "slippage_cost_usd": float(cost * 0.25),
        "holding_bars": int(holding_bars),
        "forced_close": bool(forced),
        "equity": float(equity),
        "net_pips": float(gross_pnl * 10.0),
        "gross_pips": float(gross_pnl * 10.0),
    }


class _DummyModel:
    def predict(self, obs, action_masks=None, deterministic=True):
        return np.array([0], dtype=np.int64), None


class _EpisodeAuditEnv:
    def __init__(self, trade_log: list[dict[str, Any]], execution_log: list[dict[str, Any]], diagnostics: dict[str, Any]):
        self._trade_log = trade_log
        self._execution_log = execution_log
        self._diagnostics = diagnostics
        self._stepped = False

    def reset(self):
        self._stepped = False
        return np.zeros((1, 1), dtype=np.float32)

    def env_method(self, name: str, *args, **kwargs):
        if name == "action_masks":
            return [np.array([True], dtype=bool)]
        if name == "get_training_diagnostics":
            return [{"total_steps": 0, "action_counts": {}, "trade_stats": {}, "economics": {}, "reward_components": {}}]
        if name == "get_trade_log":
            return [[]]
        if name == "get_execution_log":
            return [[]]
        raise AssertionError(f"Unexpected env_method: {name}")

    def step(self, action):
        if self._stepped:
            raise AssertionError("Environment should terminate in one step for this test.")
        self._stepped = True
        info = {
            "equity": float(self._trade_log[-1]["equity"]) if self._trade_log else 1000.0,
            "timestamp_utc": "2024-01-01T00:00:00+00:00",
            "closed_trades_delta": [dict(item) for item in self._trade_log],
            "executed_events_delta": [dict(item) for item in self._execution_log],
            "episode_diagnostics": self._diagnostics,
        }
        return (
            np.zeros((1, 1), dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([True], dtype=bool),
            [info],
        )


class EvaluateModelEpisodeAuditTests(unittest.TestCase):
    def test_episode_audit_is_used_before_post_reset_env_queries(self) -> None:
        trade_log = [
            _make_trade(gross_pnl=0.50, cost=0.14, equity=1000.36),
            _make_trade(gross_pnl=-0.20, cost=0.14, equity=1000.02),
            _make_trade(gross_pnl=0.50, cost=0.14, forced=True, equity=1000.38),
        ]
        execution_log = [
            {"event": "order_executed", "side": "open"},
            {"event": "order_executed", "side": "close"},
            {"event": "order_executed", "side": "open"},
            {"event": "order_executed", "side": "close"},
            {"event": "order_executed", "side": "open"},
            {"event": "order_executed", "side": "close", "forced": True},
        ]
        diagnostics = {
            "total_steps": 12,
            "action_counts": {"hold": 6, "close": 2, "long": 2, "short": 2},
            "trade_stats": {
                "action_selected_count": 12,
                "action_accepted_count": 6,
                "accepted_open_count": 3,
                "accepted_close_count": 3,
                "order_executed_count": 6,
                "executed_open_count": 3,
                "executed_close_count": 3,
                "entered_long_count": 2,
                "entered_short_count": 1,
                "entry_signal_long_count": 2,
                "entry_signal_short_count": 1,
                "closed_trade_count": 3,
                "trade_attempt_count": 6,
                "trade_reject_count": 0,
                "forced_close_count": 1,
                "flat_steps": 6,
                "long_steps": 4,
                "short_steps": 2,
                "position_duration_sum": 15.0,
                "position_duration_count": 3,
                "rapid_reversals": 0,
                "position_durations_sample": [5, 5, 5],
            },
            "economics": {
                "gross_pnl_usd": 0.8,
                "net_pnl_usd": 0.38,
                "transaction_cost_usd": 0.42,
                "commission_usd": 0.21,
                "spread_slippage_cost_usd": 0.21,
                "spread_cost_usd": 0.105,
                "slippage_cost_usd": 0.105,
            },
            "reward_components": {
                "pnl_reward_sum": 0.0,
                "slippage_penalty_sum": 0.0,
                "participation_bonus_sum": 0.0,
                "holding_penalty_sum": 0.0,
                "drawdown_penalty_sum": 0.0,
                "net_reward_sum": 0.0,
            },
        }
        env = _EpisodeAuditEnv(trade_log=trade_log, execution_log=execution_log, diagnostics=diagnostics)

        _, metrics = evaluate_model(_DummyModel(), env)

        self.assertEqual(3.0, float(metrics["trade_count"]))
        self.assertEqual(6, int(metrics["executed_order_count"]))
        self.assertFalse(bool(metrics["accounting_gap_detected"]))
        self.assertTrue(bool(metrics["metrics_reconciliation"]["passed"]))
        self.assertEqual(3, int(metrics["execution_diagnostics"]["trade_diagnostics"]["closed_trade_count"]))
        self.assertAlmostEqual(0.38, float(metrics["net_pnl_usd"]), places=6)


if __name__ == "__main__":
    unittest.main()
