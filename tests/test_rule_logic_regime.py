from __future__ import annotations

import unittest

from strategies.rule_logic import compute_mean_reversion_direction


class RuleLogicRegimeTests(unittest.TestCase):
    def test_mean_reversion_without_regime_filters_still_triggers(self) -> None:
        direction = compute_mean_reversion_direction(
            {
                "price_z": -1.6,
                "spread_z": 0.1,
                "time_delta_z": 0.0,
                "ma20_slope": 0.0,
                "ma50_slope": 0.0,
            },
            {"threshold": 1.5},
        )
        self.assertEqual(1, direction)

    def test_min_volatility_guard_blocks_low_vol_sideways(self) -> None:
        direction = compute_mean_reversion_direction(
            {
                "price_z": -1.6,
                "spread_z": 0.1,
                "time_delta_z": 0.0,
                "ma20_slope": 0.0,
                "ma50_slope": 0.0,
                "vol_norm_atr": 0.00001,
            },
            {"threshold": 1.5, "min_vol_norm_atr": 0.00005},
        )
        self.assertEqual(0, direction)

    def test_spike_guard_blocks_large_news_like_bar(self) -> None:
        direction = compute_mean_reversion_direction(
            {
                "price_z": -1.6,
                "spread_z": 0.1,
                "time_delta_z": 0.0,
                "ma20_slope": 0.0,
                "ma50_slope": 0.0,
                "log_return": 0.02,
            },
            {"threshold": 1.5, "max_abs_log_return": 0.005},
        )
        self.assertEqual(0, direction)


if __name__ == "__main__":
    unittest.main()
