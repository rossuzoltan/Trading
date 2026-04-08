from __future__ import annotations

import unittest

from strategies.rule_logic import compute_mean_reversion_direction


class RuleLogicTests(unittest.TestCase):
    def test_mean_reversion_uses_price_extension_not_spread_direction(self) -> None:
        signal = compute_mean_reversion_direction(
            {
                "price_z": -1.6,
                "spread_z": 0.1,
                "time_delta_z": 0.0,
                "ma20_slope": 0.04,
                "ma50_slope": 0.02,
            },
            {"threshold": 1.5},
        )
        self.assertEqual(1, signal)

    def test_mean_reversion_blocks_elevated_spread_regime(self) -> None:
        signal = compute_mean_reversion_direction(
            {
                "price_z": -2.0,
                "spread_z": 1.2,
                "time_delta_z": 0.0,
                "ma20_slope": 0.03,
                "ma50_slope": 0.02,
            },
            {"threshold": 1.5, "max_spread_z": 0.5},
        )
        self.assertEqual(0, signal)

    def test_mean_reversion_blocks_strong_trend_regime(self) -> None:
        signal = compute_mean_reversion_direction(
            {
                "price_z": 2.0,
                "spread_z": 0.1,
                "time_delta_z": 0.0,
                "ma20_slope": 0.25,
                "ma50_slope": 0.12,
            },
            {
                "threshold": 1.5,
                "max_abs_ma20_slope": 0.15,
                "max_abs_ma50_slope": 0.08,
            },
        )
        self.assertEqual(0, signal)


if __name__ == "__main__":
    unittest.main()
