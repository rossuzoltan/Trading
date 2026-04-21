from __future__ import annotations

import unittest

from risk.sizing import compute_lot_size


class RiskSizingTests(unittest.TestCase):
    def test_compute_lot_size_rounds_down_and_clamps(self) -> None:
        # $10k equity, 1% risk => $100 risk budget.
        # Stop distance 20 pips, $10/pip per lot => $200 risk per 1.0 lot.
        # Raw lots = 0.5 -> step 0.01 -> 0.50
        res = compute_lot_size(
            equity=10_000.0,
            risk_fraction=0.01,
            stop_distance_pips=20.0,
            pip_value_per_lot=10.0,
            lot_min=0.01,
            lot_max=10.0,
            lot_step=0.01,
        )
        self.assertAlmostEqual(0.50, res.lots, places=6)
        self.assertIn(res.reason, {"sizing_ok", "sizing_rounded_down"})

        # Clamp to max
        res2 = compute_lot_size(
            equity=1_000_000.0,
            risk_fraction=0.02,
            stop_distance_pips=10.0,
            pip_value_per_lot=10.0,
            lot_min=0.01,
            lot_max=0.10,
            lot_step=0.01,
        )
        self.assertAlmostEqual(0.10, res2.lots, places=6)
        self.assertEqual("sizing_clamped_to_max", res2.reason)

    def test_compute_lot_size_invalid_inputs_fail_closed(self) -> None:
        res = compute_lot_size(
            equity=0.0,
            risk_fraction=0.01,
            stop_distance_pips=20.0,
            pip_value_per_lot=10.0,
            lot_min=0.01,
            lot_max=1.0,
            lot_step=0.01,
        )
        self.assertEqual(0.0, res.lots)
        self.assertEqual("sizing_invalid_equity", res.reason)


if __name__ == "__main__":
    unittest.main()

