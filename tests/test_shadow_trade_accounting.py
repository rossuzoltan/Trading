from __future__ import annotations

import unittest

from shadow_trade_accounting import summarize_shadow_trade_accounting


class ShadowTradeAccountingTests(unittest.TestCase):
    def test_snapshot_accounting_closes_before_opening_on_reversal(self) -> None:
        events = [
            {
                "event_index": 1,
                "timestamp_utc": "2026-04-08T10:00:00+00:00",
                "entry_snapshot": {
                    "event_index": 1,
                    "bar_ts_utc": "2026-04-08T10:00:00+00:00",
                    "direction_opening": 1,
                    "bid_proxy": 1.1000,
                    "ask_proxy": 1.1001,
                },
            },
            {
                "event_index": 2,
                "timestamp_utc": "2026-04-08T10:05:00+00:00",
                "exit_snapshot": {
                    "event_index": 2,
                    "direction_closing": 1,
                    "bid_proxy": 1.1010,
                    "ask_proxy": 1.1011,
                    "opened_at_event_index": 1,
                    "opened_at_bar_ts_utc": "2026-04-08T10:00:00+00:00",
                    "entry_bid_proxy": 1.1000,
                    "entry_ask_proxy": 1.1001,
                    "bars_held": 1,
                },
                "entry_snapshot": {
                    "event_index": 2,
                    "bar_ts_utc": "2026-04-08T10:05:00+00:00",
                    "direction_opening": -1,
                    "bid_proxy": 1.1010,
                    "ask_proxy": 1.1011,
                },
            },
            {
                "event_index": 3,
                "timestamp_utc": "2026-04-08T10:10:00+00:00",
                "exit_snapshot": {
                    "event_index": 3,
                    "direction_closing": -1,
                    "bid_proxy": 1.1001,
                    "ask_proxy": 1.1002,
                    "opened_at_event_index": 2,
                    "opened_at_bar_ts_utc": "2026-04-08T10:05:00+00:00",
                    "entry_bid_proxy": 1.1010,
                    "entry_ask_proxy": 1.1011,
                    "bars_held": 1,
                },
            },
        ]

        summary = summarize_shadow_trade_accounting(events=events, symbol="EURUSD")

        self.assertEqual(2, summary["trade_count"])
        self.assertAlmostEqual(1.0, summary["realized_trade_coverage"])
        self.assertAlmostEqual(17.0, float(summary["net_pips"] or 0.0), places=6)
        self.assertEqual(1, summary["trades"][0]["direction"])
        self.assertEqual(-1, summary["trades"][1]["direction"])

    def test_snapshot_accounting_applies_round_trip_costs(self) -> None:
        events = [
            {
                "event_index": 1,
                "timestamp_utc": "2026-04-08T10:00:00+00:00",
                "entry_snapshot": {
                    "event_index": 1,
                    "bar_ts_utc": "2026-04-08T10:00:00+00:00",
                    "direction_opening": 1,
                    "bid_proxy": 1.1000,
                    "ask_proxy": 1.1001,
                },
            },
            {
                "event_index": 2,
                "timestamp_utc": "2026-04-08T10:05:00+00:00",
                "exit_snapshot": {
                    "event_index": 2,
                    "direction_closing": 1,
                    "bid_proxy": 1.1009,
                    "ask_proxy": 1.1010,
                    "opened_at_event_index": 1,
                    "opened_at_bar_ts_utc": "2026-04-08T10:00:00+00:00",
                    "entry_bid_proxy": 1.1000,
                    "entry_ask_proxy": 1.1001,
                    "bars_held": 1,
                },
            },
        ]

        summary = summarize_shadow_trade_accounting(
            events=events,
            symbol="EURUSD",
            commission_per_lot=7.0,
            slippage_pips=0.25,
            account_currency="USD",
        )

        self.assertEqual(1, summary["trade_count"])
        self.assertAlmostEqual(8.0, summary["trades"][0]["gross_pips"], places=6)
        self.assertAlmostEqual(1.4, summary["trades"][0]["commission_pips"], places=4)
        self.assertAlmostEqual(0.5, summary["trades"][0]["slippage_pips"], places=6)
        self.assertAlmostEqual(6.1, float(summary["net_pips"] or 0.0), places=4)


if __name__ == "__main__":
    unittest.main()
