from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from selector_manifest import (
    CostModel,
    RuntimeConstraints,
    ThresholdPolicy,
    compute_execution_cost_profile_hash,
    create_rule_manifest,
    resolve_execution_cost_profile,
    save_selector_manifest,
)
from shadow_sweep_report import build_report


class ShadowSweepReportTests(unittest.TestCase):
    def test_build_report_uses_cost_parity_and_event_snapshot_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_path = root / "dataset.csv"
            dataset_path.write_text("stub\n", encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest = create_rule_manifest(
                strategy_symbol="EURUSD",
                rule_family="mean_reversion",
                rule_params={"threshold": 1.0, "sl_value": 1.5, "tp_value": 3.0},
                dataset_path=dataset_path,
                ticks_per_bar=5000,
                cost_model=CostModel(commission_per_lot=7.0, slippage_pips=0.25),
                threshold_policy=ThresholdPolicy(min_edge_pips=0.0, reject_ambiguous=True),
                runtime_constraints=RuntimeConstraints(
                    session_filter_active=True,
                    spread_sanity_max_pips=1.5,
                    max_concurrent_positions=1,
                    daily_loss_stop_usd=100.0,
                ),
                release_stage="paper_live_candidate",
                evaluator_hash="eval-hash",
                logic_hash="logic-hash",
            )
            save_selector_manifest(manifest, manifest_path)
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_hash = saved_manifest["manifest_hash"]

            ladder_path = root / "ladder.json"
            ladder_path.write_text(
                json.dumps(
                    {
                        "generated": [
                            {
                                "profile_id": "p01_guarded_core",
                                "manifest_hash": manifest_hash,
                                "manifest_path": str(manifest_path),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            audit_dir = root / "artifacts"
            profile_dir = audit_dir / "EURUSD" / manifest_hash
            profile_dir.mkdir(parents=True, exist_ok=True)
            events = [
                {
                    "event_index": 1,
                    "timestamp_utc": "2026-04-08T10:00:00+00:00",
                    "bar_ts": "2026-04-08T10:00:00+00:00",
                    "action_state": "open",
                    "reason": "authorized",
                    "signal_direction": 1,
                    "would_open": True,
                    "would_close": False,
                    "would_hold": False,
                    "allow_execution": True,
                    "core_features": {"price_z": -1.6, "spread_z": 0.0, "ma20_slope": 0.0, "ma50_slope": 0.0},
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
                    "timestamp_utc": "2026-04-09T10:00:00+00:00",
                    "bar_ts": "2026-04-09T10:00:00+00:00",
                    "action_state": "close",
                    "reason": "authorized_exit",
                    "signal_direction": 0,
                    "would_open": False,
                    "would_close": True,
                    "would_hold": False,
                    "allow_execution": True,
                    "core_features": {"price_z": 0.0, "spread_z": 0.0, "ma20_slope": 0.0, "ma50_slope": 0.0},
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
            (profile_dir / "events.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            resolved_cost = resolve_execution_cost_profile(manifest)
            (profile_dir / "run_meta.json").write_text(
                json.dumps(
                    {
                        "resolved_execution_cost_profile": resolved_cost,
                        "resolved_execution_cost_profile_hash": compute_execution_cost_profile_hash(resolved_cost),
                    }
                ),
                encoding="utf-8",
            )

            payload, markdown = build_report(
                audit_root=audit_dir,
                ladder_json_path=ladder_path,
                symbol="EURUSD",
                account_currency="USD",
            )

            profile = payload["profiles"]["p01_guarded_core"]
            self.assertTrue(profile["cost_parity_ok"])
            self.assertEqual(1, profile["est_trade_count"])
            self.assertAlmostEqual(6.1, float(profile["est_net_pips"] or 0.0), places=4)
            self.assertFalse(profile["eligible_for_ranking"])
            self.assertIn("trading_days<20", profile["ranking_blockers"])
            self.assertIn("Ranking withheld", markdown)


if __name__ == "__main__":
    unittest.main()
