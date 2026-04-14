from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from paper_live_metrics import compute_drift_metrics, summarize_shadow_events
from selector_manifest import (
    CostModel,
    RuntimeConstraints,
    ThresholdPolicy,
    create_rule_manifest,
    save_selector_manifest,
)
from tools.paper_live_gate import build_paper_live_gate


class PaperLiveGateTests(unittest.TestCase):
    def _write_manifest_bundle(self, root: Path, *, symbol: str = "EURUSD") -> Path:
        pack_dir = root / "models" / "rc1" / "eurusd_5k_v1_mr_rc1"
        pack_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = root / "data" / "DATA_CLEAN_VOLUME_5000.csv"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text("Gmt time,Symbol,Volume\n2024-01-01T00:00:00Z,EURUSD,5000\n", encoding="utf-8")
        manifest = create_rule_manifest(
            strategy_symbol=symbol,
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
        manifest_path = pack_dir / "manifest.json"
        save_selector_manifest(manifest, manifest_path)
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scoreboard = {
            "name": "eurusd_5k_v1_mr_rc1",
            "symbol": symbol,
            "ticks_per_bar": 5000,
            "release_stage": "paper_live_candidate",
            "live_trading_approved": False,
            "manifest_hash": saved_manifest["manifest_hash"],
            "evaluator_hash": saved_manifest["evaluator_hash"],
            "logic_hash": saved_manifest["logic_hash"],
            "rc_candidate": {
                "net_pnl_usd": 120.0,
                "profit_factor": 1.8,
                "expectancy_usd": 4.0,
                "trade_count": 30,
                "long_count": 15,
                "short_count": 15,
            },
            "baselines": {
                "runtime_flat": {"net_pnl_usd": 0.0, "profit_factor": 0.0, "expectancy_usd": 0.0, "trade_count": 0, "long_count": 0, "short_count": 0},
                "runtime_always_short": {"net_pnl_usd": -10.0, "profit_factor": 0.8, "expectancy_usd": -1.0, "trade_count": 5, "long_count": 0, "short_count": 5},
                "runtime_trend": {"net_pnl_usd": -12.0, "profit_factor": 0.7, "expectancy_usd": -1.2, "trade_count": 6, "long_count": 0, "short_count": 6},
                "runtime_mean_reversion": {"net_pnl_usd": 110.0, "profit_factor": 1.5, "expectancy_usd": 3.5, "trade_count": 28, "long_count": 14, "short_count": 14},
            },
        }
        (pack_dir / "baseline_scoreboard_rc1.json").write_text(json.dumps(scoreboard), encoding="utf-8")
        (pack_dir / "baseline_scoreboard_rc1.md").write_text("# scoreboard\n", encoding="utf-8")
        (pack_dir / "release_notes_rc1.md").write_text("# notes\n", encoding="utf-8")
        return manifest_path

    def _write_shadow_events(self, root: Path, manifest_hash: str, *, days: int, actionable_events: int) -> Path:
        shadow_dir = root / "artifacts" / "shadow" / "EURUSD" / manifest_hash
        shadow_dir.mkdir(parents=True, exist_ok=True)
        events_path = shadow_dir / "events.jsonl"
        lines: list[str] = []
        base_ts = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
        open_directions: list[int] = []
        for index in range(days):
            ts = base_ts + timedelta(days=index)
            is_actionable_open = index < actionable_events
            signal_direction = 1 if index % 2 == 0 else -1
            if is_actionable_open:
                open_directions.append(signal_direction)
            record = {
                "timestamp_utc": ts.isoformat(),
                "symbol": "EURUSD",
                "ticks_per_bar": 5000,
                "manifest_hash": manifest_hash,
                "logic_hash": "logic-hash",
                "evaluator_hash": "eval-hash",
                "signal_direction": signal_direction if is_actionable_open else 0,
                "action_state": "open" if is_actionable_open else "flat",
                "would_open": is_actionable_open,
                "would_close": False,
                "would_hold": False,
                "no_trade_reason": "authorized" if is_actionable_open else "no signal",
                "spread_pips": 0.7,
                "session_filter_pass": True,
                "risk_filter_pass": True,
                "spread_ok": True,
                "position_state": ("long" if signal_direction > 0 else "short") if is_actionable_open else "flat",
            }
            lines.append(json.dumps(record))
        extra_close_events = max(0, actionable_events - days)
        for index in range(extra_close_events):
            ts = base_ts + timedelta(days=index, hours=1)
            prior_direction = open_directions[index] if index < len(open_directions) else -1
            lines.append(
                json.dumps(
                    {
                        "timestamp_utc": ts.isoformat(),
                        "symbol": "EURUSD",
                        "ticks_per_bar": 5000,
                        "manifest_hash": manifest_hash,
                        "logic_hash": "logic-hash",
                        "evaluator_hash": "eval-hash",
                        "signal_direction": 0,
                        "action_state": "close",
                        "would_open": False,
                        "would_close": True,
                        "would_hold": False,
                        "no_trade_reason": "authorized",
                        "spread_pips": 0.7,
                        "session_filter_pass": True,
                        "risk_filter_pass": True,
                        "spread_ok": True,
                        "position_state": "long" if prior_direction > 0 else "short",
                    }
                )
            )
        events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return events_path

    def _write_gate_dependencies(self, root: Path) -> tuple[Path, Path, Path]:
        restart_path = root / "models" / "restart_drill_eurusd.json"
        restart_path.parent.mkdir(parents=True, exist_ok=True)
        restart_path.write_text(
            json.dumps(
                {
                    "startup_reconcile_ok": True,
                    "state_restored_ok": True,
                    "confirmed_position_restored_ok": True,
                }
            ),
            encoding="utf-8",
        )
        preflight_path = root / "models" / "live_preflight_eurusd.json"
        preflight_path.write_text(json.dumps({"approved_for_live_runtime": True}), encoding="utf-8")
        ops_path = root / "models" / "ops_attestation_eurusd.json"
        ops_path.write_text(json.dumps({"approved": True}), encoding="utf-8")
        return restart_path, preflight_path, ops_path

    def test_summarize_shadow_events_counts_actionable_evidence(self) -> None:
        events = [
            {
                "timestamp_utc": "2026-04-01T09:00:00+00:00",
                "symbol": "EURUSD",
                "signal_direction": -1,
                "would_open": True,
                "would_close": False,
                "session_filter_pass": True,
                "risk_filter_pass": True,
                "spread_ok": True,
                "position_state": "flat",
            },
            {
                "timestamp_utc": "2026-04-02T09:00:00+00:00",
                "symbol": "EURUSD",
                "signal_direction": 0,
                "would_open": False,
                "would_close": True,
                "session_filter_pass": True,
                "risk_filter_pass": True,
                "spread_ok": True,
                "position_state": "short",
            },
        ]
        summary = summarize_shadow_events(events, min_trading_days=2, min_actionable_events=2)
        self.assertEqual(2, summary["trading_days"])
        self.assertEqual(2, summary["actionable_event_count"])
        self.assertTrue(summary["evidence_sufficient"])

    def test_compute_drift_metrics_flags_multiple_threshold_breaches_as_critical(self) -> None:
        summary = {
            "rates": {
                "signal_density": 0.50,
                "would_open_density": 0.45,
                "spread_rejection_pct": 25.0,
                "session_rejection_pct": 21.0,
            },
            "directional_occupancy": {"long": 0.90},
        }
        drift = compute_drift_metrics(
            summary,
            replay_reference={
                "signal_density": 0.20,
                "would_open_density": 0.20,
                "spread_rejection_pct": 0.0,
                "session_rejection_pct": 0.0,
                "long_share": 0.0,
            },
        )
        self.assertTrue(drift["critical"])

    def test_build_paper_live_gate_promotes_with_sufficient_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = self._write_manifest_bundle(root)
            manifest_hash = json.loads(manifest_path.read_text(encoding="utf-8"))["manifest_hash"]
            self._write_shadow_events(root, manifest_hash, days=20, actionable_events=30)
            restart_path, preflight_path, ops_path = self._write_gate_dependencies(root)

            payload = build_paper_live_gate(
                manifest_path=manifest_path,
                shadow_dir=root / "artifacts" / "shadow",
                restart_drill_path=restart_path,
                preflight_path=preflight_path,
                ops_attestation_path=ops_path,
                output_dir=root / "artifacts" / "gates",
            )

            self.assertEqual("paper_live_profitable", payload["final_verdict"])
            self.assertEqual("paper_live_profitable", payload["anchor_status"])

    def test_build_paper_live_gate_stays_candidate_when_shadow_sample_is_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = self._write_manifest_bundle(root)
            manifest_hash = json.loads(manifest_path.read_text(encoding="utf-8"))["manifest_hash"]
            self._write_shadow_events(root, manifest_hash, days=5, actionable_events=5)
            restart_path, preflight_path, ops_path = self._write_gate_dependencies(root)

            payload = build_paper_live_gate(
                manifest_path=manifest_path,
                shadow_dir=root / "artifacts" / "shadow",
                restart_drill_path=restart_path,
                preflight_path=preflight_path,
                ops_attestation_path=ops_path,
                output_dir=root / "artifacts" / "gates",
            )

            self.assertEqual("candidate", payload["final_verdict"])
            self.assertEqual("candidate", payload["anchor_status"])

    def test_build_paper_live_gate_includes_historical_replay_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = self._write_manifest_bundle(root)
            manifest_hash = json.loads(manifest_path.read_text(encoding="utf-8"))["manifest_hash"]
            self._write_shadow_events(root, manifest_hash, days=5, actionable_events=5)
            restart_path, preflight_path, ops_path = self._write_gate_dependencies(root)
            historical_path = manifest_path.parent / "mt5_historical_replay_report.json"
            historical_path.write_text(
                json.dumps({"overall_verdict": "DRIFT_CRITICAL", "bars_processed": 100}),
                encoding="utf-8",
            )

            payload = build_paper_live_gate(
                manifest_path=manifest_path,
                shadow_dir=root / "artifacts" / "shadow",
                restart_drill_path=restart_path,
                preflight_path=preflight_path,
                ops_attestation_path=ops_path,
                output_dir=root / "artifacts" / "gates",
            )

            self.assertTrue(payload["historical_replay_status"]["present"])
            self.assertFalse(payload["historical_replay_status"]["ok"])


if __name__ == "__main__":
    unittest.main()
