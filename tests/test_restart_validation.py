from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from artifact_manifest import ArtifactManifest, save_manifest
from event_pipeline import BarBuilderState, JsonStateStore, RuntimeSnapshot, TickCursor, VolumeBar
from runtime_common import ConfirmedPosition
from trading_config import deployment_paths

import live_operating_checklist
import ops_attestation_helper
import restart_drill


TEST_TMP_ROOT = Path("tests/tmp")


def make_test_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class DummyRuntime:
    def __init__(self, state_store: JsonStateStore, *, symbol: str) -> None:
        self.symbol = symbol.upper()
        self.state_store = state_store
        self.snapshot = RuntimeSnapshot(
            cursor=TickCursor(time_msc=0, offset=0),
            bar_builder=BarBuilderState(ticks_per_bar=2000),
            confirmed_position=ConfirmedPosition(
                direction=1,
                entry_price=1.1000,
                volume=0.10,
                broker_ticket=77,
                order_id=88,
                last_confirmed_time_msc=0,
            ),
            last_equity=1000.0,
            high_water_mark=1000.0,
            day_start_equity=1000.0,
        )
        self.confirmed_position = self.snapshot.confirmed_position

    def process_bar(self, bar: VolumeBar) -> SimpleNamespace:
        self.snapshot.cursor = TickCursor(time_msc=bar.end_time_msc, offset=self.snapshot.cursor.offset + 1)
        self.snapshot.last_tick_time_msc = bar.end_time_msc
        self.snapshot.confirmed_position.time_in_trade_bars += 1
        self.snapshot.confirmed_position.last_confirmed_time_msc = bar.end_time_msc
        return SimpleNamespace(kill_switch_active=False)

    def persist(self) -> None:
        self.state_store.save(self.snapshot)


class RestartValidationTests(unittest.TestCase):
    def test_restart_drill_round_trip_uses_fake_mt5_style_bootstrap(self):
        tmpdir = make_test_dir("restart_drill")
        try:
            state_path = tmpdir / "live_state_eurusd.json"
            report_path = tmpdir / "restart_drill_eurusd.json"

            def bootstrap_fn(*, symbol, state_path, ticks_per_bar, mt5_module):
                store = JsonStateStore(state_path, ticks_per_bar=ticks_per_bar)
                runtime = DummyRuntime(store, symbol=symbol)
                if store.path.exists():
                    runtime.snapshot = store.load()
                    runtime.confirmed_position = runtime.snapshot.confirmed_position
                builder = SimpleNamespace(state=runtime.snapshot.bar_builder)
                source = SimpleNamespace()
                return runtime, builder, store, source

            with unittest.mock.patch("live_bridge.bootstrap_live_runtime", side_effect=bootstrap_fn):
                report = restart_drill.run_restart_drill(
                    symbol="EURUSD",
                    state_path=str(state_path),
                    report_path=str(report_path),
                    ticks_per_bar=2000,
                    mt5_module=SimpleNamespace(),
                    bars_before_restart=1,
                    bars_after_restart=1,
                )

            self.assertTrue(report.startup_reconcile_ok)
            self.assertTrue(report.state_restored_ok)
            self.assertTrue(report.confirmed_position_restored_ok)
            self.assertTrue(report_path.exists())
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["symbol"], "EURUSD")
            self.assertEqual(saved["ticks_per_bar"], 2000)
            self.assertEqual(saved["evidence_mode"], "fake_mt5")
            self.assertFalse(saved["attestable_for_live"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_json_state_store_recovers_from_backup_when_primary_is_corrupt(self):
        tmpdir = make_test_dir("state_store_backup")
        try:
            store = JsonStateStore(tmpdir / "state.json", ticks_per_bar=2000)
            first_snapshot = RuntimeSnapshot(
                cursor=TickCursor(time_msc=100, offset=1),
                bar_builder=BarBuilderState(ticks_per_bar=2000, tick_count=2),
                confirmed_position=ConfirmedPosition(direction=1, entry_price=1.1000, volume=0.05),
                last_equity=1001.0,
                high_water_mark=1005.0,
                day_start_equity=1000.0,
            )
            second_snapshot = RuntimeSnapshot(
                cursor=TickCursor(time_msc=200, offset=2),
                bar_builder=BarBuilderState(ticks_per_bar=2000, tick_count=3),
                confirmed_position=ConfirmedPosition(direction=0),
                last_equity=1002.0,
                high_water_mark=1006.0,
                day_start_equity=1000.0,
            )
            store.save(first_snapshot)
            store.save(second_snapshot)
            store.path.write_text("{invalid json", encoding="utf-8")

            restored = store.load()

            self.assertEqual(restored.cursor.time_msc, 100)
            self.assertEqual(restored.cursor.offset, 1)
            self.assertEqual(restored.confirmed_position.direction, 1)
            self.assertAlmostEqual(restored.last_equity, 1001.0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_ops_attestation_helper_validates_evidence(self):
        tmpdir = make_test_dir("ops_attestation")
        try:
            paths = deployment_paths("EURUSD", model_dir=tmpdir)
            write_jsonl(
                paths.execution_audit_path,
                [
                    {"accepted": True, "fill_delta_pips": 0.05, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": -0.02, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": 0.01, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": 0.00, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": 0.03, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": -0.04, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": 0.02, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": 0.01, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": -0.01, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": 0.00, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": 0.02, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": -0.03, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": 0.04, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": 0.01, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": -0.02, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": 0.03, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": 0.01, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": 0.02, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": -0.01, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": 0.00, "retcode": 10009},
                    {"accepted": True, "fill_delta_pips": 0.02, "retcode": 10009},
                ],
            )
            restart_report = {
                "symbol": "EURUSD",
                "ticks_per_bar": 2000,
                "state_path": str(tmpdir / "live_state_eurusd.json"),
                "report_path": str(tmpdir / "restart_drill_eurusd.json"),
                "startup_reconcile_ok": True,
                "state_restored_ok": True,
                "confirmed_position_restored_ok": True,
                "evidence_mode": "real_mt5",
                "attestable_for_live": True,
                "bars_processed_before_restart": 1,
                "bars_processed_after_restart": 1,
                "pre_restart_snapshot": {"cursor": {"time_msc": 1}},
                "post_restart_snapshot": {"cursor": {"time_msc": 1}},
                "notes": [],
            }
            (tmpdir / "restart_drill_eurusd.json").write_text(json.dumps(restart_report, indent=2), encoding="utf-8")

            payload = ops_attestation_helper.build_ops_attestation(
                symbol="EURUSD",
                attested_by="qa",
                notes="shadow evidence",
                shadow_days_completed=14,
                execution_audit_path=paths.execution_audit_path,
                restart_drill_path=tmpdir / "restart_drill_eurusd.json",
                output_path=paths.ops_attestation_path,
                model_dir=tmpdir,
            )
            self.assertTrue(payload["approved"])
            self.assertTrue(payload["execution_drift_ok"])
            self.assertTrue(payload["position_reconciliation_ok"])
            self.assertEqual(payload["restart_drill_evidence"]["evidence_mode"], "real_mt5")
            self.assertTrue(paths.ops_attestation_path.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_ops_attestation_helper_rejects_fake_restart_evidence(self):
        tmpdir = make_test_dir("ops_attestation_fake")
        try:
            paths = deployment_paths("EURUSD", model_dir=tmpdir)
            write_jsonl(
                paths.execution_audit_path,
                [{"accepted": True, "fill_delta_pips": 0.01, "retcode": 10009} for _ in range(20)],
            )
            restart_report = {
                "symbol": "EURUSD",
                "ticks_per_bar": 2000,
                "state_path": str(tmpdir / "live_state_eurusd.json"),
                "report_path": str(tmpdir / "restart_drill_eurusd.json"),
                "startup_reconcile_ok": True,
                "state_restored_ok": True,
                "confirmed_position_restored_ok": True,
                "evidence_mode": "fake_mt5",
                "attestable_for_live": False,
                "bars_processed_before_restart": 1,
                "bars_processed_after_restart": 1,
                "pre_restart_snapshot": {"cursor": {"time_msc": 1}},
                "post_restart_snapshot": {"cursor": {"time_msc": 1}},
                "notes": [],
            }
            (tmpdir / "restart_drill_eurusd.json").write_text(json.dumps(restart_report, indent=2), encoding="utf-8")

            payload = ops_attestation_helper.build_ops_attestation(
                symbol="EURUSD",
                attested_by="qa",
                notes="shadow evidence",
                shadow_days_completed=14,
                execution_audit_path=paths.execution_audit_path,
                restart_drill_path=tmpdir / "restart_drill_eurusd.json",
                output_path=paths.ops_attestation_path,
                model_dir=tmpdir,
            )
            self.assertFalse(payload["approved"])
            self.assertTrue(any("fake" in blocker.lower() or "attest" in blocker.lower() for blocker in payload["blockers"]))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_operating_checklist_reports_ready_when_evidence_is_present(self):
        tmpdir = make_test_dir("operating_checklist")
        try:
            paths = deployment_paths("EURUSD", model_dir=tmpdir)
            manifest = ArtifactManifest(
                manifest_version="1",
                strategy_symbol="EURUSD",
                model_path=str(tmpdir / "model.zip"),
                scaler_path=str(tmpdir / "scaler.pkl"),
                model_version="v1",
                model_sha256="deadbeef",
                scaler_sha256="deadbeef",
                feature_columns=[],
                observation_shape=[1, 1],
                action_map=[],
                dataset_id="dataset",
                sb3_version="1",
                sb3_contrib_version="1",
                sklearn_version="1",
                ticks_per_bar=2000,
            )
            save_manifest(manifest, tmpdir / "artifact_manifest_EURUSD.json")
            (tmpdir / "deployment_gate_eurusd.json").write_text(
                json.dumps({"approved_for_live": True, "blockers": []}, indent=2),
                encoding="utf-8",
            )
            (tmpdir / "live_preflight_eurusd.json").write_text(
                json.dumps(
                    {
                        "approved_for_live_runtime": True,
                        "blockers": [],
                        "account_mode_supported": True,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            write_jsonl(
                paths.execution_audit_path,
                [{"accepted": True, "fill_delta_pips": 0.01, "retcode": 10009} for _ in range(20)],
            )
            (tmpdir / "restart_drill_eurusd.json").write_text(
                json.dumps(
                    {
                        "symbol": "EURUSD",
                        "ticks_per_bar": 2000,
                        "state_path": str(tmpdir / "live_state_eurusd.json"),
                        "report_path": str(tmpdir / "restart_drill_eurusd.json"),
                        "startup_reconcile_ok": True,
                        "state_restored_ok": True,
                        "confirmed_position_restored_ok": True,
                        "evidence_mode": "real_mt5",
                        "attestable_for_live": True,
                        "bars_processed_before_restart": 1,
                        "bars_processed_after_restart": 1,
                        "pre_restart_snapshot": {"cursor": {"time_msc": 1}},
                        "post_restart_snapshot": {"cursor": {"time_msc": 1}},
                        "notes": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            ops_attestation_helper.build_ops_attestation(
                symbol="EURUSD",
                attested_by="qa",
                notes="shadow evidence",
                shadow_days_completed=14,
                execution_audit_path=paths.execution_audit_path,
                restart_drill_path=tmpdir / "restart_drill_eurusd.json",
                output_path=paths.ops_attestation_path,
                model_dir=tmpdir,
            )

            checklist = live_operating_checklist.build_operating_checklist(
                symbol="EURUSD",
                ticks_per_bar=2000,
                model_dir=tmpdir,
            )
            self.assertTrue(checklist["approved_for_live"])
            self.assertTrue(all(item["ok"] for item in checklist["items"]))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
