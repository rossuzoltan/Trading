from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from artifact_manifest import ArtifactManifest
from selector_manifest import CostModel, RuntimeConstraints, ThresholdPolicy, create_rule_manifest, save_selector_manifest
from trading_config import DeploymentPaths

import evaluate_oos
import live_bridge
import mt5_live_preflight


TEST_TMP_ROOT = Path("tests/tmp")


def make_test_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class LiveReadinessTests(unittest.TestCase):
    def test_live_broker_normalizes_open_price_and_protective_levels_to_broker_tick_size(self):
        sent_requests: list[dict] = []

        class FakeMt5Broker:
            ORDER_TYPE_BUY = 0
            ORDER_TYPE_SELL = 1
            TRADE_ACTION_DEAL = 1
            ORDER_TIME_GTC = 0
            ORDER_FILLING_IOC = 1
            ORDER_FILLING_FOK = 2
            ORDER_FILLING_RETURN = 3
            TRADE_RETCODE_DONE = 10009
            SYMBOL_TRADE_MODE_FULL = 4

            def terminal_info(self):
                return SimpleNamespace(trade_allowed=True)

            def account_info(self):
                return SimpleNamespace(trade_allowed=True)

            def symbol_info_tick(self, symbol):
                return SimpleNamespace(bid=1.10002, ask=1.10012)

            def symbol_info(self, symbol):
                return SimpleNamespace(
                    point=0.00001,
                    digits=5,
                    visible=True,
                    volume_min=0.01,
                    volume_max=100.0,
                    volume_step=0.01,
                    trade_stops_level=5,
                    trade_freeze_level=0,
                    trade_tick_size=0.00005,
                    trade_tick_value=1.0,
                    trade_contract_size=100000.0,
                    trade_mode=self.SYMBOL_TRADE_MODE_FULL,
                )

            def order_send(self, request):
                sent_requests.append(dict(request))
                return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=123, price=request.get("price", 0.0))

        broker = live_bridge.LiveMt5Broker(FakeMt5Broker(), symbol="EURUSD")
        result = broker.submit_order(
            live_bridge.OrderIntent(
                symbol="EURUSD",
                action=live_bridge.ActionSpec(live_bridge.ActionType.OPEN, direction=1),
                volume=0.031,
                submitted_time_msc=0,
                requested_price=1.10011,
                sl_price=1.09987,
                tp_price=1.10088,
            )
        )
        self.assertTrue(result.accepted)
        self.assertEqual(1, len(sent_requests))
        self.assertEqual(1.10010, sent_requests[0]["price"])
        self.assertEqual(1.09985, sent_requests[0]["sl"])
        self.assertEqual(1.10090, sent_requests[0]["tp"])
        self.assertEqual(0.03, sent_requests[0]["volume"])

    def test_live_broker_rejects_open_when_symbol_trade_mode_blocks_new_orders(self):
        class FakeMt5Broker:
            ORDER_TYPE_BUY = 0
            ORDER_TYPE_SELL = 1
            TRADE_ACTION_DEAL = 1
            ORDER_TIME_GTC = 0
            ORDER_FILLING_IOC = 1
            ORDER_FILLING_FOK = 2
            ORDER_FILLING_RETURN = 3
            TRADE_RETCODE_DONE = 10009
            SYMBOL_TRADE_MODE_DISABLED = 0
            SYMBOL_TRADE_MODE_CLOSEONLY = 3

            def terminal_info(self):
                return SimpleNamespace(trade_allowed=True)

            def account_info(self):
                return SimpleNamespace(trade_allowed=True)

            def symbol_info_tick(self, symbol):
                return SimpleNamespace(bid=1.1000, ask=1.1002)

            def symbol_info(self, symbol):
                return SimpleNamespace(
                    point=0.00001,
                    digits=5,
                    visible=True,
                    volume_min=0.01,
                    volume_max=100.0,
                    volume_step=0.01,
                    trade_stops_level=0,
                    trade_freeze_level=0,
                    trade_mode=self.SYMBOL_TRADE_MODE_CLOSEONLY,
                )

            def order_send(self, request):
                raise AssertionError("order_send should not be called when trade mode blocks opens")

        broker = live_bridge.LiveMt5Broker(FakeMt5Broker(), symbol="EURUSD")
        result = broker.submit_order(
            live_bridge.OrderIntent(
                symbol="EURUSD",
                action=live_bridge.ActionSpec(live_bridge.ActionType.OPEN, direction=1),
                volume=0.01,
                submitted_time_msc=0,
                requested_price=1.1001,
            )
        )
        self.assertFalse(result.accepted)
        self.assertIn("trade mode", str(result.error).lower())

    def test_live_broker_aggregates_same_direction_strategy_positions_and_closes_all(self):
        sent_requests: list[dict] = []

        class FakeMt5Broker:
            ORDER_TYPE_BUY = 0
            ORDER_TYPE_SELL = 1
            TRADE_ACTION_DEAL = 1
            ORDER_TIME_GTC = 0
            ORDER_FILLING_IOC = 1
            ORDER_FILLING_FOK = 2
            ORDER_FILLING_RETURN = 3
            TRADE_RETCODE_DONE = 10009

            def symbol_info_tick(self, symbol):
                return SimpleNamespace(bid=1.1000, ask=1.1002)

            def symbol_info(self, symbol):
                return SimpleNamespace(point=0.00001, visible=True, volume_min=0.01, volume_max=100.0, volume_step=0.01, trade_stops_level=0, trade_freeze_level=0)

            def positions_get(self, symbol=None):
                return [
                    SimpleNamespace(ticket=11, identifier=101, type=0, volume=0.03, price_open=1.1000, sl=1.0950, tp=1.1100, magic=live_bridge.ORDER_MAGIC, time=1),
                    SimpleNamespace(ticket=12, identifier=102, type=0, volume=0.02, price_open=1.1010, sl=1.0950, tp=1.1100, magic=live_bridge.ORDER_MAGIC, time=2),
                    SimpleNamespace(ticket=99, identifier=999, type=0, volume=0.01, price_open=1.2000, sl=0.0, tp=0.0, magic=777, time=3),
                ]

            def order_send(self, request):
                sent_requests.append(dict(request))
                return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=request.get("position", 0), price=request.get("price", 0.0))

        broker = live_bridge.LiveMt5Broker(FakeMt5Broker(), symbol="EURUSD")
        position = broker.current_position("EURUSD")
        self.assertEqual(position.direction, 1)
        self.assertAlmostEqual(position.volume, 0.05)
        self.assertAlmostEqual(position.entry_price, ((0.03 * 1.1000) + (0.02 * 1.1010)) / 0.05)

        result = broker.submit_order(
            live_bridge.OrderIntent(
                symbol="EURUSD",
                action=live_bridge.ActionSpec(live_bridge.ActionType.CLOSE),
                volume=0.05,
                submitted_time_msc=0,
                requested_price=1.1001,
            )
        )
        self.assertTrue(result.accepted)
        self.assertEqual(len(sent_requests), 2)
        self.assertEqual({request["position"] for request in sent_requests}, {11, 12})

    def test_bootstrap_blocks_manifest_ticks_per_bar_mismatch(self):
        manifest = ArtifactManifest(
            manifest_version="1",
            strategy_symbol="EURUSD",
            model_path="dummy.zip",
            scaler_path="dummy.pkl",
            model_version="dummy-v1",
            model_sha256="x",
            scaler_sha256="y",
            feature_columns=[],
            observation_shape=[1, 1],
            action_map=[],
            dataset_id="dataset",
            sb3_version="2.5.0",
            sb3_contrib_version="2.5.0",
            sklearn_version=__import__("sklearn").__version__,
            ticks_per_bar=5000,
        )
        with patch("live_bridge.resolve_dataset_path", return_value=Path("data/FOREX_MULTI_SET.csv")), \
            patch("live_bridge.dataset_id_for_path", return_value="dataset"), \
            patch("live_bridge.resolve_manifest_path", return_value=Path("models/artifact_manifest_EURUSD.json")), \
            patch("live_bridge.load_manifest", return_value=manifest):
            with self.assertRaises(RuntimeError) as ctx:
                live_bridge.bootstrap_live_runtime(symbol="EURUSD", ticks_per_bar=2000, mt5_module=object())
        self.assertIn("ticks_per_bar", str(ctx.exception))

    def test_bootstrap_live_runtime_loads_trained_churn_guards_from_diagnostics(self):
        tmpdir = make_test_dir("live_bootstrap_guards")
        try:
            diagnostics_path = tmpdir / "training_diagnostics.json"
            diagnostics_path.write_text(
                json.dumps(
                    {
                        "training_window_size": 3,
                        "training_churn_min_hold_bars": 8,
                        "training_churn_action_cooldown": 5,
                        "training_entry_spread_z_limit": 0.75,
                    }
                ),
                encoding="utf-8",
            )
            manifest = ArtifactManifest(
                manifest_version="1",
                strategy_symbol="EURUSD",
                model_path="dummy.zip",
                scaler_path="dummy.pkl",
                model_version="dummy-v1",
                model_sha256="x",
                scaler_sha256="y",
                feature_columns=[],
                observation_shape=[3, 1],
                action_map=[],
                dataset_id="dataset",
                sb3_version="2.5.0",
                sb3_contrib_version="2.5.0",
                sklearn_version=__import__("sklearn").__version__,
                ticks_per_bar=2000,
                training_diagnostics_path=str(diagnostics_path),
            )
            warmup_idx = pd.date_range("2024-01-01", periods=160, freq="min", tz="UTC")
            warmup_frame = pd.DataFrame(
                {
                    "Open": np.linspace(1.0, 1.1, 160),
                    "High": np.linspace(1.0002, 1.1002, 160),
                    "Low": np.linspace(0.9998, 1.0998, 160),
                    "Close": np.linspace(1.0, 1.1, 160),
                    "Volume": np.full(160, 5000.0),
                    "avg_spread": np.full(160, 0.0001),
                    "time_delta_s": np.full(160, 60.0),
                },
                index=warmup_idx,
            )
            scaler = StandardScaler().fit(np.ones((20, len(live_bridge.FEATURE_COLS))))

            class FakeMt5:
                def initialize(self):
                    return True

                def login(self, *_args):
                    return True

                def shutdown(self):
                    return None

                def account_info(self):
                    return SimpleNamespace(equity=1000.0, login=123)

                def positions_get(self, symbol=None):
                    return []

                def symbol_info_tick(self, symbol):
                    return SimpleNamespace(bid=1.1000, ask=1.1001)

                def symbol_info(self, symbol):
                    return SimpleNamespace(point=0.00001, visible=True)

            with patch("live_bridge.resolve_dataset_path", return_value=tmpdir / "dataset.csv"), \
                patch("live_bridge.dataset_id_for_path", return_value="dataset"), \
                patch("live_bridge.resolve_manifest_path", return_value=tmpdir / "artifact_manifest.json"), \
                patch("live_bridge.load_manifest", return_value=manifest), \
                patch("live_bridge.load_validated_model", return_value=SimpleNamespace()), \
                patch("live_bridge.load_validated_vecnormalize", return_value=None), \
                patch("live_bridge.load_validated_scaler", return_value=scaler), \
                patch("live_bridge._load_warmup_bars", return_value=warmup_frame):
                runtime, *_ = live_bridge.bootstrap_live_runtime(
                    symbol="EURUSD",
                    state_path=str(tmpdir / "state.json"),
                    ticks_per_bar=2000,
                    mt5_module=FakeMt5(),
                )
            self.assertEqual(3, runtime.window_size)
            self.assertEqual(8, runtime.churn_min_hold_bars)
            self.assertEqual(5, runtime.churn_action_cooldown)
            self.assertAlmostEqual(0.75, runtime.entry_spread_z_limit)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_replay_and_live_load_the_same_training_runtime_options(self):
        tmpdir = make_test_dir("runtime_option_parity")
        try:
            diagnostics_path = tmpdir / "training_diagnostics.json"
            diagnostics_path.write_text(
                json.dumps(
                    {
                        "training_window_size": 8,
                        "training_churn_min_hold_bars": 6,
                        "training_churn_action_cooldown": 4,
                        "training_entry_spread_z_limit": 0.8,
                        "training_alpha_gate_enabled": True,
                        "training_alpha_gate_model": "auto",
                        "training_alpha_gate_probability_threshold": 0.57,
                        "training_alpha_gate_probability_margin": 0.04,
                        "training_alpha_gate_min_edge_pips": 0.2,
                        "baseline_target_horizon_bars": 12,
                    }
                ),
                encoding="utf-8",
            )

            replay_options = evaluate_oos._load_training_runtime_options(diagnostics_path)
            live_options = live_bridge._load_training_runtime_options(diagnostics_path, default_window_size=1)

            self.assertEqual(replay_options, live_options)
            self.assertEqual(8, replay_options["window_size"])
            self.assertEqual(6, replay_options["churn_min_hold_bars"])
            self.assertEqual(4, replay_options["churn_action_cooldown"])
            self.assertAlmostEqual(0.8, replay_options["entry_spread_z_limit"])
            self.assertTrue(replay_options["alpha_gate_enabled"])
            self.assertEqual(12, replay_options["baseline_target_horizon_bars"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_bootstrap_live_runtime_warns_when_training_diagnostics_are_missing(self):
        tmpdir = make_test_dir("live_bootstrap_missing_diag")
        try:
            manifest = ArtifactManifest(
                manifest_version="1",
                strategy_symbol="EURUSD",
                model_path="dummy.zip",
                scaler_path="dummy.pkl",
                model_version="dummy-v1",
                model_sha256="x",
                scaler_sha256="y",
                feature_columns=[],
                observation_shape=[3, 1],
                action_map=[],
                dataset_id="dataset",
                sb3_version="2.5.0",
                sb3_contrib_version="2.5.0",
                sklearn_version=__import__("sklearn").__version__,
                ticks_per_bar=2000,
                training_diagnostics_path=str(tmpdir / "missing_training_diagnostics.json"),
            )
            warmup_idx = pd.date_range("2024-01-01", periods=160, freq="min", tz="UTC")
            warmup_frame = pd.DataFrame(
                {
                    "Open": np.linspace(1.0, 1.1, 160),
                    "High": np.linspace(1.0002, 1.1002, 160),
                    "Low": np.linspace(0.9998, 1.0998, 160),
                    "Close": np.linspace(1.0, 1.1, 160),
                    "Volume": np.full(160, 5000.0),
                    "avg_spread": np.full(160, 0.0001),
                    "time_delta_s": np.full(160, 60.0),
                },
                index=warmup_idx,
            )
            scaler = StandardScaler().fit(np.ones((20, len(live_bridge.FEATURE_COLS))))

            class FakeMt5:
                def initialize(self):
                    return True

                def login(self, *_args):
                    return True

                def shutdown(self):
                    return None

                def account_info(self):
                    return SimpleNamespace(equity=1000.0, login=123)

                def positions_get(self, symbol=None):
                    return []

                def symbol_info_tick(self, symbol):
                    return SimpleNamespace(bid=1.1000, ask=1.1001)

                def symbol_info(self, symbol):
                    return SimpleNamespace(point=0.00001, visible=True)

            with patch("live_bridge.resolve_dataset_path", return_value=tmpdir / "dataset.csv"), \
                patch("live_bridge.dataset_id_for_path", return_value="dataset"), \
                patch("live_bridge.resolve_manifest_path", return_value=tmpdir / "artifact_manifest.json"), \
                patch("live_bridge.load_manifest", return_value=manifest), \
                patch("live_bridge.load_validated_model", return_value=SimpleNamespace()), \
                patch("live_bridge.load_validated_vecnormalize", return_value=None), \
                patch("live_bridge.load_validated_scaler", return_value=scaler), \
                patch("live_bridge._load_warmup_bars", return_value=warmup_frame), \
                self.assertLogs("live_bridge", level="WARNING") as captured:
                live_bridge.bootstrap_live_runtime(
                    symbol="EURUSD",
                    state_path=str(tmpdir / "state.json"),
                    ticks_per_bar=2000,
                    mt5_module=FakeMt5(),
                )
            self.assertTrue(any("default runtime guard settings" in line for line in captured.output))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_preflight_reports_gate_and_manifest_blockers(self):
        tmpdir = make_test_dir("preflight")
        try:
            paths = DeploymentPaths(
                diagnostics_path=tmpdir / "training_diagnostics_eurusd.json",
                gate_path=tmpdir / "deployment_gate_eurusd.json",
                ops_attestation_path=tmpdir / "ops_attestation_eurusd.json",
                live_preflight_path=tmpdir / "live_preflight_eurusd.json",
                execution_audit_path=tmpdir / "execution_audit_eurusd.jsonl",
            )
            paths.gate_path.write_text(json.dumps({"approved_for_live": False, "blockers": ["Training diagnostics missing."]}), encoding="utf-8")
            manifest = ArtifactManifest(
                manifest_version="1",
                strategy_symbol="EURUSD",
                model_path="dummy.zip",
                scaler_path="dummy.pkl",
                model_version="dummy-v1",
                model_sha256="x",
                scaler_sha256="y",
                feature_columns=[],
                observation_shape=[1, 1],
                action_map=[],
                dataset_id="dataset",
                sb3_version="2.5.0",
                sb3_contrib_version="2.5.0",
                sklearn_version=__import__("sklearn").__version__,
                ticks_per_bar=2000,
            )
            with patch("mt5_live_preflight.deployment_paths", return_value=paths), \
                patch("mt5_live_preflight.resolve_selector_manifest_path", return_value=None), \
                patch("mt5_live_preflight.resolve_manifest_path", return_value=tmpdir / "artifact_manifest_EURUSD.json"), \
                patch("mt5_live_preflight.load_manifest", return_value=manifest), \
                patch("mt5_live_preflight.importlib.util.find_spec", return_value=None):
                report = mt5_live_preflight.build_report("EURUSD", 2000)
            self.assertFalse(report["approved_for_live_runtime"])
            self.assertTrue(any("MetaTrader5 package is not installed" in blocker for blocker in report["blockers"]))
            self.assertTrue(paths.live_preflight_path.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_preflight_blocks_hedging_accounts(self):
        tmpdir = make_test_dir("preflight_hedging")
        try:
            paths = DeploymentPaths(
                diagnostics_path=tmpdir / "training_diagnostics_eurusd.json",
                gate_path=tmpdir / "deployment_gate_eurusd.json",
                ops_attestation_path=tmpdir / "ops_attestation_eurusd.json",
                live_preflight_path=tmpdir / "live_preflight_eurusd.json",
                execution_audit_path=tmpdir / "execution_audit_eurusd.jsonl",
            )
            paths.gate_path.write_text(json.dumps({"approved_for_live": True, "blockers": []}), encoding="utf-8")
            paths.ops_attestation_path.write_text(json.dumps({"approved": True, "blockers": []}), encoding="utf-8")
            paths.execution_audit_path.write_text(
                "\n".join(
                    json.dumps({"accepted": True, "fill_delta_pips": 0.01, "retcode": 10009})
                    for _ in range(20)
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = ArtifactManifest(
                manifest_version="1",
                strategy_symbol="EURUSD",
                model_path="dummy.zip",
                scaler_path="dummy.pkl",
                model_version="dummy-v1",
                model_sha256="x",
                scaler_sha256="y",
                feature_columns=[],
                observation_shape=[1, 1],
                action_map=[],
                dataset_id="dataset",
                sb3_version="2.5.0",
                sb3_contrib_version="2.5.0",
                sklearn_version=__import__("sklearn").__version__,
                ticks_per_bar=2000,
            )
            def _terminal_info():
                payload = SimpleNamespace(connected=True, trade_allowed=True)
                payload._asdict = lambda: {"connected": True, "trade_allowed": True}
                return payload

            def _account_info():
                payload = SimpleNamespace(trade_allowed=True, margin_mode=2)
                payload._asdict = lambda: {"trade_allowed": True, "margin_mode": 2}
                return payload

            def _symbol_info():
                payload = SimpleNamespace(visible=True, volume_min=0.01, volume_step=0.01, trade_stops_level=0)
                payload._asdict = lambda: {
                    "visible": True,
                    "volume_min": 0.01,
                    "volume_step": 0.01,
                    "trade_stops_level": 0,
                }
                return payload

            fake_mt5 = SimpleNamespace(
                ACCOUNT_MARGIN_MODE_RETAIL_HEDGING=2,
                initialize=lambda: True,
                login=lambda *args, **kwargs: True,
                terminal_info=_terminal_info,
                account_info=_account_info,
                symbol_info=lambda symbol: _symbol_info(),
                positions_get=lambda symbol=None: [],
                shutdown=lambda: None,
                symbol_select=lambda symbol, visible: True,
            )
            with patch("mt5_live_preflight.deployment_paths", return_value=paths), \
                patch("mt5_live_preflight.resolve_selector_manifest_path", return_value=None), \
                patch("mt5_live_preflight.resolve_manifest_path", return_value=tmpdir / "artifact_manifest_EURUSD.json"), \
                patch("mt5_live_preflight.load_manifest", return_value=manifest), \
                patch("mt5_live_preflight.importlib.util.find_spec", return_value=object()), \
                patch.dict("sys.modules", {"MetaTrader5": fake_mt5}):
                report = mt5_live_preflight.build_report("EURUSD", 2000)
            self.assertFalse(report["approved_for_live_runtime"])
            self.assertTrue(any("hedging" in blocker.lower() for blocker in report["blockers"]))
            self.assertEqual(2000, report["bar_construction_ticks_per_bar"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_preflight_blocks_foreign_positions_by_default(self):
        tmpdir = make_test_dir("preflight_foreign")
        try:
            paths = DeploymentPaths(
                diagnostics_path=tmpdir / "training_diagnostics_eurusd.json",
                gate_path=tmpdir / "deployment_gate_eurusd.json",
                ops_attestation_path=tmpdir / "ops_attestation_eurusd.json",
                live_preflight_path=tmpdir / "live_preflight_eurusd.json",
                execution_audit_path=tmpdir / "execution_audit_eurusd.jsonl",
            )
            paths.gate_path.write_text(json.dumps({"approved_for_live": True, "blockers": []}), encoding="utf-8")
            paths.ops_attestation_path.write_text(json.dumps({"approved": True, "blockers": []}), encoding="utf-8")
            paths.execution_audit_path.write_text(
                "\n".join(
                    json.dumps({"accepted": True, "fill_delta_pips": 0.01, "retcode": 10009})
                    for _ in range(20)
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = ArtifactManifest(
                manifest_version="1",
                strategy_symbol="EURUSD",
                model_path="dummy.zip",
                scaler_path="dummy.pkl",
                model_version="dummy-v1",
                model_sha256="x",
                scaler_sha256="y",
                feature_columns=[],
                observation_shape=[1, 1],
                action_map=[],
                dataset_id="dataset",
                sb3_version="2.5.0",
                sb3_contrib_version="2.5.0",
                sklearn_version=__import__("sklearn").__version__,
                ticks_per_bar=2000,
            )

            def _terminal_info():
                payload = SimpleNamespace(connected=True, trade_allowed=True)
                payload._asdict = lambda: {"connected": True, "trade_allowed": True}
                return payload

            def _account_info():
                payload = SimpleNamespace(trade_allowed=True, margin_mode=0)
                payload._asdict = lambda: {"trade_allowed": True, "margin_mode": 0}
                return payload

            def _symbol_info():
                payload = SimpleNamespace(visible=True, volume_min=0.01, volume_step=0.01, trade_stops_level=0)
                payload._asdict = lambda: {
                    "visible": True,
                    "volume_min": 0.01,
                    "volume_step": 0.01,
                    "trade_stops_level": 0,
                }
                return payload

            fake_mt5 = SimpleNamespace(
                ACCOUNT_MARGIN_MODE_RETAIL_HEDGING=2,
                initialize=lambda: True,
                login=lambda *args, **kwargs: True,
                terminal_info=_terminal_info,
                account_info=_account_info,
                symbol_info=lambda symbol: _symbol_info(),
                positions_get=lambda symbol=None: [
                    SimpleNamespace(ticket=1, identifier=1, type=0, volume=0.01, price_open=1.1000, sl=0.0, tp=0.0, magic=999999, time=1)
                ],
                shutdown=lambda: None,
                symbol_select=lambda symbol, visible: True,
            )
            with patch("mt5_live_preflight.deployment_paths", return_value=paths), \
                patch("mt5_live_preflight.resolve_selector_manifest_path", return_value=None), \
                patch("mt5_live_preflight.resolve_manifest_path", return_value=tmpdir / "artifact_manifest_EURUSD.json"), \
                patch("mt5_live_preflight.load_manifest", return_value=manifest), \
                patch("mt5_live_preflight.importlib.util.find_spec", return_value=object()), \
                patch.dict("sys.modules", {"MetaTrader5": fake_mt5}):
                report = mt5_live_preflight.build_report("EURUSD", 2000)
            self.assertFalse(report["approved_for_live_runtime"])
            self.assertTrue(any("non-strategy position" in blocker.lower() for blocker in report["blockers"]))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_preflight_blocks_symbols_with_disabled_trade_mode(self):
        tmpdir = make_test_dir("preflight_trade_mode")
        try:
            paths = DeploymentPaths(
                diagnostics_path=tmpdir / "training_diagnostics_eurusd.json",
                gate_path=tmpdir / "deployment_gate_eurusd.json",
                ops_attestation_path=tmpdir / "ops_attestation_eurusd.json",
                live_preflight_path=tmpdir / "live_preflight_eurusd.json",
                execution_audit_path=tmpdir / "execution_audit_eurusd.jsonl",
            )
            paths.gate_path.write_text(json.dumps({"approved_for_live": True, "blockers": []}), encoding="utf-8")
            paths.ops_attestation_path.write_text(json.dumps({"approved": True, "blockers": []}), encoding="utf-8")
            paths.execution_audit_path.write_text(
                "\n".join(json.dumps({"accepted": True, "fill_delta_pips": 0.01, "retcode": 10009}) for _ in range(20)) + "\n",
                encoding="utf-8",
            )
            manifest = ArtifactManifest(
                manifest_version="1",
                strategy_symbol="EURUSD",
                model_path="dummy.zip",
                scaler_path="dummy.pkl",
                model_version="dummy-v1",
                model_sha256="x",
                scaler_sha256="y",
                feature_columns=[],
                observation_shape=[1, 1],
                action_map=[],
                dataset_id="dataset",
                sb3_version="2.5.0",
                sb3_contrib_version="2.5.0",
                sklearn_version=__import__("sklearn").__version__,
                ticks_per_bar=2000,
            )

            def _terminal_info():
                payload = SimpleNamespace(connected=True, trade_allowed=True)
                payload._asdict = lambda: {"connected": True, "trade_allowed": True}
                return payload

            def _account_info():
                payload = SimpleNamespace(trade_allowed=True, margin_mode=0)
                payload._asdict = lambda: {"trade_allowed": True, "margin_mode": 0}
                return payload

            def _symbol_info():
                payload = SimpleNamespace(
                    visible=True,
                    digits=5,
                    point=0.00001,
                    volume_min=0.01,
                    volume_max=1.0,
                    volume_step=0.01,
                    trade_stops_level=0,
                    trade_freeze_level=0,
                    trade_tick_size=0.00001,
                    trade_tick_value=1.0,
                    trade_contract_size=100000.0,
                    trade_mode=0,
                )
                payload._asdict = lambda: {
                    "visible": True,
                    "digits": 5,
                    "point": 0.00001,
                    "volume_min": 0.01,
                    "volume_max": 1.0,
                    "volume_step": 0.01,
                    "trade_stops_level": 0,
                    "trade_freeze_level": 0,
                    "trade_tick_size": 0.00001,
                    "trade_tick_value": 1.0,
                    "trade_contract_size": 100000.0,
                    "trade_mode": 0,
                }
                return payload

            fake_mt5 = SimpleNamespace(
                ACCOUNT_MARGIN_MODE_RETAIL_HEDGING=2,
                SYMBOL_TRADE_MODE_DISABLED=0,
                SYMBOL_TRADE_MODE_CLOSEONLY=3,
                SYMBOL_TRADE_MODE_LONGONLY=1,
                SYMBOL_TRADE_MODE_SHORTONLY=2,
                SYMBOL_TRADE_MODE_FULL=4,
                initialize=lambda: True,
                login=lambda *args, **kwargs: True,
                terminal_info=_terminal_info,
                account_info=_account_info,
                symbol_info=lambda symbol: _symbol_info(),
                positions_get=lambda symbol=None: [],
                shutdown=lambda: None,
                symbol_select=lambda symbol, visible: True,
            )
            with patch("mt5_live_preflight.deployment_paths", return_value=paths), \
                patch("mt5_live_preflight.resolve_selector_manifest_path", return_value=None), \
                patch("mt5_live_preflight.resolve_manifest_path", return_value=tmpdir / "artifact_manifest_EURUSD.json"), \
                patch("mt5_live_preflight.load_manifest", return_value=manifest), \
                patch("mt5_live_preflight.importlib.util.find_spec", return_value=object()), \
                patch.dict("sys.modules", {"MetaTrader5": fake_mt5}):
                report = mt5_live_preflight.build_report("EURUSD", 2000)
            self.assertFalse(report["approved_for_live_runtime"])
            self.assertTrue(any("trade mode blocks new orders" in blocker.lower() for blocker in report["blockers"]))
            self.assertEqual(0, report["symbol_capabilities"]["trade_mode"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_preflight_prefers_selector_manifest_for_rc1_rule_pack(self):
        tmpdir = make_test_dir("preflight_selector_manifest")
        try:
            paths = DeploymentPaths(
                diagnostics_path=tmpdir / "training_diagnostics_eurusd.json",
                gate_path=tmpdir / "deployment_gate_eurusd.json",
                ops_attestation_path=tmpdir / "ops_attestation_eurusd.json",
                live_preflight_path=tmpdir / "live_preflight_eurusd.json",
                execution_audit_path=tmpdir / "execution_audit_eurusd.jsonl",
            )
            paths.gate_path.write_text(json.dumps({"approved_for_live": False, "blockers": []}), encoding="utf-8")
            dataset_path = tmpdir / "DATA_CLEAN_VOLUME_5000.csv"
            dataset_path.write_text("stub dataset", encoding="utf-8")
            manifest_path = tmpdir / "manifest.json"
            manifest = create_rule_manifest(
                strategy_symbol="EURUSD",
                rule_family="mean_reversion",
                rule_params={"threshold": 1.5, "sl_value": 1.5, "tp_value": 3.0},
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
                evaluator_hash="eval",
                logic_hash="logic",
            )
            save_selector_manifest(manifest, manifest_path)

            with patch("mt5_live_preflight.deployment_paths", return_value=paths), \
                patch("mt5_live_preflight.resolve_selector_manifest_path", return_value=manifest_path), \
                patch("mt5_live_preflight.resolve_manifest_path", side_effect=FileNotFoundError("legacy missing")), \
                patch("mt5_live_preflight.importlib.util.find_spec", return_value=None):
                report = mt5_live_preflight.build_report("EURUSD", 5000)

            self.assertEqual("selector_manifest", report["manifest_source"])
            self.assertEqual("RULE", report["manifest_engine_type"])
            self.assertEqual(str(manifest_path), report["manifest_path"])
            self.assertEqual(5000, report["manifest_bar_construction_ticks_per_bar"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
