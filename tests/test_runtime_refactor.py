from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.vec_env import DummyVecEnv

import live_bridge
from artifact_manifest import ArtifactManifest, load_validated_scaler
from event_pipeline import (
    BarBuilderState,
    BaseBroker,
    BrokerPositionSnapshot,
    JsonStateStore,
    ModelPolicy,
    Mt5CursorTickSource,
    RiskEngine,
    RiskLimits,
    RuntimeEngine,
    RuntimeSnapshot,
    SubmitResult,
    TickCursor,
    TickEvent,
    VolumeBar,
    VolumeBarBuilder,
)
from feature_engine import FEATURE_COLS, WARMUP_BARS
from runtime_common import ActionSpec, ActionType, ConfirmedPosition, build_action_map
from symbol_utils import pip_size_for_symbol, pip_value_for_volume
from trading_env import ForexTradingEnv
from train_agent import sync_vecnormalize_stats, wrap_vecnormalize


TEST_TMP_ROOT = Path("tests/tmp")


def make_test_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class FakeModel:
    def __init__(self):
        self.last_observation = None

    def predict(self, observation, action_masks=None, deterministic=True):
        self.last_observation = np.array(observation, copy=True)
        if action_masks is not None:
            for idx, allowed in enumerate(action_masks):
                if allowed:
                    return idx, None
        return 0, None


class RecordingNormalizer:
    def __init__(self, offset: float = 1.0):
        self.offset = offset
        self.seen = None

    def normalize_obs(self, observation):
        self.seen = np.array(observation, copy=True)
        return np.array(observation, copy=True) + self.offset


class FakeMt5:
    COPY_TICKS_ALL = 0

    def __init__(self, ticks=None):
        self._ticks = list(ticks or [])
        self.initialized = False

    def initialize(self):
        self.initialized = True
        return True

    def login(self, login, password, server):
        return True

    def shutdown(self):
        self.initialized = False

    def account_info(self):
        return SimpleNamespace(equity=1000.0, login=123)

    def positions_get(self, symbol=None):
        return []

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=1.1000, ask=1.1001)

    def symbol_info(self, symbol):
        return SimpleNamespace(point=0.00001, visible=True)

    def order_send(self, request):
        return SimpleNamespace(retcode=10009, order=123, price=request.get("price", 0.0))

    def copy_ticks_from(self, symbol, start_dt, count, flags):
        start_ms = int(round(start_dt.timestamp() * 1000))
        rows = [row for row in self._ticks if int(row["time_msc"]) >= start_ms]
        return rows[:count]


class StubFeatureEngine:
    def __init__(self):
        self._buffer = pd.DataFrame(
            [
                {
                    "atr_14": 0.0010,
                    "spread_z": 0.0,
                }
            ]
        )
        self._latest = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    def push(self, bar_series: pd.Series) -> None:
        self._latest = np.array(
            [
                float(bar_series["Close"]),
                float(bar_series["Volume"]),
                float(bar_series.get("time_delta_s", 0.0)),
            ],
            dtype=np.float32,
        )
        self._buffer = pd.concat(
            [
                self._buffer,
                pd.DataFrame(
                    [
                        {
                            "atr_14": 0.0010,
                            "spread_z": 0.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    @property
    def latest_observation(self) -> np.ndarray:
        return self._latest


class SequencePolicy:
    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0

    def decide(self, observation, action_mask):
        desired = self.actions[self.index] if self.index < len(self.actions) else 0
        self.index += 1
        if action_mask[desired]:
            return desired, ACTION_MAP[desired]
        for idx, allowed in enumerate(action_mask):
            if allowed:
                return idx, ACTION_MAP[idx]
        return 0, ACTION_MAP[0]


class RejectingBroker(BaseBroker):
    def submit_order(self, intent):
        return SubmitResult(accepted=False, error="rejected")

    def current_position(self, symbol):
        return BrokerPositionSnapshot(symbol=symbol)

    def current_equity(self, symbol, mark_price=None):
        return 1000.0


class DrawdownBroker(BaseBroker):
    def __init__(self):
        self.position = BrokerPositionSnapshot(
            symbol="EURUSD",
            direction=1,
            volume=0.05,
            entry_price=1.1000,
            broker_ticket=77,
        )
        self.close_requests = 0

    def advance_bar(self, bar):
        return None

    def submit_order(self, intent):
        if intent.action.action_type == ActionType.CLOSE:
            self.close_requests += 1
            self.position = BrokerPositionSnapshot(symbol=intent.symbol)
            return SubmitResult(accepted=True, order_id=9001)
        return SubmitResult(accepted=False, error="open disabled in test")

    def current_position(self, symbol):
        return self.position

    def current_equity(self, symbol, mark_price=None):
        return 800.0


ACTION_MAP = build_action_map([0.5], [1.0])


def make_bar(index: int, close: float) -> VolumeBar:
    ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(minutes=index)
    start_ms = int(ts.timestamp() * 1000)
    return VolumeBar(
        timestamp=ts.to_pydatetime(),
        open=close - 0.0002,
        high=close + 0.0003,
        low=close - 0.0003,
        close=close,
        volume=3.0,
        avg_spread=0.0001,
        time_delta_s=60.0 if index else 0.0,
        start_time_msc=start_ms,
        end_time_msc=start_ms + 60_000,
    )


def make_runtime(broker: BaseBroker, policy) -> RuntimeEngine:
    snapshot = RuntimeSnapshot(last_equity=1000.0, high_water_mark=1000.0, day_start_equity=1000.0)
    return RuntimeEngine(
        symbol="EURUSD",
        feature_engine=StubFeatureEngine(),
        policy=policy,
        broker=broker,
        action_map=ACTION_MAP,
        risk_engine=RiskEngine(RiskLimits(), snapshot=snapshot, initial_equity=1000.0),
        snapshot=snapshot,
        state_store=None,
    )


class ReplayTraceHarness:
    def __init__(self):
        from event_pipeline import ReplayBroker

        self.broker = ReplayBroker(symbol="EURUSD", commission_per_lot=0.0, slippage_pips=0.0)
        self.runtime = make_runtime(self.broker, SequencePolicy([2, 0, 1]))
        self.runtime.startup_reconcile()

    def process_bars(self, bars):
        trace = []
        for bar in bars:
            result = self.runtime.process_bar(bar)
            trace.append(
                (
                    round(bar.open, 6),
                    round(bar.close, 6),
                    result.action_index,
                    result.action.action_type.value,
                    result.position_direction,
                    tuple(np.round(result.features, 6)),
                )
            )
        return trace


class RuntimeRefactorTests(unittest.TestCase):
    def test_live_startup_no_longer_crashes(self):
        warmup_idx = pd.date_range("2024-01-01", periods=WARMUP_BARS, freq="min", tz="UTC")
        warmup_frame = pd.DataFrame(
            {
                "Open": np.linspace(1.0, 1.1, WARMUP_BARS),
                "High": np.linspace(1.0002, 1.1002, WARMUP_BARS),
                "Low": np.linspace(0.9998, 1.0998, WARMUP_BARS),
                "Close": np.linspace(1.0, 1.1, WARMUP_BARS),
                "Volume": np.full(WARMUP_BARS, 5000.0),
                "avg_spread": np.full(WARMUP_BARS, 0.0001),
                "time_delta_s": np.full(WARMUP_BARS, 60.0),
            },
            index=warmup_idx,
        )
        scaler = StandardScaler().fit(np.ones((20, len(FEATURE_COLS))))
        manifest = ArtifactManifest(
            manifest_version="1",
            strategy_symbol="EURUSD",
            model_path="dummy.zip",
            scaler_path="dummy.pkl",
            model_version="dummy-v1",
            model_sha256="x",
            scaler_sha256="y",
            feature_columns=list(FEATURE_COLS),
            observation_shape=[1, len(FEATURE_COLS) + 4],
            action_map=[],
            dataset_id="dataset",
            sb3_version="2.5.0",
            sb3_contrib_version="2.5.0",
            sklearn_version=__import__("sklearn").__version__,
        )
        fake_mt5 = FakeMt5()
        tmpdir = make_test_dir("live_startup")
        try:
            with patch("live_bridge.resolve_dataset_path", return_value=Path(tmpdir) / "dataset.csv"), \
                patch("live_bridge.dataset_id_for_path", return_value="dataset"), \
                patch("live_bridge.resolve_manifest_path", return_value=Path(tmpdir) / "artifact_manifest.json"), \
            patch("live_bridge.load_manifest", return_value=manifest), \
            patch("live_bridge.load_validated_model", return_value=FakeModel()), \
            patch("live_bridge.load_validated_vecnormalize", return_value=None), \
            patch("live_bridge.load_validated_scaler", return_value=scaler), \
            patch("live_bridge._load_warmup_bars", return_value=warmup_frame):
                runtime, builder, store, source = live_bridge.bootstrap_live_runtime(
                    symbol="EURUSD",
                    state_path=str(Path(tmpdir) / "state.json"),
                    ticks_per_bar=2000,
                    mt5_module=fake_mt5,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        self.assertIsNotNone(runtime)
        self.assertTrue(runtime.snapshot.safe_mode_active)
        self.assertEqual(builder.state.ticks_per_bar, 2000)
        self.assertIsInstance(store, JsonStateStore)
        self.assertIsInstance(source, Mt5CursorTickSource)

    def test_tick_ingestion_processes_every_tick_once(self):
        ticks = [
            {"time_msc": 1000, "bid": 1.0, "ask": 1.0001},
            {"time_msc": 1000, "bid": 1.0001, "ask": 1.0002},
            {"time_msc": 1000, "bid": 1.0002, "ask": 1.0003},
            {"time_msc": 1001, "bid": 1.0003, "ask": 1.0004},
            {"time_msc": 1001, "bid": 1.0004, "ask": 1.0005},
            {"time_msc": 1002, "bid": 1.0005, "ask": 1.0006},
        ]
        source = Mt5CursorTickSource(FakeMt5(ticks), batch_size=2, initial_lookback_seconds=999999)
        batch_one, cursor = source.fetch("EURUSD", TickCursor(time_msc=999, offset=0))
        batch_two, cursor = source.fetch("EURUSD", cursor)
        all_ticks = batch_one + batch_two
        self.assertEqual(len(all_ticks), len(ticks))
        self.assertEqual([tick.time_msc for tick in all_ticks], [1000, 1000, 1000, 1001, 1001, 1002])
        self.assertEqual(cursor.time_msc, 1002)
        self.assertEqual(cursor.offset, 1)

    def test_broker_fill_failure_does_not_desync_state(self):
        runtime = make_runtime(RejectingBroker(), SequencePolicy([2]))
        runtime.startup_reconcile()
        result = runtime.process_bar(make_bar(0, 1.1000))
        self.assertEqual(runtime.confirmed_position.direction, 0)
        self.assertFalse(result.submit_result.accepted)
        self.assertEqual(runtime.snapshot.consecutive_broker_failures, 1)

    def test_artifact_mismatch_blocks_inference(self):
        tmpdir = make_test_dir("artifact_mismatch")
        try:
            scaler = StandardScaler().fit(
                np.arange(3 * len(FEATURE_COLS), dtype=np.float64).reshape(3, len(FEATURE_COLS))
            )
            scaler_path = Path(tmpdir) / "scaler.pkl"
            joblib.dump(scaler, scaler_path)
            manifest = ArtifactManifest(
                manifest_version="1",
                strategy_symbol="EURUSD",
                model_path="unused.zip",
                scaler_path=str(scaler_path),
                model_version="v1",
                model_sha256="unused",
                scaler_sha256="0" * 64,
                feature_columns=list(FEATURE_COLS),
                observation_shape=[1, len(FEATURE_COLS) + 4],
                action_map=[{"action_type": "HOLD", "direction": None, "sl_value": None, "tp_value": None}],
                dataset_id="wrong",
                sb3_version=__import__("stable_baselines3").__version__,
                sb3_contrib_version=__import__("sb3_contrib").__version__,
                sklearn_version=__import__("sklearn").__version__,
            )
            with self.assertRaises(RuntimeError):
                load_validated_scaler(
                    manifest,
                    expected_symbol="EURUSD",
                    expected_action_map=[ActionSpec(ActionType.HOLD)],
                    expected_observation_shape=[1, len(FEATURE_COLS) + 4],
                    expected_dataset_id="different",
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_model_policy_applies_obs_normalizer_before_predict(self):
        model = FakeModel()
        normalizer = RecordingNormalizer(offset=0.5)
        action_map = build_action_map([0.5], [1.0])
        policy = ModelPolicy(model, action_map, obs_normalizer=normalizer)

        observation = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        action_mask = np.array([True, False, False, False], dtype=bool)
        action_index, action = policy.decide(observation, action_mask)

        self.assertEqual(action_index, 0)
        self.assertEqual(action.action_type, ActionType.HOLD)
        np.testing.assert_allclose(normalizer.seen, observation)
        np.testing.assert_allclose(model.last_observation, observation + 0.5)

    def test_training_env_reward_is_bounded_and_tracks_risk_terms(self):
        frame = pd.DataFrame(
            {
                "Close": [1.1000, 1.1000, 1.1010, 1.0980, 1.0980],
                "High": [1.1002, 1.1002, 1.1012, 1.0982, 1.0982],
                "Low": [1.0998, 1.0998, 1.1008, 1.0978, 1.0978],
                "atr_14": [0.0010] * 5,
            }
        )
        env = ForexTradingEnv(
            df=frame,
            feature_columns=[],
            sl_options=[100.0],
            tp_options=[100.0],
            window_size=1,
            spread_pips=1.0,
            max_slippage_pips=0.0,
            lot_size=0.01,
            swap_pips_per_day=0.0,
            random_start=False,
            min_episode_steps=3,
            use_trailing_stop=False,
            use_variable_spread=False,
            atr_scaled=False,
            vol_scaling=False,
        )

        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            _obs, _info = reset_out

        step_one = env.step(2)
        reward_one = step_one[1]
        info_one = step_one[-1]
        self.assertGreaterEqual(reward_one, -1.0)
        self.assertLessEqual(reward_one, 1.0)
        self.assertGreater(info_one["transaction_cost_penalty"], 0.0)
        self.assertGreater(info_one["position_change_penalty"], 0.0)

        step_two = env.step(0)
        reward_two = step_two[1]
        info_two = step_two[-1]
        self.assertGreaterEqual(reward_two, -1.0)
        self.assertLessEqual(reward_two, 1.0)
        self.assertEqual(info_two["position_change_penalty"], 0.0)

        step_three = env.step(0)
        reward_three = step_three[1]
        info_three = step_three[-1]
        self.assertGreaterEqual(reward_three, -1.0)
        self.assertLessEqual(reward_three, 1.0)
        self.assertGreater(info_three["drawdown"], 0.0)

    def test_training_env_open_does_not_consume_next_bar_stop_immediately(self):
        frame = pd.DataFrame(
            {
                "Close": [1.1000, 1.1000, 1.1000, 1.0990],
                "High": [1.1001, 1.1001, 1.1001, 1.0991],
                "Low": [1.0999, 1.0999, 1.0990, 1.0989],
                "feature": [0.0, 1.0, 2.0, 3.0],
                "atr_14": [0.0010] * 4,
            }
        )
        env = ForexTradingEnv(
            df=frame,
            feature_columns=["feature"],
            sl_options=[5.0],
            tp_options=[100.0],
            window_size=1,
            spread_pips=0.0,
            max_slippage_pips=0.0,
            lot_size=0.01,
            swap_pips_per_day=0.0,
            random_start=False,
            min_episode_steps=2,
            use_trailing_stop=False,
            use_variable_spread=False,
            atr_scaled=False,
            vol_scaling=False,
        )
        env.reset()
        env.step(2)
        self.assertEqual(env.position, 1)
        env.step(0)
        self.assertEqual(env.position, 0)
        self.assertEqual(env.trade_log[-1]["reason"], "SL")

    def test_vecnormalize_stats_are_synced_into_eval_env(self):
        frame = pd.DataFrame(
            {
                "Close": [1.1000, 1.1001, 1.1002, 1.1003],
                "High": [1.1001, 1.1002, 1.1003, 1.1004],
                "Low": [1.0999, 1.1000, 1.1001, 1.1002],
                "feature": [0.0, 1.0, 2.0, 3.0],
                "atr_14": [0.0010] * 4,
            }
        )

        def make_env():
            return ForexTradingEnv(
                df=frame,
                feature_columns=["feature"],
                sl_options=[5.0],
                tp_options=[10.0],
                window_size=1,
                spread_pips=0.0,
                max_slippage_pips=0.0,
                lot_size=0.01,
                swap_pips_per_day=0.0,
                random_start=False,
                min_episode_steps=2,
                use_trailing_stop=False,
                use_variable_spread=False,
                atr_scaled=False,
                vol_scaling=False,
            )

        train_vec = wrap_vecnormalize(DummyVecEnv([make_env]), training=True)
        val_vec = wrap_vecnormalize(DummyVecEnv([make_env]), training=False)
        train_vec.obs_rms.mean[:] = 3.0
        train_vec.obs_rms.var[:] = 4.0
        train_vec.ret_rms.mean = 2.0
        train_vec.ret_rms.var = 5.0
        sync_vecnormalize_stats(train_vec, val_vec)
        np.testing.assert_allclose(val_vec.obs_rms.mean, train_vec.obs_rms.mean)
        np.testing.assert_allclose(val_vec.obs_rms.var, train_vec.obs_rms.var)
        self.assertEqual(val_vec.ret_rms.mean, train_vec.ret_rms.mean)
        self.assertEqual(val_vec.ret_rms.var, train_vec.ret_rms.var)
        self.assertFalse(val_vec.training)
        self.assertFalse(val_vec.norm_reward)
        train_vec.close()
        val_vec.close()

    def test_jpy_pip_logic_is_symbol_aware(self):
        self.assertEqual(pip_size_for_symbol("EURUSD"), 0.0001)
        self.assertEqual(pip_size_for_symbol("USDJPY"), 0.01)
        self.assertAlmostEqual(
            pip_value_for_volume("EURUSD", price=1.10, volume_lots=1.0),
            10.0,
            places=6,
        )
        self.assertAlmostEqual(
            pip_value_for_volume("USDJPY", price=150.0, volume_lots=1.0),
            1000.0 / 150.0,
            places=6,
        )

    def test_restart_recovery_restores_state_correctly(self):
        tmpdir = make_test_dir("restart_recovery")
        try:
            store = JsonStateStore(Path(tmpdir) / "state.json", ticks_per_bar=3)
            snapshot = RuntimeSnapshot(
                cursor=TickCursor(time_msc=1000, offset=2),
                bar_builder=BarBuilderState(
                    ticks_per_bar=3,
                    tick_count=2,
                    bar_open=1.1000,
                    bar_high=1.1004,
                    bar_low=1.0998,
                    spread_total=0.0002,
                    bar_start_time_msc=1000,
                    last_emitted_bar_start_time_msc=500,
                ),
                confirmed_position=ConfirmedPosition(direction=1, entry_price=1.1000, volume=0.05),
                last_equity=1000.0,
                high_water_mark=1010.0,
                day_start_equity=1000.0,
            )
            store.save(snapshot)
            restored = store.load()
            builder = VolumeBarBuilder(3, restored.bar_builder)
            bar = builder.push_tick(TickEvent(time_msc=1200, bid=1.1001, ask=1.1002))
            self.assertIsNotNone(bar)
            self.assertEqual(restored.cursor.time_msc, 1000)
            self.assertEqual(restored.cursor.offset, 2)
            self.assertEqual(restored.confirmed_position.direction, 1)
            self.assertAlmostEqual(bar.open, 1.1000, places=6)
            self.assertAlmostEqual(bar.high, 1.1004, places=6)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_runtime_kill_switch_flattens_and_enters_safe_mode(self):
        broker = DrawdownBroker()
        snapshot = RuntimeSnapshot(last_equity=1000.0, high_water_mark=1000.0, day_start_equity=1000.0)
        runtime = RuntimeEngine(
            symbol="EURUSD",
            feature_engine=StubFeatureEngine(),
            policy=SequencePolicy([0]),
            broker=broker,
            action_map=ACTION_MAP,
            risk_engine=RiskEngine(
                RiskLimits(max_drawdown_fraction=0.10, daily_loss_fraction=0.50),
                snapshot=snapshot,
                initial_equity=1000.0,
            ),
            snapshot=snapshot,
            state_store=None,
        )
        runtime.startup_reconcile()
        result = runtime.process_bar(make_bar(0, 1.1000))
        self.assertTrue(result.kill_switch_active)
        self.assertEqual(broker.close_requests, 1)
        self.assertTrue(runtime.snapshot.kill_switch_active)
        self.assertTrue(runtime.snapshot.safe_mode_active)

    def test_backtest_live_replay_parity_holds_for_shared_pipeline(self):
        ticks = [
            TickEvent(time_msc=1000, bid=1.1000, ask=1.1001),
            TickEvent(time_msc=1001, bid=1.1002, ask=1.1003),
            TickEvent(time_msc=1002, bid=1.1001, ask=1.1002),
            TickEvent(time_msc=2000, bid=1.1005, ask=1.1006),
            TickEvent(time_msc=2001, bid=1.1004, ask=1.1005),
            TickEvent(time_msc=2002, bid=1.1006, ask=1.1007),
            TickEvent(time_msc=3000, bid=1.1007, ask=1.1008),
            TickEvent(time_msc=3001, bid=1.1008, ask=1.1009),
            TickEvent(time_msc=3002, bid=1.1009, ask=1.1010),
        ]
        builder = VolumeBarBuilder(3)
        bars = []
        for tick in ticks:
            bar = builder.push_tick(tick)
            if bar is not None:
                bars.append(bar)

        live_harness = ReplayTraceHarness()
        live_trace = []
        tick_builder = VolumeBarBuilder(3)
        for tick in ticks:
            bar = tick_builder.push_tick(tick)
            if bar is not None:
                live_trace.extend(live_harness.process_bars([bar]))

        replay_harness = ReplayTraceHarness()
        replay_trace = replay_harness.process_bars(bars)
        self.assertEqual(live_trace, replay_trace)


if __name__ == "__main__":
    unittest.main()
