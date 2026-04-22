from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from evaluate_oos import ReplayContext, _build_runtime_parity_verdict, run_replay
from selector_manifest import _file_sha256
from tools.verify_v1_rc import verify_component_hashes


class VerifyRc1Tests(unittest.TestCase):
    def test_verify_component_hashes_accepts_current_files(self) -> None:
        payload = {
            "evaluator_hash": _file_sha256(Path("evaluate_oos.py")),
            "logic_hash": _file_sha256(Path("strategies") / "rule_logic.py"),
        }
        result = verify_component_hashes(payload)
        self.assertEqual(payload["evaluator_hash"], result["evaluator_hash"])
        self.assertEqual(payload["logic_hash"], result["logic_hash"])

    def test_verify_component_hashes_rejects_truth_engine_drift(self) -> None:
        payload = {
            "evaluator_hash": "not-the-current-hash",
            "logic_hash": _file_sha256(Path("strategies") / "rule_logic.py"),
        }
        with self.assertRaises(RuntimeError):
            verify_component_hashes(payload)

    def test_runtime_parity_uses_rule_provider_for_rule_context(self) -> None:
        frame = pd.DataFrame()
        context = ReplayContext(
            symbol="EURUSD",
            source="unit",
            dataset_path=Path("dataset.csv"),
            action_map=tuple(),
            model=None,
            obs_normalizer=None,
            scaler=None,
            execution_cost_profile={"slippage_pips": 0.25},
            reward_profile={},
            warmup_frame=frame,
            replay_frame=frame,
            replay_feature_frame=frame,
            full_feature_frame=frame,
            trainable_feature_frame=frame,
            holdout_feature_frame=frame,
            holdout_start_utc=None,
            diagnostics_path=None,
            manifest_path=None,
            artifact_metadata={},
            runtime_options={},
            engine_type="RULE",
            rule_family="mean_reversion",
            rule_params={"threshold": 1.0},
        )
        observed_providers: list[object] = []

        def _fake_evaluate_policy(*, replay_context, action_index_provider, disable_alpha_gate=False):
            observed_providers.append(action_index_provider)
            return {
                "metrics": {
                    "trade_count": 0,
                    "net_pnl_usd": 0.0,
                    "validation_status": {"passed": True},
                }
            }

        with (
            patch("evaluate_oos._evaluate_runtime_baselines", return_value={"runtime_flat": {"metrics": {}}}),
            patch("evaluate_oos._load_research_baseline_summary", return_value=None),
            patch("evaluate_oos._evaluate_research_best_runtime_baseline", return_value=None),
            patch("evaluate_oos._evaluate_policy", side_effect=_fake_evaluate_policy),
        ):
            verdict = _build_runtime_parity_verdict(
                context=context,
                replay_metrics={"trade_count": 0, "net_pnl_usd": 0.0, "validation_status": {"passed": True}},
                training_diagnostics=None,
            )

        self.assertIn("gate_off_replay_metrics", verdict)
        self.assertEqual(3, len(observed_providers))
        self.assertTrue(all(callable(provider) for provider in observed_providers))

    def test_run_replay_disables_runtime_alpha_gate_for_rule_provider(self) -> None:
        frame = pd.DataFrame()
        context = ReplayContext(
            symbol="EURUSD",
            source="unit",
            dataset_path=Path("dataset.csv"),
            action_map=tuple(),
            model=None,
            obs_normalizer=None,
            scaler=None,
            execution_cost_profile={"slippage_pips": 0.25},
            reward_profile={},
            warmup_frame=frame,
            replay_frame=frame,
            replay_feature_frame=frame,
            full_feature_frame=frame,
            trainable_feature_frame=frame,
            holdout_feature_frame=frame,
            holdout_start_utc=None,
            diagnostics_path=None,
            manifest_path=None,
            artifact_metadata={},
            runtime_options={},
            engine_type="RULE",
            rule_family="mean_reversion",
            rule_params={"threshold": 1.0},
        )
        observed_disable_flags: list[bool] = []

        def _fake_build_runtime(*, context, use_policy, disable_alpha_gate, prefitted_alpha_gate=None):
            observed_disable_flags.append(bool(disable_alpha_gate))
            runtime = SimpleNamespace(
                minimal_post_cost_reward=False,
                confirmed_position=SimpleNamespace(direction=0, time_in_trade_bars=0),
            )
            broker = SimpleNamespace(trade_log=[], execution_log=[])
            return runtime, broker

        with (
            patch("evaluate_oos._build_runtime", side_effect=_fake_build_runtime),
            patch("evaluate_oos._frame_to_bars", return_value=[]),
        ):
            run_replay(replay_context=context, action_index_provider=lambda **_: 0)

        self.assertEqual([True], observed_disable_flags)


if __name__ == "__main__":
    unittest.main()
