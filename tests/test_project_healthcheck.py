from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from artifact_manifest import create_manifest, save_manifest
from feature_engine import FEATURE_COLS
from runtime_common import STATE_FEATURE_COUNT, build_action_map
from tools import project_healthcheck
import project_paths
from trading_config import ACTION_SL_MULTS, ACTION_TP_MULTS


TEST_TMP_ROOT = Path("tests/tmp")


def make_test_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_manifest_fixture(tmpdir: Path, symbol: str) -> tuple[Path, Path, Path, Path]:
    data_dir = tmpdir / "data"
    models_dir = tmpdir / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "DATA_CLEAN_VOLUME.csv"
    dataset_path.write_text(
        "Gmt time,Symbol,Volume\n2024-01-01T00:00:00Z,"
        f"{symbol},2000\n",
        encoding="utf-8",
    )
    (data_dir / "FOREX_MULTI_SET.csv").write_text(dataset_path.read_text(encoding="utf-8"), encoding="utf-8")

    model_path = models_dir / f"model_{symbol.lower()}_best.zip"
    scaler_path = models_dir / f"scaler_{symbol}.pkl"
    model_path.write_bytes(b"model")
    scaler_path.write_bytes(b"scaler")

    manifest = create_manifest(
        strategy_symbol=symbol,
        model_path=model_path,
        scaler_path=scaler_path,
        model_version=f"{symbol.lower()}-v1",
        feature_columns=FEATURE_COLS,
        observation_shape=[1, len(FEATURE_COLS) + STATE_FEATURE_COUNT],
        action_map=build_action_map(list(ACTION_SL_MULTS), list(ACTION_TP_MULTS)),
        dataset_path=dataset_path,
        bar_construction_ticks_per_bar=2000,
    )
    manifest_path = models_dir / f"artifact_manifest_{symbol}.json"
    save_manifest(manifest, manifest_path)
    return data_dir, models_dir, dataset_path, manifest_path


class ProjectHealthcheckTests(unittest.TestCase):
    def test_check_runtime_assets_validates_symbol_scoped_manifest_bundle(self):
        tmpdir = make_test_dir("healthcheck_bundle")
        try:
            data_dir, models_dir, dataset_path, manifest_path = make_manifest_fixture(tmpdir, "GBPUSD")
            model_loader = Mock(return_value=object())
            scaler_loader = Mock(return_value=object())
            vecnormalize_loader = Mock(return_value=None)

            with patch.object(project_healthcheck, "ROOT_DIR", tmpdir), \
                patch.object(project_healthcheck, "DATA_DIR", data_dir), \
                patch.object(project_healthcheck, "MODELS_DIR", models_dir), \
                patch.object(project_healthcheck, "resolve_dataset_path", return_value=dataset_path), \
                patch.object(project_healthcheck, "load_dataset_build_info", return_value={"bar_construction_ticks_per_bar": 2000}), \
                patch.object(project_healthcheck, "validate_dataset_bar_spec", return_value=None), \
                patch.object(project_healthcheck, "validate_dataset_integrity", return_value={"symbols": ["GBPUSD"]}), \
                patch.object(project_healthcheck, "validate_symbol_bar_spec", return_value={"partial_rows": 0, "exact_match_rows": 1}), \
                patch.object(project_healthcheck, "list_manifest_paths", return_value=[manifest_path]), \
                patch.object(project_healthcheck, "load_validated_model", model_loader), \
                patch.object(project_healthcheck, "load_validated_scaler", scaler_loader), \
                patch.object(project_healthcheck, "load_validated_vecnormalize", vecnormalize_loader):
                issues, warnings = project_healthcheck._check_runtime_assets()

            self.assertEqual([], issues)
            self.assertEqual([], warnings)
            self.assertEqual("GBPUSD", model_loader.call_args.kwargs["expected_symbol"])
            self.assertEqual(1, scaler_loader.call_count)
            self.assertEqual(1, vecnormalize_loader.call_count)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_check_runtime_assets_reports_manifest_validation_failures(self):
        tmpdir = make_test_dir("healthcheck_manifest_failure")
        try:
            data_dir, models_dir, dataset_path, manifest_path = make_manifest_fixture(tmpdir, "GBPUSD")
            with patch.object(project_healthcheck, "ROOT_DIR", tmpdir), \
                patch.object(project_healthcheck, "DATA_DIR", data_dir), \
                patch.object(project_healthcheck, "MODELS_DIR", models_dir), \
                patch.object(project_healthcheck, "resolve_dataset_path", return_value=dataset_path), \
                patch.object(project_healthcheck, "load_dataset_build_info", return_value={"bar_construction_ticks_per_bar": 2000}), \
                patch.object(project_healthcheck, "validate_dataset_bar_spec", return_value=None), \
                patch.object(project_healthcheck, "validate_dataset_integrity", return_value={"symbols": ["GBPUSD"]}), \
                patch.object(project_healthcheck, "validate_symbol_bar_spec", return_value={"partial_rows": 0, "exact_match_rows": 1}), \
                patch.object(project_healthcheck, "list_manifest_paths", return_value=[manifest_path]), \
                patch.object(project_healthcheck, "load_validated_model", side_effect=RuntimeError("Dataset id mismatch")), \
                patch.object(project_healthcheck, "load_validated_scaler", return_value=object()), \
                patch.object(project_healthcheck, "load_validated_vecnormalize", return_value=None):
                issues, warnings = project_healthcheck._check_runtime_assets()

            self.assertTrue(any("Runtime bundle GBPUSD failed validation" in issue for issue in issues))
            self.assertEqual([], warnings)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class ProjectPathsTests(unittest.TestCase):
    def test_resolve_model_path_prefers_symbol_specific_model(self):
        tmpdir = make_test_dir("project_paths_model")
        try:
            models_dir = tmpdir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            expected_path = models_dir / "model_gbpusd_best.zip"
            expected_path.write_bytes(b"model")
            (models_dir / "model_eurusd_best.zip").write_bytes(b"eur")

            with patch.object(project_paths, "MODELS_DIR", models_dir), \
                patch.object(project_paths, "DEFAULT_MODEL_CANDIDATES", (models_dir / "model_eurusd_best.zip",)):
                resolved = project_paths.resolve_model_path(symbol="GBPUSD")

            self.assertEqual(expected_path, resolved)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_resolve_scaler_path_discovers_symbol_scoped_scaler_without_alias(self):
        tmpdir = make_test_dir("project_paths_scaler")
        try:
            models_dir = tmpdir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            expected_path = models_dir / "scaler_GBPUSD.pkl"
            expected_path.write_bytes(b"scaler")

            with patch.object(project_paths, "MODELS_DIR", models_dir):
                resolved = project_paths.resolve_scaler_path(required=False)

            self.assertEqual(expected_path, resolved)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
