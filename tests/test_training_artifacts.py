from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from train_agent import (
    TrainingDiagnosticsCallback,
    _archive_paths,
    _build_promoted_training_diagnostics,
    _clear_legacy_checkpoint_artifacts,
)


TEST_TMP_ROOT = Path("tests/tmp")


def make_test_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class TrainingArtifactTests(unittest.TestCase):
    def test_promoted_diagnostics_always_include_run_metadata(self):
        diagnostics = {
            "train_bars": 5000,
            "holdout_bars": 500,
            "point_in_time_verified": True,
            "dataset_integrity_verified": True,
        }
        payload = _build_promoted_training_diagnostics(
            diagnostics,
            run_id="run-123",
            artifact_candidate_selected=False,
            artifact_candidate_reason="Economics gate failed.",
        )
        self.assertEqual("run-123", payload["run_id"])
        self.assertFalse(payload["artifact_candidate_selected"])
        self.assertEqual("Economics gate failed.", payload["artifact_candidate_reason"])
        self.assertEqual(2000, payload["bar_construction_ticks_per_bar"])
        self.assertEqual(2000, payload["ticks_per_bar"])

    def test_archive_paths_moves_stale_files_under_archive_root(self):
        tmpdir = make_test_dir("train_artifacts")
        archive_root = tmpdir / "archive"
        created_paths = [
            tmpdir / "model.zip",
            tmpdir / "vecnormalize.pkl",
            tmpdir / "manifest.json",
            tmpdir / "diagnostics.json",
        ]
        for path in created_paths:
            path.write_text("stale", encoding="utf-8")

        try:
            archived = _archive_paths(created_paths, archive_root=archive_root)
            self.assertGreaterEqual(len(archived), len(created_paths))
            for path in created_paths:
                self.assertFalse(path.exists())
            self.assertTrue(archive_root.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_clear_legacy_checkpoint_artifacts_archives_fold_dirs(self):
        tmpdir = make_test_dir("legacy_checkpoint_archive")
        checkpoints_dir = tmpdir / "checkpoints"
        legacy_fold = checkpoints_dir / "fold_legacy_test"
        legacy_file = checkpoints_dir / "best_model.zip"
        legacy_fold.mkdir(parents=True, exist_ok=True)
        (legacy_fold / "training_heartbeat.json").write_text("{}", encoding="utf-8")
        legacy_file.write_text("old-model", encoding="utf-8")

        try:
            archived = _clear_legacy_checkpoint_artifacts(run_id="test-run", checkpoints_root=checkpoints_dir)
            self.assertTrue(any("fold_legacy_test" in item for item in archived))
            self.assertTrue(any("best_model.zip" in item for item in archived))
            self.assertFalse(legacy_fold.exists())
            self.assertFalse(legacy_file.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_training_diagnostics_summary_is_json_serializable(self):
        callback = TrainingDiagnosticsCallback(verbose=0)
        callback.metrics["train/approx_kl"] = [0.02]
        callback.metrics["train/explained_variance"] = [0.35]
        callback.metrics["train/value_loss"] = [1.0, 1.1, 0.9]

        summary = callback.summary()

        self.assertIsInstance(summary["value_loss_stable"], bool)
        json.dumps(summary)

    def test_training_diagnostics_records_each_logger_update_once(self):
        callback = TrainingDiagnosticsCallback(verbose=0)
        callback.model = SimpleNamespace(
            logger=SimpleNamespace(
                name_to_value={
                    "train/approx_kl": 0.02,
                    "train/explained_variance": 0.15,
                    "train/value_loss": 1.5,
                    "train/n_updates": 10,
                }
            )
        )

        callback._on_step()
        callback._on_step()

        self.assertEqual([0.02], callback.metrics["train/approx_kl"])
        self.assertEqual([0.15], callback.metrics["train/explained_variance"])
        self.assertEqual([1.5], callback.metrics["train/value_loss"])


if __name__ == "__main__":
    unittest.main()
