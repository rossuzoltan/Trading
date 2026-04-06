from __future__ import annotations

import json
import logging
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import run_logging


TEST_TMP_ROOT = Path("tests/tmp")


def make_test_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class RunLoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        run_logging.shutdown_run_logging()

    def test_configure_run_logging_captures_stdout_stderr_and_structured_context(self):
        tmpdir = make_test_dir("run_logging_capture")
        text_log_path = tmpdir / "capture.log"
        jsonl_log_path = tmpdir / "capture.jsonl"

        try:
            with patch.object(run_logging, "ensure_runtime_dirs", return_value=None):
                run_logging.configure_run_logging(
                    "train_agent",
                    symbol="EURUSD",
                    run_id="run-123",
                    text_log_path=text_log_path,
                    jsonl_log_path=jsonl_log_path,
                    capture_print=True,
                )

            print("stdout line for capture")
            sys.stderr.write("stderr line for capture\n")
            sys.stderr.flush()
            logging.getLogger("train_agent").warning("warning line", extra={"event": "unit_warning"})
            run_logging.shutdown_run_logging()

            text_output = text_log_path.read_text(encoding="utf-8")
            self.assertIn("stdout line for capture", text_output)
            self.assertIn("stderr line for capture", text_output)
            self.assertIn("warning line", text_output)
            self.assertIn("symbol=EURUSD", text_output)
            self.assertIn("run=run-123", text_output)

            records = [
                json.loads(line)
                for line in jsonl_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            stdout_records = [record for record in records if record.get("stream") == "stdout"]
            stderr_records = [record for record in records if record.get("stream") == "stderr"]
            warning_records = [record for record in records if record.get("event") == "unit_warning"]

            self.assertTrue(any(record["message"] == "stdout line for capture" for record in stdout_records))
            self.assertTrue(any(record["message"] == "stderr line for capture" for record in stderr_records))
            self.assertEqual("EURUSD", warning_records[-1]["symbol"])
            self.assertEqual("run-123", warning_records[-1]["run_id"])
        finally:
            run_logging.shutdown_run_logging()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_set_log_context_updates_future_records(self):
        tmpdir = make_test_dir("run_logging_context")
        text_log_path = tmpdir / "context.log"
        jsonl_log_path = tmpdir / "context.jsonl"

        try:
            with patch.object(run_logging, "ensure_runtime_dirs", return_value=None):
                run_logging.configure_run_logging(
                    "train_agent",
                    symbol="GBPUSD",
                    text_log_path=text_log_path,
                    jsonl_log_path=jsonl_log_path,
                    capture_print=False,
                )

            run_logging.set_log_context(run_id="run-ctx-1")
            logging.getLogger("train_agent").info("context update line", extra={"event": "context_update"})
            run_logging.shutdown_run_logging()

            records = [
                json.loads(line)
                for line in jsonl_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            target = next(record for record in records if record.get("event") == "context_update")
            self.assertEqual("GBPUSD", target["symbol"])
            self.assertEqual("run-ctx-1", target["run_id"])
        finally:
            run_logging.shutdown_run_logging()
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
