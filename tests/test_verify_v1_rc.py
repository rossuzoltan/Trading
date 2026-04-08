from __future__ import annotations

import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
