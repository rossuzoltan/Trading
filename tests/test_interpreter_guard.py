from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from interpreter_guard import (
    launched_script_matches,
    project_venv_python,
    should_reexec_to_project_venv,
    using_project_venv,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
VENV_PYTHON = (ROOT_DIR / ".venv" / "Scripts" / "python.exe").resolve()
VENV_PREFIX = VENV_PYTHON.parent.parent.resolve()


class InterpreterGuardTests(unittest.TestCase):
    def test_project_venv_python_discovers_repo_virtualenv(self):
        self.assertEqual(VENV_PYTHON, project_venv_python(ROOT_DIR))

    def test_guard_accepts_base_launcher_when_prefix_is_repo_venv(self):
        with patch.object(sys, "executable", str(sys.base_prefix) + "\\python.exe"):
            with patch.object(sys, "prefix", str(VENV_PREFIX)):
                self.assertTrue(using_project_venv(ROOT_DIR))
                self.assertEqual((False, VENV_PYTHON), should_reexec_to_project_venv(ROOT_DIR))

    def test_guard_requests_reexec_for_external_python(self):
        with patch.object(sys, "executable", "C:\\Python312\\python.exe"):
            with patch.object(sys, "prefix", "C:\\Python312"):
                self.assertFalse(using_project_venv(ROOT_DIR))
                self.assertEqual((True, VENV_PYTHON), should_reexec_to_project_venv(ROOT_DIR))

    def test_launched_script_matches_only_for_direct_entrypoint(self):
        with patch.object(sys, "argv", [str(ROOT_DIR / "train_agent.py")]):
            self.assertTrue(launched_script_matches(ROOT_DIR / "train_agent.py"))
        with patch.object(sys, "argv", ["-c"]):
            self.assertFalse(launched_script_matches(ROOT_DIR / "train_agent.py"))


if __name__ == "__main__":
    unittest.main()
