from __future__ import annotations

import os
import sys
from pathlib import Path


_REEXEC_ENV = "PROJECT_VENV_REEXEC"


def _resolve_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path).absolute()


def project_venv_python(project_root: str | Path | None = None) -> Path | None:
    root = _resolve_path(project_root) or Path(__file__).resolve().parent
    candidates = (
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def launched_script_matches(script_path: str | Path | None) -> bool:
    if not script_path:
        return True
    argv0 = sys.argv[0] if sys.argv else ""
    if not argv0:
        return False
    launched = _resolve_path(argv0)
    expected = _resolve_path(script_path)
    return launched is not None and expected is not None and launched == expected


def using_project_venv(project_root: str | Path | None = None) -> bool:
    target = project_venv_python(project_root)
    if target is None:
        return False
    current_executable = _resolve_path(sys.executable)
    current_prefix = _resolve_path(sys.prefix)
    target_prefix = target.parent.parent.resolve()
    return current_executable == target or current_prefix == target_prefix


def should_reexec_to_project_venv(project_root: str | Path | None = None) -> tuple[bool, Path | None]:
    target = project_venv_python(project_root)
    if target is None:
        return False, None
    if using_project_venv(project_root):
        return False, target
    return True, target


def ensure_project_venv(project_root: str | Path | None = None, *, script_path: str | Path | None = None) -> None:
    if not launched_script_matches(script_path):
        return
    should_reexec, target = should_reexec_to_project_venv(project_root)
    if not should_reexec or target is None:
        return
    if os.environ.get(_REEXEC_ENV) == "1":
        return

    env = os.environ.copy()
    env[_REEXEC_ENV] = "1"
    env["VIRTUAL_ENV"] = str(target.parent.parent)
    os.execve(str(target), [str(target), *sys.argv], env)
