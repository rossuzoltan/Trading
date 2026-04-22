from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def _load_impl() -> ModuleType:
    module_path = Path(__file__).resolve().parent / "tools" / "training_status.py"
    spec = importlib.util.spec_from_file_location("_training_status_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training_status implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_IMPL = _load_impl()

sys.modules[__name__] = _IMPL


def main() -> int:
    return _IMPL.main()


if __name__ == "__main__":
    raise SystemExit(main())
