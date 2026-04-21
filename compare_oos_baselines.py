from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import evaluate_oos


def _load_impl() -> ModuleType:
    module_path = Path(__file__).resolve().parent / "tools" / "compare_oos_baselines.py"
    spec = importlib.util.spec_from_file_location("_compare_oos_baselines_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load compare_oos_baselines implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_IMPL = _load_impl()

_IMPL._clone_context_with_segment = evaluate_oos._clone_context_with_segment
_IMPL._with_cost_stress = evaluate_oos._with_cost_stress

# Return the implementation module itself so patching helpers on the root import
# affects the function globals used inside build_baseline_comparison().
sys.modules[__name__] = _IMPL


def main() -> int:
    return _IMPL.main()


if __name__ == "__main__":
    raise SystemExit(main())
