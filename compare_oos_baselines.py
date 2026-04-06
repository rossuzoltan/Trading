from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate_oos import (
    _evaluate_policy as _shared_evaluate_policy,
    _flat_provider as _shared_flat_provider,
    _with_cost_stress as _shared_with_cost_stress,
    _clone_context_with_segment as _shared_clone_context_with_segment,
)


_TARGET = Path(__file__).resolve().parent / "tools" / "compare_oos_baselines.py"
__file__ = str(_TARGET)
exec(compile(_TARGET.read_text(encoding="utf-8"), str(_TARGET), "exec"), globals())


def _flat_provider(**_: object) -> int:
    return _shared_flat_provider(**_)


def _clone_context_with_segment(context, segment: pd.DataFrame):
    return _shared_clone_context_with_segment(context, segment)


def _with_cost_stress(context, *, slippage_multiplier: float):
    return _shared_with_cost_stress(context, slippage_multiplier=slippage_multiplier)


def _evaluate_policy(*, replay_context, action_index_provider):
    return _shared_evaluate_policy(replay_context=replay_context, action_index_provider=action_index_provider)
