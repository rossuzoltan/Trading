"""
event_pipeline.py  –  Deprecated Compatibility Wrapper
======================================================
This file is maintained for backward compatibility. 
New code should import directly from the modular packages:
- domain.models
- domain.enums
- risk.risk_engine
- execution.broker
- execution.replay_broker
- runtime.runtime_engine

DEPRECATION NOTICE: This wrapper will be removed in a future version.
"""

# Re-exporting domain models and enums
from domain.enums import ActionType, Side
from domain.models import (
    AccountState,
    ActionSpec,
    BarBuilderState,
    BrokerPositionSnapshot,
    ConfirmedPosition,
    OrderIntent,
    SubmitResult,
    TickCursor,
    TickEvent,
    VolumeBar,
)

# Re-exporting risk engine
from risk.risk_engine import RiskEngine, RiskLimits, sync_confirmed_position

# Re-exporting execution/broker
from execution.broker import BaseBroker
from execution.replay_broker import ReplayBroker

# Re-exporting runtime engine and orchestration
from runtime.runtime_engine import (
    JsonStateStore,
    ModelPolicy,
    Mt5CursorTickSource,
    ProcessResult,
    RuntimeEngine,
    RuntimeSnapshot,
    VolumeBarBuilder,
    advance_cursor,
)

import warnings

warnings.warn(
    "Importing from 'event_pipeline' is deprecated and will be removed. "
    "Please update your imports to use the modular domain/, risk/, execution/, and runtime/ packages.",
    DeprecationWarning,
    stacklevel=2,
)
