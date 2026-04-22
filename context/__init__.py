"""
context/
---------
Deterministic, AI-free trading context layer.

This package is intentionally small and serializable:
- It never generates trade direction.
- It only provides calendar/session/blackout context and gating signals
  that can reduce/disable execution (fail closed when configured).
"""

