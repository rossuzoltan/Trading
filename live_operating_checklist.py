from __future__ import annotations

from pathlib import Path


_TARGET = Path(__file__).resolve().parent / "tools" / "live_operating_checklist.py"
__file__ = str(_TARGET)
exec(compile(_TARGET.read_text(encoding="utf-8"), str(_TARGET), "exec"), globals())
