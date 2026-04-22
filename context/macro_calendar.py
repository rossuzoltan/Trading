from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        ts = value
    else:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("Missing timestamp_utc.")
        # Accept trailing "Z" as UTC.
        raw = raw.replace("Z", "+00:00")
        ts = datetime.fromisoformat(raw)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


@dataclass(frozen=True)
class MacroEvent:
    event_id: str
    timestamp_utc: str
    currency: str
    tier: int
    title: str = ""
    kind: str = "unknown"

    @property
    def timestamp(self) -> datetime:
        return _parse_utc_timestamp(self.timestamp_utc)

    @property
    def date_utc(self) -> str:
        return self.timestamp.date().isoformat()


@dataclass(frozen=True)
class MacroCalendar:
    schema_version: int
    events: tuple[MacroEvent, ...]
    source: str = "operator"
    generated_at_utc: str | None = None
    notes: str | None = None

    def events_for_day(self, *, date_utc: str, currencies: Iterable[str], min_tier: int = 1) -> list[MacroEvent]:
        wanted = {str(c).strip().upper() for c in currencies if str(c).strip()}
        out: list[MacroEvent] = []
        for ev in self.events:
            if ev.tier < int(min_tier):
                continue
            if ev.currency.upper() not in wanted:
                continue
            if ev.date_utc != date_utc:
                continue
            out.append(ev)
        out.sort(key=lambda e: e.timestamp)
        return out


@dataclass(frozen=True)
class CalendarLoadResult:
    calendar: MacroCalendar | None
    sha256: str | None
    error: str | None


def load_macro_calendar(path: str | Path, *, expected_sha256: str | None = None) -> CalendarLoadResult:
    resolved = Path(path)
    if not resolved.exists():
        return CalendarLoadResult(calendar=None, sha256=None, error=f"calendar_missing:{resolved}")

    try:
        sha256 = _file_sha256(resolved)
    except Exception as exc:  # pragma: no cover
        return CalendarLoadResult(calendar=None, sha256=None, error=f"calendar_sha_error:{type(exc).__name__}:{exc}")

    if expected_sha256:
        expected = str(expected_sha256).strip().lower()
        if expected and sha256.lower() != expected:
            return CalendarLoadResult(calendar=None, sha256=sha256, error="calendar_sha_mismatch")

    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        return CalendarLoadResult(calendar=None, sha256=sha256, error=f"calendar_invalid_json:{type(exc).__name__}")

    if not isinstance(raw, dict):
        return CalendarLoadResult(calendar=None, sha256=sha256, error="calendar_invalid_root")

    schema_version = int(raw.get("schema_version", 0) or 0)
    if schema_version != 1:
        return CalendarLoadResult(calendar=None, sha256=sha256, error=f"calendar_schema_version:{schema_version}")

    events_raw = raw.get("events", [])
    if events_raw is None:
        events_raw = []
    if not isinstance(events_raw, list):
        return CalendarLoadResult(calendar=None, sha256=sha256, error="calendar_events_not_list")

    events: list[MacroEvent] = []
    for idx, item in enumerate(events_raw):
        if not isinstance(item, dict):
            return CalendarLoadResult(calendar=None, sha256=sha256, error=f"calendar_event_not_object:{idx}")
        event_id = str(item.get("event_id", "") or "").strip()
        timestamp_utc = str(item.get("timestamp_utc", "") or "").strip()
        currency = str(item.get("currency", "") or "").strip().upper()
        tier = int(item.get("tier", 0) or 0)
        title = str(item.get("title", "") or "")
        kind = str(item.get("kind", "") or "unknown")
        if not event_id:
            return CalendarLoadResult(calendar=None, sha256=sha256, error=f"calendar_event_missing_id:{idx}")
        if not currency or len(currency) != 3:
            return CalendarLoadResult(calendar=None, sha256=sha256, error=f"calendar_event_bad_currency:{idx}")
        if tier <= 0:
            return CalendarLoadResult(calendar=None, sha256=sha256, error=f"calendar_event_bad_tier:{idx}")
        try:
            _ = _parse_utc_timestamp(timestamp_utc)
        except Exception:
            return CalendarLoadResult(calendar=None, sha256=sha256, error=f"calendar_event_bad_timestamp:{idx}")
        events.append(
            MacroEvent(
                event_id=event_id,
                timestamp_utc=timestamp_utc,
                currency=currency,
                tier=tier,
                title=title,
                kind=kind,
            )
        )

    events.sort(key=lambda e: (e.timestamp, e.event_id))
    calendar = MacroCalendar(
        schema_version=schema_version,
        events=tuple(events),
        source=str(raw.get("source", "operator") or "operator"),
        generated_at_utc=(str(raw.get("generated_at_utc")) if raw.get("generated_at_utc") else None),
        notes=(str(raw.get("notes")) if raw.get("notes") else None),
    )
    return CalendarLoadResult(calendar=calendar, sha256=sha256, error=None)


def render_default_calendar_template(*, generated_at_utc: str | None = None) -> str:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source": "operator",
        "generated_at_utc": generated_at_utc,
        "notes": (
            "Deterministic macro calendar for context gating. "
            "Keep it under version control inside the RC pack. "
            "Add tier-1 events with UTC timestamps (no scraping in-bot)."
        ),
        "events": [],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"

