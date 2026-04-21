from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .macro_calendar import CalendarLoadResult, load_macro_calendar


def _ensure_utc_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        ts = value
    else:
        raw = str(value or "").strip()
        raw = raw.replace("Z", "+00:00")
        ts = datetime.fromisoformat(raw)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _symbol_currencies(symbol: str) -> tuple[str, ...]:
    sym = str(symbol or "").strip().upper()
    if len(sym) >= 6:
        base = sym[:3]
        quote = sym[3:6]
        if base.isalpha() and quote.isalpha():
            return (base, quote)
    return tuple()


@dataclass(frozen=True)
class HighRiskWindow:
    start_utc: str
    end_utc: str
    kind: str
    event_id: str


@dataclass(frozen=True)
class DailyContext:
    symbol: str
    date_utc: str
    day_type: str  # normal|macro_day|unknown
    event_risk: str  # none|elevated|high
    allowed_setups: list[str]
    blocked_setups: list[str]
    blocked_hours_utc: list[int]
    high_risk_windows: list[dict[str, Any]]
    aggressiveness_mode: str  # normal|reduced|blocked
    context_reasoning: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "date_utc": self.date_utc,
            "day_type": self.day_type,
            "event_risk": self.event_risk,
            "allowed_setups": list(self.allowed_setups),
            "blocked_setups": list(self.blocked_setups),
            "blocked_hours_utc": list(self.blocked_hours_utc),
            "high_risk_windows": list(self.high_risk_windows),
            "aggressiveness_mode": self.aggressiveness_mode,
            "context_reasoning": list(self.context_reasoning),
        }


@dataclass(frozen=True)
class ContextSlice:
    in_blackout: bool
    blackout_kind: str | None
    active_event_id: str | None
    effective_aggressiveness_mode: str  # normal|reduced|blocked
    effective_block_policy: str  # none|block_entry|close_only_on_reversal
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_blackout": bool(self.in_blackout),
            "blackout_kind": self.blackout_kind,
            "active_event_id": self.active_event_id,
            "effective_aggressiveness_mode": self.effective_aggressiveness_mode,
            "effective_block_policy": self.effective_block_policy,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class ContextGateVerdict:
    daily: DailyContext
    slice: ContextSlice
    transformed_signal: int
    reason_override: str | None = None


class ContextGate:
    """
    Deterministic context gate that can only reduce execution:
    - It never creates a direction signal.
    - It may block new entries or downgrade reversals to close-only.
    """

    def __init__(
        self,
        *,
        symbol: str,
        rule_family: str,
        manifest_dir: Path,
        runtime_constraints: dict[str, Any],
    ) -> None:
        self.symbol = str(symbol or "").strip().upper()
        self.rule_family = str(rule_family or "").strip()
        self.manifest_dir = Path(manifest_dir)
        self.runtime_constraints = dict(runtime_constraints or {})

        context_cfg = dict((self.runtime_constraints.get("context") or {}) if isinstance(self.runtime_constraints, dict) else {})
        self.enabled = bool(context_cfg.get("enabled", False))
        self.fail_closed = bool(context_cfg.get("fail_closed_on_calendar_error", True))
        self.blackout_before_min = int(context_cfg.get("tier1_blackout_minutes_before", 30) or 30)
        self.blackout_after_min = int(context_cfg.get("tier1_blackout_minutes_after", 30) or 30)
        self.macro_day_blocked_setups = [
            str(item).strip()
            for item in list(context_cfg.get("macro_day_blocked_setups", []) or [])
            if str(item).strip()
        ]

        calendar_path_raw = str(context_cfg.get("calendar_path", "") or "").strip()
        self.calendar_path = (self.manifest_dir / calendar_path_raw) if calendar_path_raw else None
        self.calendar_expected_sha256 = str(context_cfg.get("calendar_sha256", "") or "").strip() or None
        self._calendar_result: CalendarLoadResult | None = None

    def _load_calendar(self) -> CalendarLoadResult:
        if self._calendar_result is not None:
            return self._calendar_result
        if not self.enabled or self.calendar_path is None:
            self._calendar_result = CalendarLoadResult(calendar=None, sha256=None, error="context_disabled")
            return self._calendar_result
        self._calendar_result = load_macro_calendar(self.calendar_path, expected_sha256=self.calendar_expected_sha256)
        return self._calendar_result

    def _rollover_hours(self) -> list[int]:
        hours = list(self.runtime_constraints.get("rollover_block_utc_hours", []) or [])
        normalized: list[int] = []
        for h in hours:
            try:
                ih = int(h)
            except Exception:
                continue
            if 0 <= ih <= 23:
                normalized.append(ih)
        return sorted(set(normalized))

    def _tier1_blackout_windows(self, *, date_utc: str) -> list[tuple[datetime, datetime, str, str]]:
        result = self._load_calendar()
        if result.calendar is None:
            return []
        currencies = _symbol_currencies(self.symbol)
        events = result.calendar.events_for_day(date_utc=date_utc, currencies=currencies, min_tier=1)
        windows: list[tuple[datetime, datetime, str, str]] = []
        for ev in events:
            if int(ev.tier) != 1:
                continue
            start = ev.timestamp - timedelta(minutes=self.blackout_before_min)
            end = ev.timestamp + timedelta(minutes=self.blackout_after_min)
            windows.append((start, end, "tier1_event", ev.event_id))
        windows.sort(key=lambda item: (item[0], item[3]))
        return windows

    def _rollover_windows(self, *, day: datetime) -> list[tuple[datetime, datetime, str, str]]:
        windows: list[tuple[datetime, datetime, str, str]] = []
        rollover_hours = self._rollover_hours()
        for hour in rollover_hours:
            start = datetime(day.year, day.month, day.day, hour, 0, 0, tzinfo=timezone.utc)
            end = start + timedelta(hours=1)
            windows.append((start, end, "rollover", "ROLLOVER"))
        return windows

    def _build_daily_context(self, *, bar_time_utc: datetime) -> DailyContext:
        date_key = bar_time_utc.date().isoformat()
        currencies = _symbol_currencies(self.symbol)
        cal = self._load_calendar()
        reason_codes: list[str] = []
        day_type = "unknown" if self.enabled else "unknown"
        macro_day = False
        tier1_events = []
        if self.enabled and cal.calendar is not None and currencies:
            tier1_events = [
                ev
                for ev in cal.calendar.events_for_day(date_utc=date_key, currencies=currencies, min_tier=1)
                if int(ev.tier) == 1
            ]
            macro_day = bool(tier1_events)
            day_type = "macro_day" if macro_day else "normal"
        elif self.enabled and cal.error and self.fail_closed:
            reason_codes.append(f"CONTEXT:CALENDAR_ERROR:{cal.error}")

        if macro_day:
            reason_codes.append("CONTEXT:MACRO_DAY:TIER1")

        blocked_hours = self._rollover_hours()
        windows: list[dict[str, Any]] = []
        for start, end, kind, event_id in self._tier1_blackout_windows(date_utc=date_key):
            windows.append(
                {
                    "start_utc": start.isoformat(),
                    "end_utc": end.isoformat(),
                    "kind": kind,
                    "event_id": event_id,
                }
            )
        for start, end, kind, event_id in self._rollover_windows(day=bar_time_utc):
            windows.append(
                {
                    "start_utc": start.isoformat(),
                    "end_utc": end.isoformat(),
                    "kind": kind,
                    "event_id": event_id,
                }
            )

        blocked_setups: list[str] = []
        if macro_day and self.rule_family and self.rule_family in set(self.macro_day_blocked_setups):
            blocked_setups.append(self.rule_family)
            reason_codes.append(f"CONTEXT:BLOCK_SETUP:MACRO_DAY:{self.rule_family}")

        allowed_setups = [self.rule_family] if self.rule_family else []

        aggressiveness_mode = "normal"
        event_risk = "none"
        if day_type == "macro_day":
            aggressiveness_mode = "reduced"
            event_risk = "elevated"
        elif day_type == "unknown":
            aggressiveness_mode = "normal"
            event_risk = "none"

        reason_codes = sorted(set(reason_codes))
        return DailyContext(
            symbol=self.symbol,
            date_utc=date_key,
            day_type=day_type,
            event_risk=event_risk,
            allowed_setups=allowed_setups,
            blocked_setups=blocked_setups,
            blocked_hours_utc=blocked_hours,
            high_risk_windows=windows,
            aggressiveness_mode=aggressiveness_mode,
            context_reasoning=reason_codes,
        )

    def _build_slice(self, *, daily: DailyContext, bar_time_utc: datetime) -> ContextSlice:
        reason_codes: list[str] = []
        in_blackout = False
        blackout_kind: str | None = None
        active_event_id: str | None = None

        # Rollover hour is a deterministic structural blackout.
        rollover_hours = set(daily.blocked_hours_utc or [])
        if int(bar_time_utc.hour) in rollover_hours:
            in_blackout = True
            blackout_kind = "rollover"
            active_event_id = "ROLLOVER"
            reason_codes.append("CONTEXT:BLACKOUT:ROLLOVER")

        # Tier-1 blackout windows.
        for window in daily.high_risk_windows:
            if str(window.get("kind")) != "tier1_event":
                continue
            try:
                start = _ensure_utc_datetime(window.get("start_utc"))
                end = _ensure_utc_datetime(window.get("end_utc"))
            except Exception:
                continue
            if start <= bar_time_utc <= end:
                in_blackout = True
                blackout_kind = "tier1_event"
                active_event_id = str(window.get("event_id") or "") or None
                if active_event_id:
                    reason_codes.append(f"CONTEXT:BLACKOUT:TIER1:{active_event_id}")
                else:
                    reason_codes.append("CONTEXT:BLACKOUT:TIER1")
                break

        # Calendar errors can trigger fail-closed semantics.
        cal = self._load_calendar()
        if self.enabled and self.fail_closed and cal.error and cal.error != "context_disabled":
            in_blackout = True
            blackout_kind = blackout_kind or "calendar_error"
            reason_codes.append(f"CONTEXT:CALENDAR_ERROR:{cal.error}")

        effective_aggressiveness = daily.aggressiveness_mode
        if in_blackout:
            effective_aggressiveness = "blocked"

        reason_codes = sorted(set(reason_codes + list(daily.context_reasoning or [])))
        return ContextSlice(
            in_blackout=in_blackout,
            blackout_kind=blackout_kind,
            active_event_id=active_event_id,
            effective_aggressiveness_mode=effective_aggressiveness,
            effective_block_policy="none",
            reason_codes=reason_codes,
        )

    def evaluate(
        self,
        *,
        bar_time_utc: Any,
        signal: int,
        current_direction: int,
    ) -> ContextGateVerdict:
        ts = _ensure_utc_datetime(bar_time_utc)
        daily = self._build_daily_context(bar_time_utc=ts)
        slice_ctx = self._build_slice(daily=daily, bar_time_utc=ts)

        transformed = int(signal or 0)
        reason_override: str | None = None

        # Setup-level day block.
        setup_blocked_today = bool(self.rule_family and self.rule_family in set(daily.blocked_setups or []))

        # Blackout semantics: block new entries + block reversal opens (close-only),
        # but never force-liquidate.
        block_new_exposure = bool(slice_ctx.in_blackout or setup_blocked_today)

        effective_policy = "none"
        if block_new_exposure:
            if int(current_direction or 0) == 0:
                if transformed != 0:
                    transformed = 0
                    effective_policy = "block_entry"
                    reason_override = "context blocked entry"
            else:
                cur = int(current_direction or 0)
                if transformed != 0 and transformed != cur:
                    transformed = 0
                    effective_policy = "close_only_on_reversal"
                    reason_override = "context close-only reversal"

        slice_ctx = ContextSlice(
            in_blackout=slice_ctx.in_blackout,
            blackout_kind=slice_ctx.blackout_kind,
            active_event_id=slice_ctx.active_event_id,
            effective_aggressiveness_mode=slice_ctx.effective_aggressiveness_mode,
            effective_block_policy=effective_policy,
            reason_codes=slice_ctx.reason_codes,
        )
        return ContextGateVerdict(daily=daily, slice=slice_ctx, transformed_signal=transformed, reason_override=reason_override)

