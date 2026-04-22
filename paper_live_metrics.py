from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from project_paths import gate_artifact_dir, shadow_artifact_dir


MIN_PROMOTION_TRADING_DAYS = 20
MIN_PROMOTION_ACTIONABLE_EVENTS = 30
MIN_WEEKLY_REVIEW_TRADING_DAYS = 5
MIN_WEEKLY_REVIEW_ACTIONABLE_EVENTS = 10
MIN_DRIFT_EVENT_COUNT = 50

NORMAL_SIGNAL_DENSITY_RATIO_RANGE = (0.75, 1.25)
CRITICAL_SIGNAL_DENSITY_RATIO_RANGE = (0.60, 1.40)
NORMAL_WOULD_OPEN_RATIO_RANGE = (0.70, 1.30)
CRITICAL_WOULD_OPEN_RATIO_RANGE = (0.60, 1.40)
NORMAL_SPREAD_REJECTION_DELTA_PP = 10.0
CRITICAL_REJECTION_DELTA_PP = 20.0
NORMAL_SESSION_REJECTION_DELTA_PP = 10.0
NORMAL_DIRECTIONAL_OCCUPANCY_DELTA_PP = 15.0
CRITICAL_DIRECTIONAL_OCCUPANCY_DELTA_PP = 20.0


@dataclass(frozen=True)
class ShadowEvidencePaths:
    root_dir: Path
    events_path: Path
    summary_json_path: Path
    summary_markdown_path: Path


@dataclass(frozen=True)
class PaperLiveGatePaths:
    root_dir: Path
    json_path: Path
    markdown_path: Path


def resolve_shadow_evidence_paths(
    *,
    symbol: str,
    manifest_hash: str,
    base_dir: str | Path | None = None,
) -> ShadowEvidencePaths:
    root_dir = shadow_artifact_dir(symbol, manifest_hash, base_dir=base_dir)
    return ShadowEvidencePaths(
        root_dir=root_dir,
        events_path=root_dir / "events.jsonl",
        summary_json_path=root_dir / "shadow_summary.json",
        summary_markdown_path=root_dir / "shadow_summary.md",
    )


def resolve_paper_live_gate_paths(
    *,
    symbol: str,
    manifest_hash: str,
    base_dir: str | Path | None = None,
) -> PaperLiveGatePaths:
    root_dir = gate_artifact_dir(symbol, manifest_hash, base_dir=base_dir)
    return PaperLiveGatePaths(
        root_dir=root_dir,
        json_path=root_dir / "paper_live_gate.json",
        markdown_path=root_dir / "paper_live_gate.md",
    )


def _safe_iso(value: Any) -> str | None:
    if value in (None, ""):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _safe_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return rows
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_shadow_events(path: str | Path) -> list[dict[str, Any]]:
    return _load_jsonl(path)


def _event_timestamp(row: dict[str, Any]) -> pd.Timestamp:
    raw_value = row.get("timestamp_utc") or row.get("bar_ts")
    ts = pd.Timestamp(raw_value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _normalize_event(row: dict[str, Any]) -> dict[str, Any]:
    timestamp = _event_timestamp(row)
    signal = _safe_int(row.get("signal_direction", row.get("signal", 0)))
    action_state = str(row.get("action_state", "") or "").strip().lower()
    would_open = _safe_bool(row.get("would_open"))
    would_close = _safe_bool(row.get("would_close"))
    would_hold = _safe_bool(row.get("would_hold", row.get("would_hold_position", False)))
    no_trade_reason = str(row.get("no_trade_reason", row.get("reason", "")) or "")
    active_position_state = str(
        row.get("position_state", row.get("active_position_state", "flat")) or "flat"
    ).strip().lower()
    context_day_type = row.get("context_day_type")
    context_event_risk = row.get("context_event_risk")
    context_in_blackout = _safe_bool(row.get("context_in_blackout", False))
    context_blackout_kind = row.get("context_blackout_kind")
    context_active_event_id = row.get("context_active_event_id")
    context_aggressiveness_mode = row.get("context_aggressiveness_mode")
    context_block_policy = row.get("context_block_policy")
    context_reason_codes = row.get("context_reason_codes")
    return {
        "timestamp_utc": timestamp.isoformat(),
        "timestamp": timestamp,
        "symbol": str(row.get("symbol", "") or "").upper(),
        "ticks_per_bar": _safe_int(row.get("ticks_per_bar", 0)),
        "manifest_hash": str(row.get("manifest_hash", row.get("manifest_fingerprint", "")) or ""),
        "logic_hash": str(row.get("logic_hash", "") or ""),
        "evaluator_hash": str(row.get("evaluator_hash", "") or ""),
        "signal_direction": signal,
        "signal_present": signal != 0,
        "action_state": action_state,
        "would_open": would_open,
        "would_close": would_close,
        "would_hold": would_hold,
        "would_hold": _safe_bool(row.get("would_hold", row.get("would_hold_position"))),
        "no_trade_reason": no_trade_reason,
        "spread_pips": _safe_float(row.get("spread_pips", row.get("spread", 0.0))),
        "session_filter_pass": _safe_bool(row.get("session_filter_pass", row.get("session_ok"))),
        "risk_filter_pass": _safe_bool(row.get("risk_filter_pass", row.get("risk_ok"))),
        "spread_ok": _safe_bool(row.get("spread_ok", True)),
        "position_state": active_position_state,
        "context_day_type": str(context_day_type) if context_day_type is not None else None,
        "context_event_risk": str(context_event_risk) if context_event_risk is not None else None,
        "context_in_blackout": bool(context_in_blackout),
        "context_blackout_kind": str(context_blackout_kind) if context_blackout_kind is not None else None,
        "context_active_event_id": str(context_active_event_id) if context_active_event_id is not None else None,
        "context_aggressiveness_mode": str(context_aggressiveness_mode) if context_aggressiveness_mode is not None else None,
        "context_block_policy": str(context_block_policy) if context_block_policy is not None else None,
        "context_reason_codes": list(context_reason_codes) if isinstance(context_reason_codes, list) else None,
    }


def summarize_shadow_events(
    events: list[dict[str, Any]],
    *,
    min_trading_days: int = MIN_PROMOTION_TRADING_DAYS,
    min_actionable_events: int = MIN_PROMOTION_ACTIONABLE_EVENTS,
) -> dict[str, Any]:
    normalized = [_normalize_event(row) for row in events]
    if not normalized:
        return {
            "event_count": 0,
            "trading_days": 0,
            "actionable_event_count": 0,
            "evidence_sufficient": False,
            "shadow_window_start": None,
            "shadow_window_end": None,
            "no_trade_reason_counts": {},
            "directional_occupancy": {"flat": 0.0, "long": 0.0, "short": 0.0},
            "counts": {},
        }

    timestamps = [row["timestamp"] for row in normalized]
    trading_days = sorted({ts.date().isoformat() for ts in timestamps})
    would_open_count = sum(1 for row in normalized if row["would_open"])
    would_close_count = sum(1 for row in normalized if row["would_close"])
    would_hold_count = sum(1 for row in normalized if row["would_hold"])
    signal_count = sum(1 for row in normalized if row["signal_present"])
    long_signal_count = sum(1 for row in normalized if row["signal_direction"] > 0)
    short_signal_count = sum(1 for row in normalized if row["signal_direction"] < 0)
    long_open_count = sum(1 for row in normalized if row["would_open"] and row["signal_direction"] > 0)
    short_open_count = sum(1 for row in normalized if row["would_open"] and row["signal_direction"] < 0)
    spread_rejection_count = sum(1 for row in normalized if not row["spread_ok"])
    session_rejection_count = sum(1 for row in normalized if not row["session_filter_pass"])
    risk_rejection_count = sum(1 for row in normalized if not row["risk_filter_pass"])
    actionable_event_count = would_open_count + would_close_count

    context_macro_day_count = sum(1 for row in normalized if row.get("context_day_type") == "macro_day")
    context_blackout_count = sum(1 for row in normalized if bool(row.get("context_in_blackout", False)))
    context_block_entry_count = sum(1 for row in normalized if row.get("context_block_policy") == "block_entry")
    context_close_only_reversal_count = sum(
        1 for row in normalized if row.get("context_block_policy") == "close_only_on_reversal"
    )

    no_trade_reason_counts: dict[str, int] = {}
    occupancy_counts = {"flat": 0, "long": 0, "short": 0}
    for row in normalized:
        reason = row["no_trade_reason"].strip().lower()
        if reason:
            no_trade_reason_counts[reason] = no_trade_reason_counts.get(reason, 0) + 1
        occupancy_state = row["position_state"] if row["position_state"] in occupancy_counts else "flat"
        occupancy_counts[occupancy_state] += 1

    event_count = len(normalized)
    occupancy = {
        state: float(count / event_count) if event_count else 0.0
        for state, count in occupancy_counts.items()
    }

    summary = {
        "symbol": normalized[-1]["symbol"],
        "ticks_per_bar": normalized[-1]["ticks_per_bar"],
        "manifest_hash": normalized[-1]["manifest_hash"],
        "logic_hash": normalized[-1]["logic_hash"],
        "evaluator_hash": normalized[-1]["evaluator_hash"],
        "event_count": event_count,
        "trading_days": len(trading_days),
        "trading_day_list": trading_days,
        "actionable_event_count": actionable_event_count,
        "shadow_window_start": timestamps[0].isoformat(),
        "shadow_window_end": timestamps[-1].isoformat(),
        "counts": {
            "signal_count": signal_count,
            "would_open_count": would_open_count,
            "would_close_count": would_close_count,
            "would_hold_count": would_hold_count,
            "long_signal_count": long_signal_count,
            "short_signal_count": short_signal_count,
            "long_open_count": long_open_count,
            "short_open_count": short_open_count,
            "spread_rejection_count": spread_rejection_count,
            "session_rejection_count": session_rejection_count,
            "risk_rejection_count": risk_rejection_count,
            "context_macro_day_count": context_macro_day_count,
            "context_blackout_count": context_blackout_count,
            "context_block_entry_count": context_block_entry_count,
            "context_close_only_reversal_count": context_close_only_reversal_count,
        },
        "rates": {
            "signal_density": float(signal_count / event_count),
            "would_open_density": float(would_open_count / event_count),
            "spread_rejection_pct": float(spread_rejection_count * 100.0 / event_count),
            "session_rejection_pct": float(session_rejection_count * 100.0 / event_count),
            "risk_rejection_pct": float(risk_rejection_count * 100.0 / event_count),
            "long_open_share": float(long_open_count / would_open_count) if would_open_count else 0.0,
            "context_blackout_pct": float(context_blackout_count * 100.0 / event_count),
        },
        "directional_occupancy": occupancy,
        "no_trade_reason_counts": no_trade_reason_counts,
        "evidence_sufficient": len(trading_days) >= int(min_trading_days) and actionable_event_count >= int(min_actionable_events),
        "evidence_requirements": {
            "min_trading_days": int(min_trading_days),
            "min_actionable_events": int(min_actionable_events),
        },
        "evidence_shortfall": {
            "trading_days_remaining": int(max(int(min_trading_days) - len(trading_days), 0)),
            "actionable_events_remaining": int(max(int(min_actionable_events) - actionable_event_count, 0)),
        },
    }
    return summary


def _range_pass(value: float | None, lower: float, upper: float) -> bool:
    return value is not None and lower <= value <= upper


def _delta_pass(value: float | None, limit: float) -> bool:
    return value is not None and value <= limit


def compute_drift_metrics(
    summary: dict[str, Any],
    *,
    replay_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Drift checks are not meaningful on tiny shadow samples; keep the verdict explicit
    # so operators don't chase noise from a handful of bars.
    has_event_count = "event_count" in summary
    event_count = _safe_int(summary.get("event_count"), 0) if has_event_count else None
    insufficient_samples = bool(has_event_count and event_count is not None and 0 < event_count < MIN_DRIFT_EVENT_COUNT)
    no_data = bool(has_event_count and (event_count is None or event_count <= 0))

    replay_reference = dict(replay_reference or {})
    shadow_rates = dict(summary.get("rates", {}) or {})
    shadow_occupancy = dict(summary.get("directional_occupancy", {}) or {})
    expected_signal_density = replay_reference.get("signal_density")
    expected_would_open_density = replay_reference.get("would_open_density")
    expected_spread_rejection_pct = float(replay_reference.get("spread_rejection_pct", 0.0) or 0.0)
    expected_session_rejection_pct = float(replay_reference.get("session_rejection_pct", 0.0) or 0.0)
    expected_long_share = replay_reference.get("long_share")

    signal_density_ratio = None
    if expected_signal_density not in (None, 0):
        signal_density_ratio = float(shadow_rates.get("signal_density", 0.0) / float(expected_signal_density))
    would_open_density_ratio = None
    if expected_would_open_density not in (None, 0):
        would_open_density_ratio = float(shadow_rates.get("would_open_density", 0.0) / float(expected_would_open_density))

    spread_rejection_delta_pp = abs(float(shadow_rates.get("spread_rejection_pct", 0.0)) - expected_spread_rejection_pct)
    session_rejection_delta_pp = abs(float(shadow_rates.get("session_rejection_pct", 0.0)) - expected_session_rejection_pct)
    directional_occupancy_delta_pp = None
    if expected_long_share is not None:
        directional_occupancy_delta_pp = abs(float(shadow_occupancy.get("long", 0.0)) - float(expected_long_share)) * 100.0

    normal_failures: list[str] = []
    critical_failures: list[str] = []
    if signal_density_ratio is not None:
        if not _range_pass(signal_density_ratio, *NORMAL_SIGNAL_DENSITY_RATIO_RANGE):
            normal_failures.append("signal_density_ratio")
        if not _range_pass(signal_density_ratio, *CRITICAL_SIGNAL_DENSITY_RATIO_RANGE):
            critical_failures.append("signal_density_ratio")
    if would_open_density_ratio is not None:
        if not _range_pass(would_open_density_ratio, *NORMAL_WOULD_OPEN_RATIO_RANGE):
            normal_failures.append("would_open_density_ratio")
        if not _range_pass(would_open_density_ratio, *CRITICAL_WOULD_OPEN_RATIO_RANGE):
            critical_failures.append("would_open_density_ratio")
    if not _delta_pass(spread_rejection_delta_pp, NORMAL_SPREAD_REJECTION_DELTA_PP):
        normal_failures.append("spread_rejection_delta_pp")
    if not _delta_pass(spread_rejection_delta_pp, CRITICAL_REJECTION_DELTA_PP):
        critical_failures.append("spread_rejection_delta_pp")
    if not _delta_pass(session_rejection_delta_pp, NORMAL_SESSION_REJECTION_DELTA_PP):
        normal_failures.append("session_rejection_delta_pp")
    if not _delta_pass(session_rejection_delta_pp, CRITICAL_REJECTION_DELTA_PP):
        critical_failures.append("session_rejection_delta_pp")
    if directional_occupancy_delta_pp is not None:
        if not _delta_pass(directional_occupancy_delta_pp, NORMAL_DIRECTIONAL_OCCUPANCY_DELTA_PP):
            normal_failures.append("directional_occupancy_delta_pp")
        if not _delta_pass(directional_occupancy_delta_pp, CRITICAL_DIRECTIONAL_OCCUPANCY_DELTA_PP):
            critical_failures.append("directional_occupancy_delta_pp")

    is_critical = bool(critical_failures or len(set(normal_failures)) >= 2)
    verdict = "critical" if is_critical else "watch" if normal_failures else "aligned"
    if no_data:
        verdict = "no_data"
        is_critical = False
    elif insufficient_samples:
        verdict = "insufficient"
        is_critical = False

    return {
        "reference_source": replay_reference.get("reference_source", "shadow_only"),
        "event_count": event_count,
        "min_event_count": int(MIN_DRIFT_EVENT_COUNT) if has_event_count else None,
        "insufficient_samples": insufficient_samples if has_event_count else None,
        "signal_density_ratio": signal_density_ratio,
        "would_open_density_ratio": would_open_density_ratio,
        "spread_rejection_delta_pp": spread_rejection_delta_pp,
        "session_rejection_delta_pp": session_rejection_delta_pp,
        "directional_occupancy_delta_pp": directional_occupancy_delta_pp,
        "normal_failures": sorted(set(normal_failures)),
        "critical_failures": sorted(set(critical_failures)),
        "critical": is_critical,
        "verdict": verdict,
        "thresholds": {
            "signal_density_ratio": {
                "normal_range": list(NORMAL_SIGNAL_DENSITY_RATIO_RANGE),
                "critical_range": list(CRITICAL_SIGNAL_DENSITY_RATIO_RANGE),
            },
            "would_open_density_ratio": {
                "normal_range": list(NORMAL_WOULD_OPEN_RATIO_RANGE),
                "critical_range": list(CRITICAL_WOULD_OPEN_RATIO_RANGE),
            },
            "spread_rejection_delta_pp": {
                "normal_max": NORMAL_SPREAD_REJECTION_DELTA_PP,
                "critical_max": CRITICAL_REJECTION_DELTA_PP,
            },
            "session_rejection_delta_pp": {
                "normal_max": NORMAL_SESSION_REJECTION_DELTA_PP,
                "critical_max": CRITICAL_REJECTION_DELTA_PP,
            },
            "directional_occupancy_delta_pp": {
                "normal_max": NORMAL_DIRECTIONAL_OCCUPANCY_DELTA_PP,
                "critical_max": CRITICAL_DIRECTIONAL_OCCUPANCY_DELTA_PP,
            },
        },
    }


def weekly_shadow_reviews(
    events: list[dict[str, Any]],
    *,
    replay_reference: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    normalized = [_normalize_event(row) for row in events]
    if not normalized:
        return []
    frame = pd.DataFrame(normalized)
    week_start = frame["timestamp"].dt.normalize() - pd.to_timedelta(frame["timestamp"].dt.weekday, unit="D")
    frame["week_start"] = week_start.dt.tz_convert("UTC")
    reviews: list[dict[str, Any]] = []
    for week_start, group in frame.groupby("week_start", sort=True):
        rows = group.to_dict("records")
        summary = summarize_shadow_events(
            rows,
            min_trading_days=MIN_WEEKLY_REVIEW_TRADING_DAYS,
            min_actionable_events=MIN_WEEKLY_REVIEW_ACTIONABLE_EVENTS,
        )
        drift = compute_drift_metrics(summary, replay_reference=replay_reference)
        eligible = bool(summary.get("evidence_sufficient", False))
        reviews.append(
            {
                "week_start_utc": week_start.isoformat(),
                "week_end_utc": (week_start + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)).isoformat(),
                "eligible_for_demotion": eligible,
                "summary": summary,
                "drift_metrics": drift,
                "critical_review": bool(eligible and drift.get("critical", False)),
            }
        )
    return reviews


def has_two_consecutive_critical_reviews(reviews: list[dict[str, Any]]) -> bool:
    critical_streak = 0
    for review in reviews:
        if bool(review.get("critical_review", False)):
            critical_streak += 1
            if critical_streak >= 2:
                return True
        else:
            critical_streak = 0
    return False


def render_shadow_summary_markdown(summary: dict[str, Any]) -> str:
    counts = dict(summary.get("counts", {}) or {})
    rates = dict(summary.get("rates", {}) or {})
    occupancy = dict(summary.get("directional_occupancy", {}) or {})
    requirements = dict(summary.get("evidence_requirements", {}) or {})
    shortfall = dict(summary.get("evidence_shortfall", {}) or {})

    min_days = int(requirements.get("min_trading_days", MIN_PROMOTION_TRADING_DAYS) or MIN_PROMOTION_TRADING_DAYS)
    min_actionable = int(
        requirements.get("min_actionable_events", MIN_PROMOTION_ACTIONABLE_EVENTS) or MIN_PROMOTION_ACTIONABLE_EVENTS
    )
    have_days = int(summary.get("trading_days", 0) or 0)
    have_actionable = int(summary.get("actionable_event_count", 0) or 0)
    remaining_days = int(shortfall.get("trading_days_remaining", max(min_days - have_days, 0)) or 0)
    remaining_actionable = int(shortfall.get("actionable_events_remaining", max(min_actionable - have_actionable, 0)) or 0)

    evidence_sufficient = bool(summary.get("evidence_sufficient", False))
    evidence_label = "SUFFICIENT" if evidence_sufficient else "TOO_SMALL"
    lines = [
        f"# Shadow Summary - {summary.get('symbol', 'UNKNOWN')}",
        "",
        f"* Manifest hash: `{summary.get('manifest_hash', '')}`",
        f"* Window: `{summary.get('shadow_window_start')}` -> `{summary.get('shadow_window_end')}`",
        f"* Trading days: `{summary.get('trading_days', 0)}`",
        f"* Actionable events: `{summary.get('actionable_event_count', 0)}`",
        f"* Evidence status: `{evidence_label}` (need `{min_days}` days + `{min_actionable}` actionable; remaining `{remaining_days}` days / `{remaining_actionable}` actionable)",
        "",
        "## Counts",
        f"* Events: `{summary.get('event_count', 0)}`",
        f"* Signals: `{counts.get('signal_count', 0)}`",
        f"* Opens: `{counts.get('would_open_count', 0)}`",
        f"* Closes: `{counts.get('would_close_count', 0)}`",
        f"* Holds: `{counts.get('would_hold_count', 0)}`",
        f"* Spread rejects: `{counts.get('spread_rejection_count', 0)}`",
        f"* Session rejects: `{counts.get('session_rejection_count', 0)}`",
        f"* Risk rejects: `{counts.get('risk_rejection_count', 0)}`",
        "",
        "## Context",
        f"* Macro-day bars: `{counts.get('context_macro_day_count', 0)}`",
        f"* Blackout bars: `{counts.get('context_blackout_count', 0)}` (`{rates.get('context_blackout_pct', 0.0):.2f}%`)",
        f"* Blocked entries: `{counts.get('context_block_entry_count', 0)}`",
        f"* Close-only reversals: `{counts.get('context_close_only_reversal_count', 0)}`",
        "",
        "## Rates",
        f"* Signal density: `{rates.get('signal_density', 0.0):.4f}`",
        f"* Would-open density: `{rates.get('would_open_density', 0.0):.4f}`",
        f"* Spread rejection pct: `{rates.get('spread_rejection_pct', 0.0):.2f}`",
        f"* Session rejection pct: `{rates.get('session_rejection_pct', 0.0):.2f}`",
        f"* Long open share: `{rates.get('long_open_share', 0.0):.2%}`",
        "",
        "## Occupancy",
        f"* Flat: `{occupancy.get('flat', 0.0):.2%}`",
        f"* Long: `{occupancy.get('long', 0.0):.2%}`",
        f"* Short: `{occupancy.get('short', 0.0):.2%}`",
    ]
    reason_counts = dict(summary.get("no_trade_reason_counts", {}) or {})
    if reason_counts:
        lines.extend(["", "## No-Trade Reasons"])
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0]))):
            lines.append(f"* `{reason}`: `{count}`")
    return "\n".join(lines) + "\n"


def write_shadow_summary(
    *,
    events_path: str | Path,
    summary_json_path: str | Path,
    summary_markdown_path: str | Path,
) -> dict[str, Any]:
    summary = summarize_shadow_events(load_shadow_events(events_path))
    json_path = Path(summary_json_path)
    md_path = Path(summary_markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_shadow_summary_markdown(summary), encoding="utf-8")
    return summary


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
