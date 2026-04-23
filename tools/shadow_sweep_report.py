from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live_metrics import (
    MIN_PROMOTION_ACTIONABLE_EVENTS,
    MIN_PROMOTION_REALIZED_TRADE_COVERAGE,
    MIN_PROMOTION_TRADING_DAYS,
    resolve_shadow_evidence_paths,
)
from selector_manifest import (
    assert_execution_cost_profile_parity,
    compute_execution_cost_profile_hash,
    load_selector_manifest,
    resolve_execution_cost_profile,
)
from shadow_trade_accounting import summarize_shadow_trade_accounting
from strategies.rule_logic import diagnose_rule_decision


def _load_json(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8-sig")
    return json.loads(raw)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _runtime_gate_reason(decision_reason: str) -> str | None:
    lowered = _lower(decision_reason)
    if lowered == "session blocked":
        return "session"
    if lowered == "spread too high":
        return "spread_pips_limit"
    if lowered == "max position blocked":
        return "max_position_limit"
    if lowered == "loss stop blocked":
        return "daily_loss_stop"
    return None


@dataclass(frozen=True)
class ProfileConfig:
    profile_id: str
    manifest_hash: str
    manifest_path: Path
    rule_family: str
    rule_params: dict[str, Any]
    resolved_cost_profile: dict[str, float]
    resolved_cost_hash: str


@dataclass
class ProfileStats:
    profile_id: str
    manifest_hash: str
    event_count: int = 0
    trading_days: int = 0
    actionable_events: int = 0
    opens: int = 0
    closes: int = 0
    holds: int = 0
    flat: int = 0
    signals: int = 0
    rule_candidate_signals: int = 0
    raw_price_signals: int = 0
    guard_blocked_price_signals: int = 0
    guard_failure_counts: dict[str, int] = None  # type: ignore[assignment]
    rule_block_counts: dict[str, int] = None  # type: ignore[assignment]
    runtime_gate_block_counts: dict[str, int] = None  # type: ignore[assignment]
    no_trade_reason_counts: dict[str, int] = None  # type: ignore[assignment]
    est_trade_count: int = 0
    est_net_pips: float | None = None
    avg_net_pips: float | None = None
    avg_net_pips_ci95_low: float | None = None
    avg_net_pips_ci95_high: float | None = None
    realized_trade_coverage: float = 0.0
    cost_parity_ok: bool = False
    cost_parity_reason: str | None = None
    trade_accounting_mode: str | None = None
    trade_accounting_issues: list[str] = None  # type: ignore[assignment]
    eligible_for_ranking: bool = False
    ranking_blockers: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.guard_failure_counts is None:
            self.guard_failure_counts = {}
        if self.rule_block_counts is None:
            self.rule_block_counts = {}
        if self.runtime_gate_block_counts is None:
            self.runtime_gate_block_counts = {}
        if self.no_trade_reason_counts is None:
            self.no_trade_reason_counts = {}
        if self.trade_accounting_issues is None:
            self.trade_accounting_issues = []
        if self.ranking_blockers is None:
            self.ranking_blockers = []


def _count(d: dict[str, int], key: str) -> None:
    d[key] = int(d.get(key, 0) or 0) + 1


def _extract_features(row: dict[str, Any]) -> dict[str, Any]:
    core = row.get("core_features")
    if isinstance(core, dict):
        return dict(core)
    full = row.get("full_features")
    if isinstance(full, dict):
        return dict(full)
    return {}


def _load_run_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ranking_blockers(stats: ProfileStats) -> list[str]:
    blockers: list[str] = []
    if stats.trading_days < MIN_PROMOTION_TRADING_DAYS:
        blockers.append(f"trading_days<{MIN_PROMOTION_TRADING_DAYS}")
    if stats.actionable_events < MIN_PROMOTION_ACTIONABLE_EVENTS:
        blockers.append(f"actionable_events<{MIN_PROMOTION_ACTIONABLE_EVENTS}")
    if stats.realized_trade_coverage < MIN_PROMOTION_REALIZED_TRADE_COVERAGE:
        blockers.append(f"realized_trade_coverage<{MIN_PROMOTION_REALIZED_TRADE_COVERAGE:.2f}")
    if not stats.cost_parity_ok:
        blockers.append("unproven_cost_parity")
    if not stats.est_trade_count or stats.est_net_pips is None:
        blockers.append("no_realized_trade_accounting")
    return blockers


def build_report(
    *,
    audit_root: Path,
    ladder_json_path: Path,
    symbol: str,
    account_currency: str = "USD",
) -> tuple[dict[str, Any], str]:
    ladder = _load_json(ladder_json_path)
    generated = list(ladder.get("generated", []) or [])
    if not generated:
        raise RuntimeError(f"ladder_json has no generated profiles: {ladder_json_path}")

    profiles: list[ProfileConfig] = []
    for item in generated:
        profile_id = str(item.get("profile_id") or "").strip()
        manifest_hash = str(item.get("manifest_hash") or "").strip()
        manifest_path_raw = str(item.get("manifest_path") or "").strip()
        if not profile_id or not manifest_hash or not manifest_path_raw:
            continue
        manifest_path = Path(manifest_path_raw).resolve()
        manifest = load_selector_manifest(
            manifest_path,
            verify_manifest_hash=True,
            strict_manifest_hash=True,
            require_component_hashes=True,
        )
        profiles.append(
            ProfileConfig(
                profile_id=profile_id,
                manifest_hash=manifest_hash,
                manifest_path=manifest_path,
                rule_family=str(manifest.rule_family or ""),
                rule_params=dict(manifest.rule_params or {}),
                resolved_cost_profile=resolve_execution_cost_profile(manifest),
                resolved_cost_hash=compute_execution_cost_profile_hash(manifest),
            )
        )
    if not profiles:
        raise RuntimeError("No valid profiles loaded from ladder_json.")

    per_profile: dict[str, ProfileStats] = {}
    by_bar: dict[str, dict[str, dict[str, Any]]] = {}
    all_timestamps: list[pd.Timestamp] = []

    for profile in profiles:
        paths = resolve_shadow_evidence_paths(symbol=symbol, manifest_hash=profile.manifest_hash, base_dir=audit_root)
        events = _iter_jsonl(paths.events_path)
        events_sorted = sorted(events, key=lambda row: str(row.get("bar_ts") or row.get("timestamp_utc") or ""))
        stats = ProfileStats(profile_id=profile.profile_id, manifest_hash=profile.manifest_hash)
        stats.event_count = len(events_sorted)
        if events_sorted:
            timestamps = [_ts(row.get("timestamp_utc") or row.get("bar_ts")) for row in events_sorted]
            all_timestamps.extend(timestamps)
            stats.trading_days = len(sorted({ts.date().isoformat() for ts in timestamps}))
        run_meta = _load_run_meta(paths.run_meta_path)
        try:
            assert_execution_cost_profile_parity(
                profile.resolved_cost_profile,
                run_meta.get("resolved_execution_cost_profile") or run_meta.get("cost_model") or {},
                observed_hash=run_meta.get("resolved_execution_cost_profile_hash"),
                context_label=f"shadow run cost parity {profile.profile_id}",
            )
            stats.cost_parity_ok = True
        except Exception as exc:
            stats.cost_parity_ok = False
            stats.cost_parity_reason = str(exc)

        trade_accounting = summarize_shadow_trade_accounting(
            events=events_sorted,
            symbol=symbol,
            commission_per_lot=profile.resolved_cost_profile.get("commission_per_lot"),
            slippage_pips=profile.resolved_cost_profile.get("slippage_pips"),
            account_currency=account_currency,
        )
        stats.trade_accounting_mode = str(trade_accounting.get("pricing_mode") or "unknown")
        stats.trade_accounting_issues = list(trade_accounting.get("issues", []) or [])
        stats.realized_trade_coverage = float(trade_accounting.get("realized_trade_coverage", 0.0) or 0.0)
        if stats.cost_parity_ok:
            stats.est_trade_count = int(trade_accounting.get("trade_count", 0) or 0)
            stats.est_net_pips = trade_accounting.get("net_pips")
            stats.avg_net_pips = trade_accounting.get("avg_net_pips")
            stats.avg_net_pips_ci95_low = trade_accounting.get("avg_net_pips_ci95_low")
            stats.avg_net_pips_ci95_high = trade_accounting.get("avg_net_pips_ci95_high")
        else:
            stats.trade_accounting_issues.insert(
                0,
                stats.cost_parity_reason or "unproven runtime/report cost parity",
            )

        for row in events_sorted:
            bar_ts = str(row.get("bar_ts") or row.get("timestamp_utc") or "").strip()
            if not bar_ts:
                continue

            allow_execution = bool(row.get("allow_execution", False))
            signal = int(row.get("signal_direction", row.get("signal", 0)) or 0)
            action_state = str(row.get("action_state") or "").strip().lower()
            reason = str(row.get("reason", row.get("no_trade_reason", "")) or "")
            lowered_reason = _lower(reason)

            if bool(row.get("would_open", False)):
                stats.opens += 1
            if bool(row.get("would_close", False)):
                stats.closes += 1
            if bool(row.get("would_hold", row.get("would_hold_position", False))):
                stats.holds += 1
            if bool(row.get("would_remain_flat", False)) or action_state == "flat":
                stats.flat += 1
            if bool(row.get("would_open", False)) or bool(row.get("would_close", False)):
                stats.actionable_events += 1
            if signal != 0:
                stats.signals += 1

            if lowered_reason:
                _count(stats.no_trade_reason_counts, lowered_reason)

            features = _extract_features(row)
            diagnostics = diagnose_rule_decision(profile.rule_family, features, profile.rule_params)
            candidate = int(diagnostics.get("candidate_signal", 0) or 0)
            raw_price_signal = int(diagnostics.get("raw_price_signal", 0) or 0)
            failed_checks = diagnostics.get("failed_checks")
            if raw_price_signal != 0:
                stats.raw_price_signals += 1
            if candidate != 0:
                stats.rule_candidate_signals += 1
            if raw_price_signal != 0 and candidate == 0:
                stats.guard_blocked_price_signals += 1
                if isinstance(failed_checks, list):
                    for item in failed_checks:
                        key = str(item or "").strip() or "unknown"
                        _count(stats.guard_failure_counts, key)

            runtime_gate = _runtime_gate_reason(reason)
            if runtime_gate is not None and not allow_execution:
                _count(stats.runtime_gate_block_counts, runtime_gate)
            if lowered_reason == "no signal":
                rule_block = str(diagnostics.get("block_reason") or "").strip() or "unknown"
                _count(stats.rule_block_counts, rule_block)

            by_bar.setdefault(bar_ts, {})[profile.profile_id] = {
                "signal": signal,
                "allow": allow_execution,
                "action": action_state,
                "reason": lowered_reason,
                "candidate": candidate,
            }

        stats.ranking_blockers = _ranking_blockers(stats)
        stats.eligible_for_ranking = len(stats.ranking_blockers) == 0
        per_profile[profile.profile_id] = stats

    window_start = min(all_timestamps).isoformat() if all_timestamps else None
    window_end = max(all_timestamps).isoformat() if all_timestamps else None

    divergence: list[dict[str, Any]] = []
    for bar_ts, decisions in sorted(by_bar.items(), key=lambda item: item[0]):
        signature_to_profiles: dict[str, list[str]] = {}
        for profile_id, payload in decisions.items():
            sig = f"sig={payload['signal']} allow={int(bool(payload['allow']))} act={payload['action']} reason={payload['reason']}"
            signature_to_profiles.setdefault(sig, []).append(profile_id)
        if len(signature_to_profiles) <= 1:
            continue
        divergence.append(
            {
                "bar_ts": bar_ts,
                "clusters": [{"signature": sig, "profiles": sorted(profiles)} for sig, profiles in signature_to_profiles.items()],
                "cluster_count": len(signature_to_profiles),
            }
        )

    payload = {
        "audit_root": str(audit_root),
        "symbol": symbol,
        "window_start_utc": window_start,
        "window_end_utc": window_end,
        "profile_count": len(per_profile),
        "ranking_requirements": {
            "min_trading_days": MIN_PROMOTION_TRADING_DAYS,
            "min_actionable_events": MIN_PROMOTION_ACTIONABLE_EVENTS,
            "min_realized_trade_coverage": MIN_PROMOTION_REALIZED_TRADE_COVERAGE,
            "require_cost_parity": True,
        },
        "profiles": {
            profile_id: {
                "manifest_hash": stats.manifest_hash,
                "event_count": stats.event_count,
                "trading_days": stats.trading_days,
                "actionable_events": stats.actionable_events,
                "opens": stats.opens,
                "closes": stats.closes,
                "holds": stats.holds,
                "flat": stats.flat,
                "signals": stats.signals,
                "rule_candidate_signals": stats.rule_candidate_signals,
                "raw_price_signals": stats.raw_price_signals,
                "guard_blocked_price_signals": stats.guard_blocked_price_signals,
                "guard_failure_counts": stats.guard_failure_counts,
                "rule_block_counts": stats.rule_block_counts,
                "runtime_gate_block_counts": stats.runtime_gate_block_counts,
                "no_trade_reason_counts": stats.no_trade_reason_counts,
                "est_trade_count": stats.est_trade_count,
                "est_net_pips": stats.est_net_pips,
                "avg_net_pips": stats.avg_net_pips,
                "avg_net_pips_ci95_low": stats.avg_net_pips_ci95_low,
                "avg_net_pips_ci95_high": stats.avg_net_pips_ci95_high,
                "realized_trade_coverage": stats.realized_trade_coverage,
                "cost_parity_ok": stats.cost_parity_ok,
                "cost_parity_reason": stats.cost_parity_reason,
                "trade_accounting_mode": stats.trade_accounting_mode,
                "trade_accounting_issues": stats.trade_accounting_issues,
                "eligible_for_ranking": stats.eligible_for_ranking,
                "ranking_blockers": stats.ranking_blockers,
            }
            for profile_id, stats in per_profile.items()
        },
        "eligible_rankings": [
            {
                "profile_id": stats.profile_id,
                "est_net_pips": stats.est_net_pips,
                "avg_net_pips": stats.avg_net_pips,
                "est_trade_count": stats.est_trade_count,
            }
            for stats in sorted(
                [item for item in per_profile.values() if item.eligible_for_ranking and item.est_net_pips is not None],
                key=lambda item: (float(item.est_net_pips or 0.0), float(item.avg_net_pips or 0.0)),
                reverse=True,
            )
        ],
        "divergence_bars": divergence[-20:],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    def _top(d: dict[str, int], n: int = 6) -> list[str]:
        return [f"{k}={v}" for k, v in sorted(d.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:n]]

    lines: list[str] = []
    lines.append(f"# Shadow Sweep Report - {symbol}")
    lines.append("")
    lines.append(f"* Audit root: `{audit_root}`")
    lines.append(f"* Window: `{window_start}` -> `{window_end}`")
    lines.append(f"* Profiles: `{len(per_profile)}`")
    lines.append("")
    eligible_rankings = list(payload["eligible_rankings"])
    lines.append("## Ranking")
    if eligible_rankings:
        for index, ranked in enumerate(eligible_rankings, start=1):
            lines.append(
                f"* `{index}. {ranked['profile_id']}` net=`{float(ranked['est_net_pips'] or 0.0):.2f}` "
                f"avg=`{float(ranked['avg_net_pips'] or 0.0):.2f}` trades=`{int(ranked['est_trade_count'] or 0)}`"
            )
    else:
        lines.append(
            "* Ranking withheld: no profile met the minimum sample/realism gate "
            f"(`{MIN_PROMOTION_TRADING_DAYS}` days, `{MIN_PROMOTION_ACTIONABLE_EVENTS}` actionable, "
            f"`{MIN_PROMOTION_REALIZED_TRADE_COVERAGE:.0%}` realized coverage, proven cost parity)."
        )
    lines.append("")
    lines.append("## Per-Profile")
    for profile_id in sorted(per_profile.keys()):
        stats = per_profile[profile_id]
        lines.append(f"### {profile_id}")
        lines.append(f"* Manifest hash: `{stats.manifest_hash}`")
        lines.append(f"* Events: `{stats.event_count}` Trading days: `{stats.trading_days}` Actionable: `{stats.actionable_events}`")
        lines.append(f"* Opens: `{stats.opens}` Closes: `{stats.closes}` Holds: `{stats.holds}` Flat: `{stats.flat}`")
        lines.append(
            f"* Realism: coverage=`{stats.realized_trade_coverage:.2%}` cost_parity=`{stats.cost_parity_ok}` "
            f"rankable=`{stats.eligible_for_ranking}`"
        )
        lines.append(
            f"* Signals: `{stats.signals}` Rule-candidate: `{stats.rule_candidate_signals}` "
            f"Raw-price: `{stats.raw_price_signals}` Guard-blocked(raw): `{stats.guard_blocked_price_signals}`"
        )
        if stats.guard_failure_counts:
            lines.append(f"* Guard failures (top): `{', '.join(_top(stats.guard_failure_counts))}`")
        if stats.rule_block_counts:
            lines.append(f"* Rule blocks (top): `{', '.join(_top(stats.rule_block_counts))}`")
        if stats.runtime_gate_block_counts:
            lines.append(f"* Runtime gate blocks (top): `{', '.join(_top(stats.runtime_gate_block_counts))}`")
        if stats.est_trade_count and stats.est_net_pips is not None:
            lines.append(
                f"* Net PnL (event snapshots): `{stats.est_net_pips:.2f}` pips over `{stats.est_trade_count}` trades"
            )
            if stats.avg_net_pips is not None:
                lines.append(
                    f"* Avg/trade: `{stats.avg_net_pips:.2f}` pips "
                    f"(95% CI `{stats.avg_net_pips_ci95_low:.2f}` .. `{stats.avg_net_pips_ci95_high:.2f}`)"
                )
        if stats.cost_parity_reason:
            lines.append(f"* Cost parity note: `{stats.cost_parity_reason}`")
        if stats.ranking_blockers:
            lines.append(f"* Ranking blockers: `{', '.join(stats.ranking_blockers)}`")
        if stats.trade_accounting_issues:
            lines.append(f"* Accounting issues (top): `{'; '.join(stats.trade_accounting_issues[:4])}`")
        lines.append("")

    if divergence:
        lines.append("## Divergence (Last 10)")
        for item in divergence[-10:]:
            lines.append(f"* `{item['bar_ts']}` clusters=`{item['cluster_count']}`")
            for cluster in item["clusters"]:
                profiles_list = ", ".join(cluster["profiles"])
                lines.append(f"  - `{cluster['signature']}` -> {profiles_list}")
        lines.append("")

    markdown = "\n".join(lines).rstrip() + "\n"
    return payload, markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a multi-profile report from shadow sweep artifacts.")
    parser.add_argument("--audit-root", required=True, help="Base artifact dir (contains SYMBOL/manifest_hash/... dirs).")
    parser.add_argument("--ladder-json", required=True, help="shadow_profile_evidence_ladder_v1.json path.")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. EURUSD.")
    parser.add_argument("--out-dir", default=None, help="Optional output dir for report files.")
    parser.add_argument("--account-currency", default="USD")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_root = Path(args.audit_root).resolve()
    ladder_json = Path(args.ladder_json).resolve()
    symbol = str(args.symbol).upper()
    payload, markdown = build_report(
        audit_root=audit_root,
        ladder_json_path=ladder_json,
        symbol=symbol,
        account_currency=str(args.account_currency or "USD"),
    )

    out_dir = Path(args.out_dir).resolve() if args.out_dir else audit_root / "_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"shadow_sweep_report_{symbol}_{stamp}.md"
    json_path = out_dir / f"shadow_sweep_report_{symbol}_{stamp}.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(str(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
