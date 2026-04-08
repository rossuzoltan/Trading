from __future__ import annotations

from typing import Any


KEEP_SCORE_MIN = 0.5
PROMOTE_SCORE_MIN = 2.5
PROMOTION_SCORE_DELTA_MIN = 0.5
MIN_TRUSTED_TRADE_COUNT = 20


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, *, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _reconciliation_passed(replay_metrics: dict[str, Any] | None) -> bool:
    if not isinstance(replay_metrics, dict):
        return False
    reconciliation = dict(
        replay_metrics.get("metric_reconciliation", replay_metrics.get("metrics_reconciliation", {})) or {}
    )
    return bool(reconciliation.get("passed", False))


def compute_composite_score(
    *,
    replay_metrics: dict[str, Any] | None,
    training_diagnostics: dict[str, Any] | None,
    runtime_parity_verdict: dict[str, Any] | None,
    baseline_gate_passed: bool,
) -> dict[str, Any]:
    metrics = dict(replay_metrics or {})
    diagnostics = dict(training_diagnostics or {})
    parity = dict(runtime_parity_verdict or {})
    critical_failures: list[str] = []

    if not metrics:
        critical_failures.append("Replay metrics are missing.")

    timed_sharpe = _as_float(metrics.get("timed_sharpe"))
    profit_factor = _as_float(metrics.get("profit_factor"))
    expectancy_usd = _as_float(metrics.get("expectancy_usd", metrics.get("expectancy")))
    trade_count = _as_int(metrics.get("trade_count"))
    max_drawdown = _as_float(metrics.get("max_drawdown"))

    components = {
        "timed_sharpe": round(timed_sharpe * 2.0, 6),
        "profit_factor": round((max(min(profit_factor, 3.0), 0.0) - 1.0) * 3.0, 6),
        "expectancy_usd": round(expectancy_usd * 0.5, 6),
        "trade_count_credit": round(min(trade_count / float(MIN_TRUSTED_TRADE_COUNT), 1.0) * 1.5, 6),
        "max_drawdown_penalty": round(-max(max_drawdown, 0.0) * 8.0, 6),
    }
    penalties = {
        "low_trade_penalty": round(
            -(max(MIN_TRUSTED_TRADE_COUNT - trade_count, 0) / float(MIN_TRUSTED_TRADE_COUNT)) * 3.0,
            6,
        ),
        "baseline_gate_failed": 0.0,
        "reconciliation_failed": 0.0,
        "runtime_parity_misaligned": 0.0,
        "fragile_under_cost_stress": 0.0,
    }
    bonuses = {
        "training_deploy_ready": 0.5 if _as_bool(diagnostics.get("deploy_ready")) else 0.0,
    }

    if not baseline_gate_passed:
        penalties["baseline_gate_failed"] = -8.0
        critical_failures.append("Baseline gate failed.")

    if not _reconciliation_passed(metrics):
        penalties["reconciliation_failed"] = -10.0
        critical_failures.append("Replay accounting reconciliation failed.")

    parity_aligned = _as_bool(parity.get("research_vs_runtime_parity_aligned"), default=True)
    if not parity_aligned:
        penalties["runtime_parity_misaligned"] = -8.0
        critical_failures.append("Runtime parity baselines are not aligned with the research baseline gate.")

    fragile_under_cost_stress = _as_bool(parity.get("fragile_under_cost_stress"))
    if fragile_under_cost_stress:
        penalties["fragile_under_cost_stress"] = -4.0

    score = round(sum(components.values()) + sum(penalties.values()) + sum(bonuses.values()), 6)
    return {
        "score": score,
        "components": components,
        "penalties": penalties,
        "bonuses": bonuses,
        "critical_failures": critical_failures,
        "inputs": {
            "timed_sharpe": timed_sharpe,
            "profit_factor": profit_factor,
            "expectancy_usd": expectancy_usd,
            "trade_count": trade_count,
            "max_drawdown": max_drawdown,
            "baseline_gate_passed": bool(baseline_gate_passed),
            "reconciliation_passed": _reconciliation_passed(metrics),
            "runtime_parity_aligned": parity_aligned,
            "fragile_under_cost_stress": fragile_under_cost_stress,
        },
    }


def compare_against_baseline(
    *,
    replay_metrics: dict[str, Any] | None,
    score_summary: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = dict(replay_metrics or {})
    if baseline is None:
        return {
            "source": "none",
            "reference": None,
            "materially_better": False,
            "reason": "No comparable prior research result was available, and no baseline gate fallback was found.",
        }

    baseline_metrics = dict(baseline.get("metrics", {}) or {})
    source = str(baseline.get("source", "") or "unknown")
    if source == "research_result":
        baseline_score = _as_float(baseline.get("composite_score"))
        score_delta = round(_as_float(score_summary.get("score")) - baseline_score, 6)
        materially_better = bool(score_delta >= PROMOTION_SCORE_DELTA_MIN)
        return {
            "source": source,
            "reference": baseline.get("reference"),
            "label": baseline.get("label"),
            "baseline_score": baseline_score,
            "score_delta": score_delta,
            "materially_better": materially_better,
            "reason": (
                f"Composite score delta {score_delta:.3f} {'meets' if materially_better else 'does not meet'} "
                f"the promotion threshold of {PROMOTION_SCORE_DELTA_MIN:.3f}."
            ),
            "baseline_metrics": baseline_metrics,
        }

    replay_net_pnl = _as_float(metrics.get("net_pnl_usd"))
    replay_profit_factor = _as_float(metrics.get("profit_factor"))
    replay_expectancy = _as_float(metrics.get("expectancy_usd", metrics.get("expectancy")))
    replay_trade_count = _as_int(metrics.get("trade_count"))
    baseline_net_pnl = _as_float(baseline_metrics.get("net_pnl_usd"))
    baseline_profit_factor = _as_float(baseline_metrics.get("profit_factor"))
    baseline_expectancy = _as_float(baseline_metrics.get("expectancy_usd"))
    deltas = {
        "net_pnl_usd": round(replay_net_pnl - baseline_net_pnl, 6),
        "profit_factor": round(replay_profit_factor - baseline_profit_factor, 6),
        "expectancy_usd": round(replay_expectancy - baseline_expectancy, 6),
        "trade_count": replay_trade_count - _as_int(baseline_metrics.get("trade_count")),
    }
    materially_better = bool(
        replay_trade_count >= MIN_TRUSTED_TRADE_COUNT
        and deltas["net_pnl_usd"] > 0.0
        and deltas["profit_factor"] >= 0.0
        and deltas["expectancy_usd"] >= 0.0
    )
    return {
        "source": source,
        "reference": baseline.get("reference"),
        "label": baseline.get("label"),
        "materially_better": materially_better,
        "reason": (
            "Replay clears the baseline gate floor on shared economic metrics."
            if materially_better
            else "Replay does not clear the baseline gate floor on shared economic metrics."
        ),
        "deltas": deltas,
        "baseline_metrics": baseline_metrics,
    }


def build_research_decision(
    *,
    run_status: str,
    score_summary: dict[str, Any],
    baseline_comparison: dict[str, Any],
) -> dict[str, Any]:
    blockers = list(score_summary.get("critical_failures", []) or [])
    score = _as_float(score_summary.get("score"))
    materially_better = _as_bool(baseline_comparison.get("materially_better"))

    if run_status != "completed":
        blockers.insert(0, f"Experiment did not complete successfully: {run_status}.")
        decision = "reject"
    elif blockers:
        decision = "reject"
    elif score >= PROMOTE_SCORE_MIN and materially_better:
        decision = "promote_candidate"
    elif score >= KEEP_SCORE_MIN:
        decision = "keep"
    else:
        decision = "reject"

    rationale: list[str] = []
    if decision == "promote_candidate":
        rationale.append(
            f"Composite score {score:.3f} is above the promotion floor {PROMOTE_SCORE_MIN:.3f}."
        )
        rationale.append(str(baseline_comparison.get("reason", "Baseline improvement requirement passed.")))
    elif decision == "keep":
        rationale.append(
            f"Composite score {score:.3f} is above the keep floor {KEEP_SCORE_MIN:.3f}, "
            "but promotion requirements were not fully met."
        )
        if baseline_comparison.get("reason"):
            rationale.append(str(baseline_comparison["reason"]))
    else:
        if blockers:
            rationale.extend(str(blocker) for blocker in blockers)
        else:
            rationale.append(
                f"Composite score {score:.3f} is below the keep floor {KEEP_SCORE_MIN:.3f}."
            )
        if baseline_comparison.get("reason"):
            rationale.append(str(baseline_comparison["reason"]))

    return {
        "decision": decision,
        "blockers": blockers,
        "rationale": rationale,
    }
