from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FAST_MODE_MAX_TIMESTEPS = 120_000
CURRENT_TRAINING_RUN_PATH = Path("checkpoints") / "current_training_run.json"
TERMINAL_RUN_STATES = {"completed", "collapsed", "stopped"}
RESULTS_DIRNAME = "results"
PROPOSALS_DIRNAME = "proposals"
LEDGER_DIRNAME = "ledger"
LEDGER_FILENAME = "experiments.jsonl"
_STALE_ACTIVE_RUN_SECONDS = 300

_SCALAR_TYPES = (str, int, float, bool)
_TOP_LEVEL_FIELDS = {
    "experiment_name",
    "symbol",
    "timesteps",
    "fast_mode",
    "baseline_reference",
    "rationale",
    "overrides",
    "tags",
    "parent_experiment",
}
_EXPERIMENT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ProposalValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ResearchLayout:
    root: Path
    proposals_dir: Path
    results_dir: Path
    ledger_dir: Path
    ledger_path: Path


@dataclass(frozen=True)
class Proposal:
    experiment_name: str
    symbol: str
    timesteps: int
    fast_mode: bool
    baseline_reference: str | None
    rationale: str
    overrides: dict[str, str | int | float | bool]
    tags: tuple[str, ...]
    parent_experiment: str | None
    source_path: Path
    raw_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "symbol": self.symbol,
            "timesteps": self.timesteps,
            "fast_mode": self.fast_mode,
            "baseline_reference": self.baseline_reference,
            "rationale": self.rationale,
            "overrides": dict(self.overrides),
            "tags": list(self.tags),
            "parent_experiment": self.parent_experiment,
        }


@dataclass(frozen=True)
class OverrideSpec:
    env_var: str
    value_type: str
    minimum: float | None = None
    allow_empty: bool = False


ALLOWLIST_SPECS: dict[str, OverrideSpec] = {
    "TRAIN_EXPERIMENT_PROFILE": OverrideSpec("TRAIN_EXPERIMENT_PROFILE", "str"),
    "TRAIN_WINDOW_SIZE": OverrideSpec("TRAIN_WINDOW_SIZE", "int", minimum=1),
    "TRAIN_CHURN_MIN_HOLD_BARS": OverrideSpec("TRAIN_CHURN_MIN_HOLD_BARS", "int", minimum=0),
    "TRAIN_CHURN_ACTION_COOLDOWN": OverrideSpec("TRAIN_CHURN_ACTION_COOLDOWN", "int", minimum=0),
    "TRAIN_CHURN_PENALTY_USD": OverrideSpec("TRAIN_CHURN_PENALTY_USD", "float", minimum=0.0),
    "TRAIN_ENTRY_SPREAD_Z_LIMIT": OverrideSpec("TRAIN_ENTRY_SPREAD_Z_LIMIT", "float", minimum=0.0),
    "TRAIN_REWARD_DOWNSIDE_RISK_COEF": OverrideSpec("TRAIN_REWARD_DOWNSIDE_RISK_COEF", "float", minimum=0.0),
    "TRAIN_REWARD_TURNOVER_COEF": OverrideSpec("TRAIN_REWARD_TURNOVER_COEF", "float", minimum=0.0),
    "TRAIN_REWARD_NET_RETURN_COEF": OverrideSpec("TRAIN_REWARD_NET_RETURN_COEF", "float", minimum=0.0),
    "TRAIN_REWARD_SCALE": OverrideSpec("TRAIN_REWARD_SCALE", "float", minimum=0.0),
    "TRAIN_REWARD_CLIP_LOW": OverrideSpec("TRAIN_REWARD_CLIP_LOW", "float"),
    "TRAIN_REWARD_CLIP_HIGH": OverrideSpec("TRAIN_REWARD_CLIP_HIGH", "float"),
    "TRAIN_ALPHA_GATE_ENABLED": OverrideSpec("TRAIN_ALPHA_GATE_ENABLED", "bool"),
    "TRAIN_ALPHA_GATE_MODEL": OverrideSpec("TRAIN_ALPHA_GATE_MODEL", "str"),
    "TRAIN_ALPHA_GATE_WARMUP_STEPS": OverrideSpec("TRAIN_ALPHA_GATE_WARMUP_STEPS", "int", minimum=0),
    "TRAIN_ALPHA_GATE_WARMUP_THRESHOLD_DELTA": OverrideSpec("TRAIN_ALPHA_GATE_WARMUP_THRESHOLD_DELTA", "float", minimum=0.0),
    "TRAIN_ALPHA_GATE_WARMUP_MARGIN_SCALE": OverrideSpec("TRAIN_ALPHA_GATE_WARMUP_MARGIN_SCALE", "float", minimum=0.0),
    "TRAIN_ADAPTIVE_KL_MAX_LR": OverrideSpec("TRAIN_ADAPTIVE_KL_MAX_LR", "float", minimum=0.0),
    "TRAIN_ADAPTIVE_KL_LOW": OverrideSpec("TRAIN_ADAPTIVE_KL_LOW", "float", minimum=0.0),
    "TRAIN_ADAPTIVE_KL_UP_MULT": OverrideSpec("TRAIN_ADAPTIVE_KL_UP_MULT", "float", minimum=0.0),
    "TRAIN_FAIL_FAST_ENABLED": OverrideSpec("TRAIN_FAIL_FAST_ENABLED", "bool"),
    "TRAIN_FAIL_FAST_WARMUP_STEPS": OverrideSpec("TRAIN_FAIL_FAST_WARMUP_STEPS", "int", minimum=0),
    "TRAIN_FAIL_FAST_CONSECUTIVE": OverrideSpec("TRAIN_FAIL_FAST_CONSECUTIVE", "int", minimum=1),
    "TRAIN_FAIL_FAST_SPARSE_ALPHA_GATE_BLOCK_RATE": OverrideSpec("TRAIN_FAIL_FAST_SPARSE_ALPHA_GATE_BLOCK_RATE", "float", minimum=0.0),
    "TRAIN_FAIL_FAST_APPROX_KL_MAX": OverrideSpec("TRAIN_FAIL_FAST_APPROX_KL_MAX", "float", minimum=0.0),
    "TRAIN_FAIL_FAST_EXPLAINED_VARIANCE_MAX": OverrideSpec("TRAIN_FAIL_FAST_EXPLAINED_VARIANCE_MAX", "float"),
    "TRAIN_FAIL_FAST_MAX_TRADE_COUNT": OverrideSpec("TRAIN_FAIL_FAST_MAX_TRADE_COUNT", "int", minimum=0),
    "TRAIN_FAIL_FAST_OVERTRADE_TRADE_COUNT": OverrideSpec("TRAIN_FAIL_FAST_OVERTRADE_TRADE_COUNT", "int", minimum=0),
    "TRAIN_FAIL_FAST_COST_SHARE_MIN": OverrideSpec("TRAIN_FAIL_FAST_COST_SHARE_MIN", "float", minimum=0.0),
    "TRAIN_PPO_LEARNING_RATE": OverrideSpec("TRAIN_PPO_LEARNING_RATE", "float", minimum=0.0),
    "TRAIN_PPO_N_STEPS": OverrideSpec("TRAIN_PPO_N_STEPS", "int", minimum=1),
    "TRAIN_PPO_BATCH_SIZE": OverrideSpec("TRAIN_PPO_BATCH_SIZE", "int", minimum=1),
    "TRAIN_PPO_N_EPOCHS": OverrideSpec("TRAIN_PPO_N_EPOCHS", "int", minimum=1),
    "TRAIN_PPO_ENT_COEF": OverrideSpec("TRAIN_PPO_ENT_COEF", "float", minimum=0.0),
    "TRAIN_PPO_TARGET_KL": OverrideSpec("TRAIN_PPO_TARGET_KL", "float", minimum=0.0),
}


def ensure_research_layout(repo_root: Path) -> ResearchLayout:
    research_root = repo_root / "research"
    proposals_dir = research_root / PROPOSALS_DIRNAME
    results_dir = research_root / RESULTS_DIRNAME
    ledger_dir = research_root / LEDGER_DIRNAME
    ledger_path = ledger_dir / LEDGER_FILENAME
    proposals_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    return ResearchLayout(
        root=research_root,
        proposals_dir=proposals_dir,
        results_dir=results_dir,
        ledger_dir=ledger_dir,
        ledger_path=ledger_path,
    )


def _require_object(payload: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProposalValidationError(f"{field} must be a JSON object.")
    return dict(payload)


def _require_str(value: Any, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProposalValidationError(f"{field} must be a string.")
    normalized = value.strip()
    if not normalized and not allow_empty:
        raise ProposalValidationError(f"{field} must not be empty.")
    return normalized


def _require_int(value: Any, *, field: str, minimum: float | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProposalValidationError(f"{field} must be an integer.")
    parsed = int(value)
    if minimum is not None and parsed < minimum:
        raise ProposalValidationError(f"{field} must be >= {int(minimum)}.")
    return parsed


def _require_float(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProposalValidationError(f"{field} must be a number.")
    parsed = float(value)
    if minimum is not None and parsed < float(minimum):
        raise ProposalValidationError(f"{field} must be >= {minimum}.")
    return parsed


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ProposalValidationError(f"{field} must be a boolean.")
    return bool(value)


def _validate_override_value(name: str, value: Any) -> str | int | float | bool:
    if not isinstance(value, _SCALAR_TYPES) or value is None:
        raise ProposalValidationError(f"Override {name} must be a scalar JSON value.")
    if name not in ALLOWLIST_SPECS:
        raise ProposalValidationError(
            f"Override {name} is not allowed. Allowed overrides: {', '.join(sorted(ALLOWLIST_SPECS))}."
        )
    spec = ALLOWLIST_SPECS[name]
    if spec.value_type == "str":
        return _require_str(value, field=name, allow_empty=spec.allow_empty)
    if spec.value_type == "int":
        return _require_int(value, field=name, minimum=spec.minimum)
    if spec.value_type == "float":
        return _require_float(value, field=name, minimum=spec.minimum)
    if spec.value_type == "bool":
        return _require_bool(value, field=name)
    raise ProposalValidationError(f"Unsupported override validator for {name}.")


def _normalize_experiment_name(raw_name: Any) -> str:
    name = _require_str(raw_name, field="experiment_name")
    if not _EXPERIMENT_RE.fullmatch(name):
        raise ProposalValidationError(
            "experiment_name must be slug-safe lowercase text matching [a-z0-9][a-z0-9_-]*."
        )
    return name


def _validate_tags(raw_tags: Any) -> tuple[str, ...]:
    if raw_tags is None:
        return ()
    if not isinstance(raw_tags, list):
        raise ProposalValidationError("tags must be a list of strings.")
    tags: list[str] = []
    for idx, tag in enumerate(raw_tags):
        tags.append(_require_str(tag, field=f"tags[{idx}]"))
    return tuple(tags)


def load_proposal(path: str | Path) -> Proposal:
    source_path = Path(path).resolve()
    payload = _require_object(json.loads(source_path.read_text(encoding="utf-8")), field="proposal")
    unknown_fields = sorted(set(payload) - _TOP_LEVEL_FIELDS)
    if unknown_fields:
        raise ProposalValidationError(
            f"Unsupported top-level fields: {', '.join(unknown_fields)}. "
            f"Allowed fields: {', '.join(sorted(_TOP_LEVEL_FIELDS))}."
        )

    experiment_name = _normalize_experiment_name(payload.get("experiment_name"))
    symbol = _require_str(payload.get("symbol"), field="symbol").upper()
    timesteps = _require_int(payload.get("timesteps"), field="timesteps", minimum=1)
    fast_mode = _require_bool(payload.get("fast_mode", False), field="fast_mode")
    if fast_mode and timesteps > FAST_MODE_MAX_TIMESTEPS:
        raise ProposalValidationError(
            f"fast_mode proposals must use timesteps <= {FAST_MODE_MAX_TIMESTEPS}."
        )
    baseline_reference = payload.get("baseline_reference")
    if baseline_reference is not None:
        baseline_reference = _require_str(baseline_reference, field="baseline_reference")
    rationale = _require_str(payload.get("rationale"), field="rationale")
    overrides_payload = _require_object(payload.get("overrides"), field="overrides")
    normalized_overrides: dict[str, str | int | float | bool] = {}
    for override_name, override_value in sorted(overrides_payload.items()):
        normalized_overrides[override_name] = _validate_override_value(override_name, override_value)
    if normalized_overrides.get("TRAIN_REWARD_CLIP_LOW") is not None and normalized_overrides.get("TRAIN_REWARD_CLIP_HIGH") is not None:
        if float(normalized_overrides["TRAIN_REWARD_CLIP_LOW"]) >= float(normalized_overrides["TRAIN_REWARD_CLIP_HIGH"]):
            raise ProposalValidationError("TRAIN_REWARD_CLIP_LOW must be less than TRAIN_REWARD_CLIP_HIGH.")
    tags = _validate_tags(payload.get("tags"))
    parent_experiment = payload.get("parent_experiment")
    if parent_experiment is not None:
        parent_experiment = _require_str(parent_experiment, field="parent_experiment")

    return Proposal(
        experiment_name=experiment_name,
        symbol=symbol,
        timesteps=timesteps,
        fast_mode=fast_mode,
        baseline_reference=baseline_reference,
        rationale=rationale,
        overrides=normalized_overrides,
        tags=tags,
        parent_experiment=parent_experiment,
        source_path=source_path,
        raw_payload=payload,
    )


def scalar_to_env(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def build_research_env_overrides(proposal: Proposal, artifacts_dir: Path) -> dict[str, dict[str, str]]:
    artifacts_dir = artifacts_dir.resolve()
    symbol = proposal.symbol.upper()
    train_env = {
        "TRAIN_ENV_MODE": "runtime",
        "TRAIN_SYMBOL": symbol,
        "TRAIN_TOTAL_TIMESTEPS": str(proposal.timesteps),
        "TRAIN_MODEL_DIR": str(artifacts_dir),
        "TRAIN_EXPORT_BEST_FOLD": "1",
        "TRAIN_RESUME_LATEST": "0",
        "TRAIN_DEBUG_ALLOW_BASELINE_BYPASS": "0",
    }
    for key, value in proposal.overrides.items():
        train_env[key] = scalar_to_env(value)

    eval_env = {
        "EVAL_SYMBOL": symbol,
        "EVAL_MANIFEST_PATH": str(artifacts_dir / f"artifact_manifest_{symbol}.json"),
        "EVAL_OUTPUT_DIR": str(artifacts_dir),
    }
    if proposal.fast_mode:
        eval_env["EVAL_SKIP_PLOT"] = "1"
    return {"train": train_env, "eval": eval_env}


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ProposalValidationError(f"Invalid JSONL row in {ledger_path} at line {line_number}: {exc}") from exc
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def append_jsonl_row(path: str | Path, row: dict[str, Any]) -> Path:
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return ledger_path


def _pid_is_running(raw_pid: Any) -> bool | None:
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        process_handle = kernel32.OpenProcess(0x100000, False, pid)
        if not process_handle:
            return False
        try:
            return int(kernel32.WaitForSingleObject(process_handle, 0)) == 0x00000102
        finally:
            kernel32.CloseHandle(process_handle)
    try:
        import os

        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _parse_utc_timestamp(raw_value: Any) -> datetime | None:
    if raw_value in (None, ""):
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _active_run_context_is_stale(payload: dict[str, Any]) -> bool:
    pid_running = _pid_is_running(payload.get("pid"))
    if pid_running is False:
        return True

    updated_at = _parse_utc_timestamp(payload.get("updated_at_utc"))
    if updated_at is None:
        return False

    heartbeat_path = payload.get("heartbeat_path")
    heartbeat_exists = False
    if heartbeat_path not in (None, ""):
        heartbeat_exists = Path(str(heartbeat_path)).exists()

    age_seconds = (datetime.now(timezone.utc) - updated_at).total_seconds()
    return age_seconds >= float(_STALE_ACTIVE_RUN_SECONDS) and not heartbeat_exists


def assert_no_active_training_run(current_run_path: str | Path) -> None:
    path = Path(current_run_path)
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return
    state = str(payload.get("state", "")).strip().lower()
    if state in TERMINAL_RUN_STATES or state.startswith("failed_"):
        return
    if _active_run_context_is_stale(payload):
        return
    symbol = str(payload.get("symbol", "")).strip().upper() or "UNKNOWN"
    checkpoints_root = str(payload.get("checkpoints_root", "")).strip() or "<unknown>"
    raise ProposalValidationError(
        f"Refusing to start research while training state={state or 'unknown'} is active for {symbol} "
        f"at {checkpoints_root}."
    )


def _ledger_sort_key(row: dict[str, Any]) -> tuple[float, str, str]:
    score = float(row.get("composite_score", float("-inf")) or float("-inf"))
    completed_at = str(row.get("completed_at_utc", "") or "")
    result_id = str(row.get("result_id", "") or "")
    return (score, completed_at, result_id)


def _row_is_comparable(
    row: dict[str, Any],
    *,
    proposal: Proposal,
    dataset_id: str | None,
    bar_construction_ticks_per_bar: int | None,
) -> bool:
    if str(row.get("status", "")).strip().lower() != "completed":
        return False
    if str(row.get("symbol", "")).strip().upper() != proposal.symbol.upper():
        return False
    if bool(row.get("fast_mode", False)) != proposal.fast_mode:
        return False
    row_dataset_id = row.get("dataset_id")
    if dataset_id and row_dataset_id and str(row_dataset_id) != str(dataset_id):
        return False
    row_bar_spec = row.get("bar_construction_ticks_per_bar")
    if bar_construction_ticks_per_bar is not None and row_bar_spec not in (None, ""):
        if int(row_bar_spec) != int(bar_construction_ticks_per_bar):
            return False
    return True


def resolve_research_baseline(
    proposal: Proposal,
    ledger_rows: list[dict[str, Any]],
    *,
    dataset_id: str | None,
    bar_construction_ticks_per_bar: int | None,
) -> dict[str, Any] | None:
    comparable_rows = [
        row
        for row in ledger_rows
        if _row_is_comparable(
            row,
            proposal=proposal,
            dataset_id=dataset_id,
            bar_construction_ticks_per_bar=bar_construction_ticks_per_bar,
        )
    ]

    if proposal.baseline_reference:
        by_id = [row for row in comparable_rows if str(row.get("result_id", "")) == proposal.baseline_reference]
        if by_id:
            baseline_row = sorted(by_id, key=_ledger_sort_key, reverse=True)[0]
        else:
            by_name = [
                row for row in comparable_rows if str(row.get("experiment_name", "")) == proposal.baseline_reference
            ]
            if by_name:
                baseline_row = sorted(by_name, key=_ledger_sort_key, reverse=True)[0]
            else:
                similar_rows = [
                    row
                    for row in ledger_rows
                    if str(row.get("result_id", "")) == proposal.baseline_reference
                    or str(row.get("experiment_name", "")) == proposal.baseline_reference
                ]
                if similar_rows:
                    raise ProposalValidationError(
                        f"baseline_reference {proposal.baseline_reference!r} exists in the ledger but is not "
                        "compatible with this symbol/mode bucket."
                    )
                raise ProposalValidationError(
                    f"baseline_reference {proposal.baseline_reference!r} was not found in the research ledger."
                )
    else:
        if not comparable_rows:
            return None
        baseline_row = sorted(comparable_rows, key=_ledger_sort_key, reverse=True)[0]

    row_trade_count = baseline_row.get("trade_count")
    normalized_trade_count = int(float(row_trade_count)) if row_trade_count not in (None, "") else 0
    return {
        "source": "research_result",
        "reference": str(baseline_row.get("result_id", "") or ""),
        "label": str(baseline_row.get("experiment_name", "") or ""),
        "experiment_name": str(baseline_row.get("experiment_name", "") or ""),
        "result_id": str(baseline_row.get("result_id", "") or ""),
        "composite_score": float(baseline_row.get("composite_score", 0.0) or 0.0),
        "metrics": {
            "timed_sharpe": float(baseline_row.get("timed_sharpe", 0.0) or 0.0),
            "profit_factor": float(baseline_row.get("profit_factor", 0.0) or 0.0),
            "expectancy_usd": float(baseline_row.get("expectancy_usd", 0.0) or 0.0),
            "trade_count": normalized_trade_count,
            "net_pnl_usd": float(baseline_row.get("net_pnl_usd", 0.0) or 0.0),
            "max_drawdown": float(baseline_row.get("max_drawdown", 0.0) or 0.0),
        },
        "dataset_id": baseline_row.get("dataset_id"),
        "bar_construction_ticks_per_bar": baseline_row.get("bar_construction_ticks_per_bar"),
    }


def select_baseline_gate_fallback(baseline_report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(baseline_report, dict):
        return None
    holdout_models = dict((baseline_report.get("holdout_metrics", {}) or {}).get("models", {}) or {})
    if not holdout_models:
        return None
    best_name, best_payload = max(
        holdout_models.items(),
        key=lambda item: (
            float(((item[1] or {}).get("metrics", {}) or {}).get("expectancy_usd", 0.0)),
            float(((item[1] or {}).get("metrics", {}) or {}).get("profit_factor", 0.0)),
            float(((item[1] or {}).get("metrics", {}) or {}).get("trade_count", 0.0)),
        ),
    )
    metrics = dict((best_payload or {}).get("metrics", {}) or {})
    return {
        "source": "baseline_gate",
        "reference": best_name,
        "label": best_name,
        "metrics": {
            "profit_factor": float(metrics.get("profit_factor", 0.0) or 0.0),
            "expectancy_usd": float(metrics.get("expectancy_usd", 0.0) or 0.0),
            "trade_count": int(float(metrics.get("trade_count", 0.0) or 0.0)),
            "net_pnl_usd": float(metrics.get("net_pnl_usd", 0.0) or 0.0),
        },
        "target_definition": dict(baseline_report.get("target_definition", {}) or {}),
    }
