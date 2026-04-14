from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import sklearn

from feature_engine import FEATURE_COLS

MANIFEST_VERSION = "4"
DEFAULT_MANIFEST_NAME = "selector_artifact_manifest.json"

EngineType = Literal["ML", "RULE"]
ReleaseStage = Literal["research", "paper_live_candidate", "production"]


@dataclass(frozen=True)
class LabelDefinition:
    path: str
    target_column: str
    horizon_bars: int
    is_classification: bool


@dataclass(frozen=True)
class CostModel:
    commission_per_lot: float
    slippage_pips: float


@dataclass(frozen=True)
class ThresholdPolicy:
    min_edge_pips: float
    reject_ambiguous: bool


@dataclass(frozen=True)
class RuntimeConstraints:
    session_filter_active: bool
    spread_sanity_max_pips: float
    max_concurrent_positions: int
    daily_loss_stop_usd: float
    rollover_block_utc_hours: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class AlphaGateSpec:
    enabled: bool = False
    model_path: str | None = None
    model_sha256: str | None = None
    probability_threshold: float | None = None
    probability_margin: float | None = None
    min_edge_pips: float | None = None


@dataclass(frozen=True)
class SelectorManifest:
    manifest_version: str
    strategy_symbol: str
    engine_type: EngineType
    release_stage: ReleaseStage = "research"
    live_trading_approved: bool = False
    model_path: str | None = None
    model_version: str = "1.0.0"
    model_sha256: str | None = None
    rule_family: str | None = None
    entry_rule_version: str = "1.0.0"
    exit_rule_version: str = "1.0.0"
    evaluator_hash: str = ""
    logic_hash: str = ""
    feature_schema: list[str] = field(default_factory=list)
    feature_schema_hash: str = ""
    dataset_id: str = ""
    dataset_fingerprint: str = ""
    sklearn_version: str | None = None
    created_from_git_commit: str = "unknown"
    bar_construction_ticks_per_bar: int | None = None
    ticks_per_bar: int | None = None
    holdout_start_utc: str | None = None
    label_definition: dict[str, Any] = field(default_factory=dict)
    cost_model: dict[str, Any] = field(default_factory=dict)
    threshold_policy: dict[str, Any] = field(default_factory=dict)
    runtime_constraints: dict[str, Any] = field(default_factory=dict)
    rule_params: dict[str, Any] = field(default_factory=dict)
    alpha_gate: dict[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""
    startup_truth_snapshot: dict[str, Any] = field(default_factory=dict)
    replay_parity_reference: str = ""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(data: Any) -> str:
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def dataset_id_for_path(path: str | Path) -> str:
    return _file_sha256(Path(path))


def _normalize_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Unsupported manifest mapping payload: {type(value).__name__}")


def _normalize_manifest_payload(raw: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw)
    payload.setdefault("manifest_version", MANIFEST_VERSION)
    payload.setdefault("engine_type", "ML")
    payload.setdefault("release_stage", "research")
    payload.setdefault("live_trading_approved", False)
    if payload.get("bar_construction_ticks_per_bar") is None and payload.get("ticks_per_bar") is not None:
        payload["bar_construction_ticks_per_bar"] = payload.get("ticks_per_bar")
    if payload.get("ticks_per_bar") is None and payload.get("bar_construction_ticks_per_bar") is not None:
        payload["ticks_per_bar"] = payload.get("bar_construction_ticks_per_bar")
    for key in (
        "label_definition",
        "cost_model",
        "threshold_policy",
        "runtime_constraints",
        "rule_params",
        "alpha_gate",
        "startup_truth_snapshot",
    ):
        payload[key] = _normalize_mapping(payload.get(key))
    payload.setdefault("feature_schema", [])
    payload.setdefault("feature_schema_hash", "")
    payload.setdefault("dataset_id", "")
    payload.setdefault("dataset_fingerprint", payload.get("dataset_id", ""))
    payload.setdefault("created_from_git_commit", "unknown")
    payload.setdefault("manifest_hash", "")
    payload.setdefault("replay_parity_reference", "")
    payload.setdefault("evaluator_hash", "")
    payload.setdefault("logic_hash", "")
    payload.setdefault("entry_rule_version", "1.0.0")
    payload.setdefault("exit_rule_version", "1.0.0")
    payload.setdefault("model_version", "1.0.0")
    payload.setdefault("rule_family", None)
    payload.setdefault("model_path", None)
    payload.setdefault("model_sha256", None)
    payload.setdefault("sklearn_version", None)
    return payload


def manifest_to_payload(manifest: SelectorManifest | dict[str, Any], *, include_manifest_hash: bool = True) -> dict[str, Any]:
    payload = asdict(manifest) if isinstance(manifest, SelectorManifest) else dict(manifest)
    normalized = _normalize_manifest_payload(payload)
    if not include_manifest_hash:
        normalized["manifest_hash"] = ""
    return normalized


def compute_selector_manifest_hash(manifest: SelectorManifest | dict[str, Any]) -> str:
    return _content_hash(manifest_to_payload(manifest, include_manifest_hash=False))


def validate_selector_manifest(
    manifest: SelectorManifest,
    *,
    verify_manifest_hash: bool = False,
    require_component_hashes: bool = False,
    require_paper_live_safety: bool = False,
) -> None:
    if manifest.manifest_version != MANIFEST_VERSION:
        raise RuntimeError(
            f"Selector manifest version mismatch: expected {MANIFEST_VERSION}, got {manifest.manifest_version}."
        )
    if manifest.engine_type not in {"ML", "RULE"}:
        raise RuntimeError(f"Unsupported selector manifest engine_type={manifest.engine_type!r}.")
    if manifest.strategy_symbol.strip().upper() != manifest.strategy_symbol:
        raise RuntimeError("strategy_symbol must be stored in uppercase.")
    if manifest.engine_type == "ML":
        if not manifest.model_path:
            raise RuntimeError("ML selector manifest is missing model_path.")
        if not manifest.model_version:
            raise RuntimeError("ML selector manifest is missing model_version.")
        if not manifest.model_sha256:
            raise RuntimeError("ML selector manifest is missing model_sha256.")
    if manifest.engine_type == "RULE" and not manifest.rule_family:
        raise RuntimeError("RULE selector manifest is missing rule_family.")
    alpha_gate = dict(manifest.alpha_gate or {})
    alpha_gate_enabled = bool(alpha_gate.get("enabled", False))
    if alpha_gate_enabled:
        model_path = str(alpha_gate.get("model_path") or "").strip()
        if not model_path:
            raise RuntimeError("Selector manifest alpha_gate.enabled=true but model_path is missing.")
        model_sha = str(alpha_gate.get("model_sha256") or "").strip()
        if not model_sha:
            raise RuntimeError("Selector manifest alpha_gate.enabled=true but model_sha256 is missing.")
        threshold = alpha_gate.get("probability_threshold")
        if threshold is not None:
            threshold_f = float(threshold)
            if not 0.0 <= threshold_f <= 1.0:
                raise RuntimeError("Selector manifest alpha_gate.probability_threshold must be in [0, 1].")
        margin = alpha_gate.get("probability_margin")
        if margin is not None and float(margin) < 0.0:
            raise RuntimeError("Selector manifest alpha_gate.probability_margin must be >= 0.")
        min_edge = alpha_gate.get("min_edge_pips")
        if min_edge is not None and float(min_edge) < 0.0:
            raise RuntimeError("Selector manifest alpha_gate.min_edge_pips must be >= 0.")
    if require_component_hashes:
        if not manifest.evaluator_hash:
            raise RuntimeError("Selector manifest is missing evaluator_hash.")
        if not manifest.logic_hash:
            raise RuntimeError("Selector manifest is missing logic_hash.")
    if require_paper_live_safety:
        if manifest.release_stage != "paper_live_candidate":
            raise RuntimeError(
                f"Expected release_stage='paper_live_candidate', got {manifest.release_stage!r}."
            )
        if manifest.live_trading_approved:
            raise RuntimeError("Paper-live RC manifests must keep live_trading_approved=false.")
    if verify_manifest_hash:
        expected_hash = compute_selector_manifest_hash(manifest)
        if manifest.manifest_hash != expected_hash:
            raise RuntimeError(
                "Selector manifest hash mismatch. "
                f"stored={manifest.manifest_hash!r} computed={expected_hash!r}"
            )


def validate_paper_live_candidate_manifest(
    manifest: SelectorManifest,
    *,
    verify_manifest_hash: bool = True,
) -> None:
    validate_selector_manifest(
        manifest,
        verify_manifest_hash=verify_manifest_hash,
        require_component_hashes=True,
        require_paper_live_safety=True,
    )


def save_selector_manifest(manifest: SelectorManifest, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest_to_payload(manifest, include_manifest_hash=False)
    payload["manifest_hash"] = compute_selector_manifest_hash(payload)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def load_selector_manifest(
    path: str | Path,
    *,
    verify_manifest_hash: bool = False,
    strict_manifest_hash: bool = False,
    require_component_hashes: bool = False,
) -> SelectorManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = _normalize_manifest_payload(raw)
    if strict_manifest_hash and not str(payload.get("manifest_hash") or "").strip():
        raise RuntimeError("Selector manifest is missing manifest_hash in strict mode.")
    if not payload.get("manifest_hash"):
        payload["manifest_hash"] = compute_selector_manifest_hash(payload)
    manifest = SelectorManifest(**payload)
    validate_selector_manifest(
        manifest,
        verify_manifest_hash=verify_manifest_hash,
        require_component_hashes=require_component_hashes,
    )
    return manifest


def _default_truth_snapshot(
    *,
    strategy_symbol: str,
    engine_type: EngineType,
    ticks_per_bar: int | None,
    release_stage: ReleaseStage,
    live_trading_approved: bool,
    evaluator_hash: str,
    logic_hash: str,
) -> dict[str, Any]:
    return {
        "strategy_symbol": strategy_symbol.upper(),
        "engine_type": engine_type,
        "ticks_per_bar": int(ticks_per_bar) if ticks_per_bar is not None else None,
        "release_stage": release_stage,
        "live_trading_approved": bool(live_trading_approved),
        "evaluator_hash": evaluator_hash,
        "logic_hash": logic_hash,
    }


def create_selector_manifest(
    *,
    strategy_symbol: str,
    model_path: str | Path,
    model_version: str,
    feature_schema: list[str],
    dataset_path: str | Path,
    ticks_per_bar: int,
    bar_construction_ticks_per_bar: int | None = None,
    holdout_start_utc: str | None = None,
    label_definition: LabelDefinition | dict[str, Any] | None = None,
    cost_model: CostModel | dict[str, Any] | None = None,
    threshold_policy: ThresholdPolicy | dict[str, Any] | None = None,
    runtime_constraints: RuntimeConstraints | dict[str, Any] | None = None,
    git_commit: str = "unknown",
    release_stage: ReleaseStage = "research",
    evaluator_hash: str = "",
    logic_hash: str = "",
    replay_parity_reference: str = "",
) -> SelectorManifest:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Selector model path does not exist: {model_path}")
    resolved_ticks_per_bar = int(bar_construction_ticks_per_bar or ticks_per_bar)
    feature_schema = list(feature_schema)
    dataset_id = dataset_id_for_path(dataset_path)
    return SelectorManifest(
        manifest_version=MANIFEST_VERSION,
        strategy_symbol=strategy_symbol.upper(),
        engine_type="ML",
        release_stage=release_stage,
        live_trading_approved=False,
        model_path=str(model_path),
        model_version=model_version,
        model_sha256=_file_sha256(model_path),
        evaluator_hash=evaluator_hash,
        logic_hash=logic_hash,
        feature_schema=feature_schema,
        feature_schema_hash=_content_hash(feature_schema),
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_id,
        sklearn_version=sklearn.__version__,
        created_from_git_commit=git_commit,
        bar_construction_ticks_per_bar=resolved_ticks_per_bar,
        ticks_per_bar=resolved_ticks_per_bar,
        holdout_start_utc=holdout_start_utc,
        label_definition=_normalize_mapping(label_definition),
        cost_model=_normalize_mapping(cost_model),
        threshold_policy=_normalize_mapping(threshold_policy),
        runtime_constraints=_normalize_mapping(runtime_constraints),
        startup_truth_snapshot=_default_truth_snapshot(
            strategy_symbol=strategy_symbol,
            engine_type="ML",
            ticks_per_bar=resolved_ticks_per_bar,
            release_stage=release_stage,
            live_trading_approved=False,
            evaluator_hash=evaluator_hash,
            logic_hash=logic_hash,
        ),
        replay_parity_reference=replay_parity_reference,
    )


def create_rule_manifest(
    *,
    strategy_symbol: str,
    rule_family: str,
    rule_params: dict[str, Any],
    dataset_path: str | Path,
    ticks_per_bar: int,
    holdout_start_utc: str | None = None,
    cost_model: CostModel | dict[str, Any],
    threshold_policy: ThresholdPolicy | dict[str, Any],
    runtime_constraints: RuntimeConstraints | dict[str, Any],
    alpha_gate: AlphaGateSpec | dict[str, Any] | None = None,
    git_commit: str = "unknown",
    release_stage: ReleaseStage = "research",
    evaluator_hash: str = "",
    logic_hash: str = "",
    replay_parity_reference: str = "",
    startup_truth_snapshot: dict[str, Any] | None = None,
) -> SelectorManifest:
    dataset_id = dataset_id_for_path(dataset_path)
    truth_snapshot = dict(startup_truth_snapshot or {})
    truth_snapshot.update(
        _default_truth_snapshot(
            strategy_symbol=strategy_symbol,
            engine_type="RULE",
            ticks_per_bar=ticks_per_bar,
            release_stage=release_stage,
            live_trading_approved=False,
            evaluator_hash=evaluator_hash,
            logic_hash=logic_hash,
        )
    )
    truth_snapshot["rule_family"] = rule_family
    truth_snapshot["rule_params"] = dict(rule_params)
    return SelectorManifest(
        manifest_version=MANIFEST_VERSION,
        strategy_symbol=strategy_symbol.upper(),
        engine_type="RULE",
        release_stage=release_stage,
        live_trading_approved=False,
        rule_family=rule_family,
        evaluator_hash=evaluator_hash,
        logic_hash=logic_hash,
        feature_schema=list(FEATURE_COLS),
        feature_schema_hash=_content_hash(list(FEATURE_COLS)),
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_id,
        sklearn_version=sklearn.__version__,
        created_from_git_commit=git_commit,
        bar_construction_ticks_per_bar=int(ticks_per_bar),
        ticks_per_bar=int(ticks_per_bar),
        holdout_start_utc=holdout_start_utc,
        cost_model=_normalize_mapping(cost_model),
        threshold_policy=_normalize_mapping(threshold_policy),
        runtime_constraints=_normalize_mapping(runtime_constraints),
        rule_params=dict(rule_params),
        alpha_gate=_normalize_mapping(alpha_gate),
        startup_truth_snapshot=truth_snapshot,
        replay_parity_reference=replay_parity_reference,
    )


def load_validated_selector_model(manifest: SelectorManifest | str | Path, *, expected_symbol: str) -> object:
    if isinstance(manifest, (str, Path)):
        manifest = load_selector_manifest(manifest, verify_manifest_hash=True)
    validate_selector_manifest(manifest, verify_manifest_hash=False)
    if manifest.engine_type != "ML":
        raise RuntimeError(f"load_validated_selector_model only supports ML manifests, got {manifest.engine_type}.")
    if manifest.strategy_symbol != expected_symbol.upper():
        raise RuntimeError(
            f"Selector symbol mismatch: manifest={manifest.strategy_symbol} runtime={expected_symbol.upper()}."
        )
    model_path = Path(manifest.model_path or "")
    if not model_path.exists():
        raise FileNotFoundError(f"Selector model path does not exist: {model_path}")
    expected_sha = manifest.model_sha256 or ""
    if expected_sha and _file_sha256(model_path) != expected_sha:
        raise RuntimeError("Selector model checksum mismatch. Refusing to load inference artifact.")
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise RuntimeError(f"Loaded selector model does not expose predict(): {type(model).__name__}")
    return model
