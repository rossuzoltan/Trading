from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import joblib
import sklearn
import stable_baselines3
from sb3_contrib import MaskablePPO, __version__ as sb3_contrib_version
from stable_baselines3.common.vec_env import VecNormalize

from feature_engine import FEATURE_COLS
from runtime_common import ActionSpec, serialize_action_map


MANIFEST_VERSION = "1"
DEFAULT_MANIFEST_NAME = "artifact_manifest.json"


@dataclass(frozen=True)
class ArtifactManifest:
    manifest_version: str
    strategy_symbol: str
    model_path: str
    scaler_path: str
    model_version: str
    model_sha256: str
    scaler_sha256: str
    feature_columns: list[str]
    observation_shape: list[int]
    action_map: list[dict[str, float | int | str | None]]
    dataset_id: str
    sb3_version: str
    sb3_contrib_version: str
    sklearn_version: str
    bar_construction_ticks_per_bar: int | None = None
    ticks_per_bar: int | None = None
    vecnormalize_path: str | None = None
    vecnormalize_sha256: str | None = None
    holdout_start_utc: str | None = None
    training_diagnostics_path: str | None = None
    execution_cost_profile: dict[str, float | int | str | bool | None] | None = None
    reward_profile: dict[str, float | int | str | bool | None] | None = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_id_for_path(path: str | Path) -> str:
    return _file_sha256(Path(path))


def save_manifest(manifest: ArtifactManifest, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return out_path


def load_manifest(path: str | Path) -> ArtifactManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "bar_construction_ticks_per_bar" not in raw and "ticks_per_bar" in raw:
        raw["bar_construction_ticks_per_bar"] = raw.get("ticks_per_bar")
    if "ticks_per_bar" not in raw and "bar_construction_ticks_per_bar" in raw:
        raw["ticks_per_bar"] = raw.get("bar_construction_ticks_per_bar")
    manifest = ArtifactManifest(**raw)
    if manifest.manifest_version != MANIFEST_VERSION:
        raise RuntimeError(
            f"Artifact manifest version mismatch: expected {MANIFEST_VERSION}, got {manifest.manifest_version}."
        )
    return manifest


def create_manifest(
    *,
    strategy_symbol: str,
    model_path: str | Path,
    scaler_path: str | Path,
    vecnormalize_path: str | Path | None = None,
    holdout_start_utc: str | None = None,
    training_diagnostics_path: str | Path | None = None,
    model_version: str,
    feature_columns: Sequence[str],
    observation_shape: Sequence[int],
    action_map: Sequence[ActionSpec],
    dataset_path: str | Path,
    bar_construction_ticks_per_bar: int | None = None,
    ticks_per_bar: int | None = None,
    execution_cost_profile: dict[str, float | int | str | bool | None] | None = None,
    reward_profile: dict[str, float | int | str | bool | None] | None = None,
) -> ArtifactManifest:
    model_path = str(Path(model_path))
    scaler_path = str(Path(scaler_path))
    dataset_path = str(Path(dataset_path))
    vecnormalize_str = str(Path(vecnormalize_path)) if vecnormalize_path is not None else None
    training_diag_str = (
        str(Path(training_diagnostics_path))
        if training_diagnostics_path is not None
        else None
    )
    resolved_ticks_per_bar = (
        int(bar_construction_ticks_per_bar)
        if bar_construction_ticks_per_bar is not None
        else int(ticks_per_bar)
        if ticks_per_bar is not None
        else None
    )
    return ArtifactManifest(
        manifest_version=MANIFEST_VERSION,
        strategy_symbol=strategy_symbol.upper(),
        model_path=model_path,
        scaler_path=scaler_path,
        model_version=model_version,
        model_sha256=_file_sha256(Path(model_path)),
        scaler_sha256=_file_sha256(Path(scaler_path)),
        feature_columns=list(feature_columns),
        observation_shape=[int(v) for v in observation_shape],
        action_map=serialize_action_map(action_map),
        dataset_id=dataset_id_for_path(dataset_path),
        sb3_version=stable_baselines3.__version__,
        sb3_contrib_version=sb3_contrib_version,
        sklearn_version=sklearn.__version__,
        bar_construction_ticks_per_bar=resolved_ticks_per_bar,
        ticks_per_bar=resolved_ticks_per_bar,
        vecnormalize_path=vecnormalize_str,
        vecnormalize_sha256=_file_sha256(Path(vecnormalize_str)) if vecnormalize_str is not None else None,
        holdout_start_utc=holdout_start_utc,
        training_diagnostics_path=training_diag_str,
        execution_cost_profile=dict(execution_cost_profile) if execution_cost_profile is not None else None,
        reward_profile=dict(reward_profile) if reward_profile is not None else None,
    )


def _validate_common(
    manifest: ArtifactManifest,
    *,
    expected_symbol: str,
    expected_action_map: Sequence[ActionSpec],
    expected_observation_shape: Sequence[int],
    expected_dataset_id: str | None,
) -> None:
    if manifest.strategy_symbol != expected_symbol.upper():
        raise RuntimeError(
            f"Artifact symbol mismatch: manifest={manifest.strategy_symbol}, runtime={expected_symbol.upper()}."
        )
    if manifest.feature_columns != list(FEATURE_COLS):
        raise RuntimeError("Feature list mismatch between manifest and runtime FEATURE_COLS.")
    if manifest.observation_shape != [int(v) for v in expected_observation_shape]:
        raise RuntimeError(
            f"Observation shape mismatch: manifest={manifest.observation_shape}, "
            f"runtime={[int(v) for v in expected_observation_shape]}."
        )
    if manifest.action_map != serialize_action_map(expected_action_map):
        raise RuntimeError("Action map mismatch between manifest and runtime policy mapping.")
    if manifest.sb3_version != stable_baselines3.__version__:
        raise RuntimeError(
            f"stable_baselines3 version mismatch: manifest={manifest.sb3_version}, "
            f"runtime={stable_baselines3.__version__}."
        )
    if manifest.sb3_contrib_version != sb3_contrib_version:
        raise RuntimeError(
            f"sb3-contrib version mismatch: manifest={manifest.sb3_contrib_version}, runtime={sb3_contrib_version}."
        )
    if manifest.sklearn_version != sklearn.__version__:
        raise RuntimeError(
            f"scikit-learn version mismatch: manifest={manifest.sklearn_version}, runtime={sklearn.__version__}."
        )
    if expected_dataset_id is not None and manifest.dataset_id != expected_dataset_id:
        raise RuntimeError(
            f"Dataset id mismatch: manifest={manifest.dataset_id}, runtime={expected_dataset_id}."
        )


def load_validated_scaler(
    manifest: ArtifactManifest,
    *,
    expected_symbol: str,
    expected_action_map: Sequence[ActionSpec],
    expected_observation_shape: Sequence[int],
    expected_dataset_id: str | None = None,
) -> object:
    _validate_common(
        manifest,
        expected_symbol=expected_symbol,
        expected_action_map=expected_action_map,
        expected_observation_shape=expected_observation_shape,
        expected_dataset_id=expected_dataset_id,
    )
    scaler_path = Path(manifest.scaler_path)
    if _file_sha256(scaler_path) != manifest.scaler_sha256:
        raise RuntimeError("Scaler checksum mismatch. Refusing to load inference artifacts.")
    scaler = joblib.load(scaler_path)
    if scaler.__class__.__name__ != "StandardScaler":
        raise RuntimeError(f"Unexpected scaler type '{scaler.__class__.__name__}'.")
    return scaler


def load_validated_model(
    manifest: ArtifactManifest,
    *,
    expected_symbol: str,
    expected_action_map: Sequence[ActionSpec],
    expected_observation_shape: Sequence[int],
    expected_dataset_id: str | None = None,
) -> MaskablePPO:
    _validate_common(
        manifest,
        expected_symbol=expected_symbol,
        expected_action_map=expected_action_map,
        expected_observation_shape=expected_observation_shape,
        expected_dataset_id=expected_dataset_id,
    )
    model_path = Path(manifest.model_path)
    if _file_sha256(model_path) != manifest.model_sha256:
        raise RuntimeError("Model checksum mismatch. Refusing to load inference artifacts.")
    if not manifest.model_version:
        raise RuntimeError("Artifact manifest is missing model_version.")
    model = MaskablePPO.load(str(model_path), device="cpu")
    model_obs_shape = [int(v) for v in model.observation_space.shape]
    if model_obs_shape != manifest.observation_shape:
        raise RuntimeError(
            f"Loaded model observation shape mismatch: model={model_obs_shape}, "
            f"manifest={manifest.observation_shape}."
        )
    model_action_count = int(model.action_space.n)
    expected_action_count = len(expected_action_map)
    if model_action_count != expected_action_count:
        raise RuntimeError(
            f"Loaded model action count mismatch: model={model_action_count}, runtime={expected_action_count}."
        )
    return model


def load_validated_vecnormalize(
    manifest: ArtifactManifest,
    *,
    expected_symbol: str,
    expected_action_map: Sequence[ActionSpec],
    expected_observation_shape: Sequence[int],
    expected_dataset_id: str | None = None,
) -> VecNormalize | None:
    _validate_common(
        manifest,
        expected_symbol=expected_symbol,
        expected_action_map=expected_action_map,
        expected_observation_shape=expected_observation_shape,
        expected_dataset_id=expected_dataset_id,
    )
    if manifest.vecnormalize_path is None:
        return None
    if not manifest.vecnormalize_sha256:
        raise RuntimeError("Artifact manifest is missing vecnormalize checksum.")
    vecnormalize_path = Path(manifest.vecnormalize_path)
    if _file_sha256(vecnormalize_path) != manifest.vecnormalize_sha256:
        raise RuntimeError("VecNormalize checksum mismatch. Refusing to load inference artifacts.")
    with vecnormalize_path.open("rb") as handle:
        vecnormalize = pickle.load(handle)
    if not isinstance(vecnormalize, VecNormalize):
        raise RuntimeError(f"Unexpected vecnormalize type '{vecnormalize.__class__.__name__}'.")
    vecnormalize.training = False
    return vecnormalize
