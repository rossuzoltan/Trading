from __future__ import annotations

import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
DOCS_DIR = ROOT_DIR / "docs"

DEFAULT_DATASET_CANDIDATES = (
    DATA_DIR / "DATA_CLEAN_VOLUME.csv",
    DATA_DIR / "FOREX_MULTI_SET.csv",
)

DEFAULT_MODEL_CANDIDATES = (
    MODELS_DIR / "model_eurusd_best.zip",
)

DEFAULT_MANIFEST_CANDIDATES = (
    MODELS_DIR / "artifact_manifest.json",
    MODELS_DIR / "artifact_manifest_EURUSD.json",
)


def ensure_runtime_dirs() -> None:
    for path in (DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_dataset_path(preferred: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(Path(preferred))
    candidates.extend(DEFAULT_DATASET_CANDIDATES)

    resolved = _first_existing(candidates)
    if resolved is not None:
        return resolved

    names = ", ".join(path.name for path in DEFAULT_DATASET_CANDIDATES)
    raise FileNotFoundError(
        f"No dataset found. Expected one of: {names}. "
        "Run download_dukascopy.py or build_volume_bars.py first."
    )


def resolve_model_path(preferred: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(Path(preferred))
    candidates.extend(DEFAULT_MODEL_CANDIDATES)

    resolved = _first_existing(candidates)
    if resolved is not None:
        return resolved

    names = ", ".join(path.name for path in DEFAULT_MODEL_CANDIDATES)
    raise FileNotFoundError(
        f"No trained model found. Expected one of: {names}. Run train_agent.py first."
    )


def resolve_scaler_path(
    symbol: str | None = None,
    preferred: str | Path | None = None,
    required: bool = True,
) -> Path | None:
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(Path(preferred))
    if symbol:
        candidates.append(MODELS_DIR / f"scaler_{symbol}.pkl")
    candidates.append(MODELS_DIR / "scaler_features.pkl")

    resolved = _first_existing(candidates)
    if resolved is not None or not required:
        return resolved

    names = ", ".join(path.name for path in candidates if path.name)
    raise FileNotFoundError(
        f"No scaler found. Expected one of: {names}. Run train_agent.py first."
    )


def resolve_manifest_path(
    symbol: str | None = None,
    preferred: str | Path | None = None,
) -> Path:
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(Path(preferred))
    if symbol:
        candidates.append(MODELS_DIR / f"artifact_manifest_{symbol.upper()}.json")
    candidates.extend(DEFAULT_MANIFEST_CANDIDATES)

    if symbol:
        expected_symbol = symbol.strip().upper()
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            manifest_symbol = str(raw.get("strategy_symbol", "")).strip().upper()
            if manifest_symbol == expected_symbol:
                return candidate
    else:
        resolved = _first_existing(candidates)
        if resolved is not None:
            return resolved

    names = ", ".join(path.name for path in candidates if path.name)
    raise FileNotFoundError(
        f"No artifact manifest found. Expected one of: {names}. Run train_agent.py first."
    )
