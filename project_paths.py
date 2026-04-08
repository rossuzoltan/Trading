from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
DOCS_DIR = ROOT_DIR / "docs"
LOGS_DIR = ROOT_DIR / "logs"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
SHADOW_ARTIFACTS_DIR = ARTIFACTS_DIR / "shadow"
GATE_ARTIFACTS_DIR = ARTIFACTS_DIR / "gates"
DATASET_BUILD_INFO_PATH = DATA_DIR / "dataset_build_info.json"
LEGACY_DATASET_QC_REPORT_PATH = DATA_DIR / "volume_bars_qc_report.json"
REQUIRED_DATASET_COLUMNS = (
    "Gmt time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Symbol",
)

DEFAULT_DATASET_CANDIDATES = (
    DATA_DIR / "DATA_CLEAN_VOLUME.csv",
    DATA_DIR / "FOREX_MULTI_SET.csv",
)
DEFAULT_MODEL_GLOB = "model_*_best.zip"
DEFAULT_SCALER_GLOB = "scaler_*.pkl"
DEFAULT_SYMBOL_MANIFEST_GLOB = "artifact_manifest_*.json"

DEFAULT_MODEL_CANDIDATES = (
    MODELS_DIR / "model_eurusd_best.zip",
)

DEFAULT_MANIFEST_CANDIDATES = (
    MODELS_DIR / "artifact_manifest.json",
    MODELS_DIR / "artifact_manifest_EURUSD.json",
)


def ensure_runtime_dirs() -> None:
    for path in (DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, DOCS_DIR, LOGS_DIR, ARTIFACTS_DIR, SHADOW_ARTIFACTS_DIR, GATE_ARTIFACTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def shadow_artifact_dir(
    symbol: str,
    manifest_hash: str,
    *,
    base_dir: str | Path | None = None,
) -> Path:
    root = Path(base_dir) if base_dir is not None else SHADOW_ARTIFACTS_DIR
    normalized_symbol = symbol.strip().upper() or "UNKNOWN"
    normalized_hash = (manifest_hash or "unknown").strip() or "unknown"
    return root / normalized_symbol / normalized_hash


def gate_artifact_dir(
    symbol: str,
    manifest_hash: str,
    *,
    base_dir: str | Path | None = None,
) -> Path:
    root = Path(base_dir) if base_dir is not None else GATE_ARTIFACTS_DIR
    normalized_symbol = symbol.strip().upper() or "UNKNOWN"
    normalized_hash = (manifest_hash or "unknown").strip() or "unknown"
    return root / normalized_symbol / normalized_hash


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def list_model_paths() -> list[Path]:
    return sorted(MODELS_DIR.glob(DEFAULT_MODEL_GLOB))


def list_scaler_paths() -> list[Path]:
    return sorted(MODELS_DIR.glob(DEFAULT_SCALER_GLOB))


def list_manifest_paths() -> list[Path]:
    ordered_candidates = [MODELS_DIR / "artifact_manifest.json", *sorted(MODELS_DIR.glob(DEFAULT_SYMBOL_MANIFEST_GLOB))]
    seen: set[Path] = set()
    discovered: list[Path] = []
    for path in ordered_candidates:
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        discovered.append(path)
    return discovered


def resolve_dataset_path(preferred: str | Path | None = None, *, ticks_per_bar: int | None = None) -> Path:
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(Path(preferred))
    
    if ticks_per_bar is not None:
        candidates.append(DATA_DIR / f"DATA_CLEAN_VOLUME_{int(ticks_per_bar)}.csv")
        candidates.append(DATA_DIR / f"FOREX_MULTI_SET_{int(ticks_per_bar)}.csv")
        
    candidates.extend(DEFAULT_DATASET_CANDIDATES)

    resolved = _first_existing(candidates)
    if resolved is not None:
        return resolved

    names = ", ".join(path.name for path in candidates if path.name)
    raise FileNotFoundError(
        f"No dataset found. Expected one of: {names}. "
        "Run download_dukascopy.py or build_volume_bars.py with --ticks-per-bar first."
    )


def resolve_dataset_build_info_path(
    preferred: str | Path | None = None,
    *,
    required: bool = False,
    ticks_per_bar: int | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(Path(preferred))
    
    if ticks_per_bar is not None:
        candidates.append(DATA_DIR / f"dataset_build_info_{int(ticks_per_bar)}.json")
        candidates.append(DATA_DIR / f"volume_bars_qc_report_{int(ticks_per_bar)}.json")
        
    candidates.extend((DATASET_BUILD_INFO_PATH, LEGACY_DATASET_QC_REPORT_PATH))

    resolved = _first_existing(candidates)
    if resolved is not None or not required:
        return resolved

    names = ", ".join(path.name for path in candidates if path.name)
    raise FileNotFoundError(
        f"No dataset build metadata found. Expected one of: {names}. "
        "Run build_volume_bars.py with --ticks-per-bar first."
    )


def load_dataset_build_info(
    preferred: str | Path | None = None,
    *,
    required: bool = False,
    ticks_per_bar: int | None = None,
) -> dict | None:
    path = resolve_dataset_build_info_path(preferred=preferred, required=required, ticks_per_bar=ticks_per_bar)
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_build_info_ticks_per_bar(build_info: dict | None) -> int | None:
    if not isinstance(build_info, dict):
        return None
    raw = build_info.get("bar_construction_ticks_per_bar", build_info.get("ticks_per_bar"))
    if raw in (None, ""):
        return None
    return int(raw)


def parquet_row_count(path: str | Path) -> int:
    parquet_path = Path(path)
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(parquet_path).metadata.num_rows)
    except Exception:
        frame = pd.read_parquet(parquet_path)
        return int(len(frame))


def validate_dataset_bar_spec(
    *,
    dataset_path: str | Path,
    expected_ticks_per_bar: int,
    metadata_path: str | Path | None = None,
    metadata_required: bool = False,
) -> dict | None:
    build_info_path = resolve_dataset_build_info_path(
        preferred=metadata_path, 
        required=metadata_required,
        ticks_per_bar=expected_ticks_per_bar
    )
    if build_info_path is None:
        return None

    build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
    actual_ticks_per_bar = dataset_build_info_ticks_per_bar(build_info)
    if actual_ticks_per_bar is None:
        if metadata_required:
            raise RuntimeError(
                f"Dataset build metadata {build_info_path} does not declare bar_construction_ticks_per_bar. "
                "Rebuild the dataset with build_volume_bars.py so the active bar spec is recorded."
            )
        return build_info

    if int(actual_ticks_per_bar) != int(expected_ticks_per_bar):
        raise RuntimeError(
            f"Dataset {Path(dataset_path)} was built with bar_construction_ticks_per_bar={actual_ticks_per_bar}, "
            f"but the current run expects {int(expected_ticks_per_bar)}. "
            "Rebuild the dataset with matching --ticks-per-bar or align BAR_SPEC_TICKS_PER_BAR / "
            "TRADING_TICKS_PER_BAR before continuing."
        )
    return build_info


def validate_dataset_integrity(
    *,
    dataset_path: str | Path,
    expected_ticks_per_bar: int,
    metadata_path: str | Path | None = None,
    metadata_required: bool = False,
    symbol: str | None = None,
    raw_ticks_dir: str | Path | None = None,
) -> dict:
    dataset = Path(dataset_path)
    build_info = validate_dataset_bar_spec(
        dataset_path=dataset,
        expected_ticks_per_bar=expected_ticks_per_bar,
        metadata_path=metadata_path,
        metadata_required=metadata_required,
    )

    header = pd.read_csv(dataset, nrows=0)
    missing_columns = [column for column in REQUIRED_DATASET_COLUMNS if column not in header.columns]
    if missing_columns:
        raise RuntimeError(
            f"Dataset {dataset} is missing required columns: {', '.join(missing_columns)}"
        )

    frame = pd.read_csv(dataset, usecols=list(REQUIRED_DATASET_COLUMNS), low_memory=False)
    frame["Symbol"] = frame["Symbol"].astype(str).str.upper()
    frame["Gmt time"] = pd.to_datetime(frame["Gmt time"], utc=True, errors="coerce")
    invalid_timestamp_rows = int(frame["Gmt time"].isna().sum())
    if invalid_timestamp_rows:
        raise RuntimeError(
            f"Dataset {dataset} contains {invalid_timestamp_rows} rows with invalid Gmt time values."
        )

    if symbol:
        target_symbol = symbol.strip().upper()
        frame = frame.loc[frame["Symbol"] == target_symbol].copy()
        if frame.empty:
            raise RuntimeError(f"Dataset {dataset} does not contain rows for symbol {target_symbol}.")
    else:
        target_symbol = None

    duplicate_rows = int(frame.duplicated(subset=["Gmt time", "Symbol"]).sum())
    if duplicate_rows:
        raise RuntimeError(
            f"Dataset {dataset} contains {duplicate_rows} duplicate (Gmt time, Symbol) rows."
        )

    actual_counts = frame["Symbol"].value_counts().sort_index().to_dict()
    metadata_counts = (
        {
            str(raw_symbol).upper(): int(count)
            for raw_symbol, count in build_info.get("symbol_counts", {}).items()
        }
        if isinstance(build_info, dict)
        else {}
    )
    if target_symbol is not None:
        metadata_counts = {target_symbol: metadata_counts.get(target_symbol, 0)}
    if metadata_counts and metadata_counts != {symbol: int(count) for symbol, count in actual_counts.items()}:
        raise RuntimeError(
            f"Dataset {dataset} row counts do not match dataset_build_info.json for the active symbol set. "
            f"metadata={metadata_counts} actual={actual_counts}"
        )

    raw_ticks_root = Path(raw_ticks_dir) if raw_ticks_dir is not None else DATA_DIR
    symbol_reports: dict[str, dict] = {}
    issues: list[str] = []
    for dataset_symbol, sdf in frame.groupby("Symbol", sort=True):
        ordered = sdf.sort_values("Gmt time")
        delta = ordered["Gmt time"].diff().dt.total_seconds().dropna()
        report = {
            "rows": int(len(ordered)),
            "start_utc": ordered["Gmt time"].iloc[0].isoformat(),
            "end_utc": ordered["Gmt time"].iloc[-1].isoformat(),
            "gap_gt_6h": int((delta > 21600).sum()),
            "gap_gt_24h": int((delta > 86400).sum()),
            "max_gap_hours": float(delta.max() / 3600.0) if not delta.empty else 0.0,
            "median_gap_minutes": float(delta.median() / 60.0) if not delta.empty else 0.0,
            "file_order_monotonic": bool(sdf["Gmt time"].is_monotonic_increasing),
        }
        if not report["file_order_monotonic"]:
            issues.append(f"{dataset_symbol}: dataset rows are not ordered by Gmt time.")

        tick_path = raw_ticks_root / f"{dataset_symbol}_ticks.parquet"
        if tick_path.exists():
            tick_rows = parquet_row_count(tick_path)
            expected_bars = (tick_rows + int(expected_ticks_per_bar) - 1) // int(expected_ticks_per_bar)
            report["raw_tick_rows"] = int(tick_rows)
            report["expected_bars_from_ticks"] = int(expected_bars)
            report["bar_count_matches_raw_ticks"] = bool(expected_bars == len(ordered))
            if expected_bars != len(ordered):
                issues.append(
                    f"{dataset_symbol}: dataset has {len(ordered)} bars but raw ticks imply {expected_bars} bars "
                    f"for {int(expected_ticks_per_bar)} ticks/bar."
                )
        symbol_reports[dataset_symbol] = report

    if issues:
        raise RuntimeError("Dataset integrity validation failed: " + " | ".join(issues))

    return {
        "passed": True,
        "dataset_path": str(dataset),
        "expected_ticks_per_bar": int(expected_ticks_per_bar),
        "symbol_counts": {symbol_key: int(count) for symbol_key, count in actual_counts.items()},
        "duplicate_symbol_timestamp_rows": duplicate_rows,
        "symbols": sorted(actual_counts.keys()),
        "symbol_reports": symbol_reports,
        "metadata_path": str(resolve_dataset_build_info_path(preferred=metadata_path, required=metadata_required, ticks_per_bar=expected_ticks_per_bar))
        if resolve_dataset_build_info_path(preferred=metadata_path, required=metadata_required, ticks_per_bar=expected_ticks_per_bar) is not None
        else None,
    }


def resolve_model_path(
    preferred: str | Path | None = None,
    *,
    symbol: str | None = None,
    required: bool = True,
) -> Path | None:
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(Path(preferred))
    normalized_symbol = symbol.strip().upper() if symbol is not None else ""
    if normalized_symbol:
        candidates.append(MODELS_DIR / f"model_{normalized_symbol.lower()}_best.zip")
    else:
        candidates.extend(DEFAULT_MODEL_CANDIDATES)
        candidates.extend(list_model_paths())

    resolved = _first_existing(candidates)
    if resolved is not None or not required:
        return resolved

    names = ", ".join(dict.fromkeys(path.name for path in candidates if path.name))
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
    normalized_symbol = symbol.strip().upper() if symbol is not None else ""
    if normalized_symbol:
        candidates.append(MODELS_DIR / f"scaler_{normalized_symbol}.pkl")
    candidates.append(MODELS_DIR / "scaler_features.pkl")
    if not normalized_symbol:
        candidates.extend(list_scaler_paths())

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
