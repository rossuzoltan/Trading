from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to ensure infra modules can be imported from tools/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import subprocess
import pandas as pd
from artifact_manifest import (
    dataset_id_for_path,
    load_manifest,
    load_validated_model,
    load_validated_scaler,
    load_validated_vecnormalize,
)
from dataset_validation import validate_symbol_bar_spec
from feature_engine import FEATURE_COLS
from interpreter_guard import ensure_project_venv
from project_paths import (
    ROOT_DIR,
    DATA_DIR,
    MODELS_DIR,
    dataset_build_info_ticks_per_bar,
    list_manifest_paths,
    load_dataset_build_info,
    resolve_dataset_path,
    resolve_model_path,
    resolve_scaler_path,
    validate_dataset_bar_spec,
    validate_dataset_integrity,
)
from runtime_common import STATE_FEATURE_COUNT, deserialize_action_map
from selector_manifest import load_selector_manifest, validate_paper_live_candidate_manifest
from trading_config import resolve_bar_construction_ticks_per_bar

ensure_project_venv(project_root=ROOT, script_path=__file__)

REQUIRED_FILES = (
    ROOT_DIR / "README.md",
    ROOT_DIR / ".env.example",
    ROOT_DIR / "train_agent.py",
    ROOT_DIR / "evaluate_oos.py",
    ROOT_DIR / "live_bridge.py",
)

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "requests": "requests",
    "torch": "torch",
    "stable-baselines3": "stable_baselines3",
    "sb3-contrib": "sb3-contrib",
    "gymnasium": "gymnasium",
    "pandas-ta": "pandas-ta",
    "joblib": "joblib",
    "scikit-learn": "scikit-learn",
    "scipy": "scipy",
    "python-dotenv": "python-dotenv",
    "yfinance": "yfinance",
    "pyarrow": "pyarrow",
}

OPTIONAL_PACKAGES = {
    "MetaTrader5": "MetaTrader5",
}
DEFAULT_RUNTIME_MODEL_NAME = "models/model_<symbol>_best.zip"
DEFAULT_RUNTIME_SCALER_NAME = "models/scaler_<SYMBOL>.pkl"
RC1_REQUIRED_FILES = (
    "manifest.json",
    "baseline_scoreboard_rc1.json",
    "baseline_scoreboard_rc1.md",
    "release_notes_rc1.md",
)


def _print_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def _read_requirements() -> set[str]:
    entries: set[str] = set()
    for req_path in (ROOT_DIR / "Requirements.txt", ROOT_DIR / "requirements.project.txt"):
        if not req_path.exists():
            continue

        raw_bytes = req_path.read_bytes()
        text = raw_bytes.decode("utf-8", errors="ignore")
        if text.count("\x00") > max(len(text) // 10, 1):
            for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
                try:
                    text = raw_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

        for raw_line in text.replace("\x00", "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            normalized = (
                line.split("==")[0]
                .split(">=")[0]
                .split("<=")[0]
                .strip()
                .lower()
            )
            entries.add(normalized)
    return entries


def _check_file_presence() -> list[str]:
    issues: list[str] = []
    for path in REQUIRED_FILES:
        if not path.exists():
            issues.append(f"Missing file: {path.relative_to(ROOT_DIR)}")
    return issues


def _manifest_priority(path: Path, symbol: str) -> int:
    normalized_symbol = symbol.strip().upper()
    if path.name.lower() == f"artifact_manifest_{normalized_symbol.lower()}.json":
        return 0
    if path.name == "artifact_manifest.json":
        return 1
    return 2


def _select_runtime_manifests() -> tuple[list[tuple[Path, object]], list[str]]:
    selected: dict[str, tuple[int, Path, object]] = {}
    issues: list[str] = []

    for manifest_path in list_manifest_paths():
        try:
            manifest = load_manifest(manifest_path)
        except Exception as exc:
            issues.append(f"Invalid artifact manifest {_display_path(manifest_path)}: {exc}")
            continue

        symbol = str(getattr(manifest, "strategy_symbol", "")).strip().upper()
        if not symbol:
            issues.append(f"Artifact manifest {_display_path(manifest_path)} is missing strategy_symbol.")
            continue

        priority = _manifest_priority(manifest_path, symbol)
        current = selected.get(symbol)
        if current is None or priority < current[0]:
            selected[symbol] = (priority, manifest_path, manifest)

    bundles = [(path, manifest) for _, path, manifest in sorted(selected.values(), key=lambda item: item[2].strategy_symbol)]
    return bundles, issues


def _validate_runtime_manifest_bundle(
    manifest_path: Path,
    manifest,
    *,
    expected_ticks: int,
    expected_dataset_id: str,
) -> None:
    manifest_ticks = getattr(manifest, "bar_construction_ticks_per_bar", None) or getattr(manifest, "ticks_per_bar", None)
    if manifest_ticks is not None and int(manifest_ticks) != int(expected_ticks):
        raise RuntimeError(
            f"Manifest bar_construction_ticks_per_bar={int(manifest_ticks)} does not match active dataset spec {int(expected_ticks)}."
        )

    symbol = str(manifest.strategy_symbol).strip().upper()
    expected_action_map = deserialize_action_map(manifest.action_map)
    expected_observation_shape = list(
        getattr(manifest, "observation_shape", None) or [1, len(FEATURE_COLS) + STATE_FEATURE_COUNT]
    )
    load_validated_model(
        manifest,
        expected_symbol=symbol,
        expected_action_map=expected_action_map,
        expected_observation_shape=expected_observation_shape,
        expected_dataset_id=expected_dataset_id,
    )
    load_validated_scaler(
        manifest,
        expected_symbol=symbol,
        expected_action_map=expected_action_map,
        expected_observation_shape=expected_observation_shape,
        expected_dataset_id=expected_dataset_id,
    )
    load_validated_vecnormalize(
        manifest,
        expected_symbol=symbol,
        expected_action_map=expected_action_map,
        expected_observation_shape=expected_observation_shape,
        expected_dataset_id=expected_dataset_id,
    )


def _check_rl_runtime_assets(*, strict_runtime_assets: bool = False) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    dataset_path: Path | None = None
    expected_ticks: int | None = None
    expected_dataset_id: str | None = None

    try:
        dataset_path = resolve_dataset_path()
        expected_dataset_id = dataset_id_for_path(dataset_path)
        print(f"Dataset: {_display_path(dataset_path)}")
        expected_ticks = resolve_bar_construction_ticks_per_bar("BAR_SPEC_TICKS_PER_BAR", "TRADING_TICKS_PER_BAR")
        dataset_build_info = load_dataset_build_info(required=False)
        if dataset_build_info is None:
            issues.append("Dataset build metadata missing: data/dataset_build_info.json")
        else:
            actual_ticks = dataset_build_info_ticks_per_bar(dataset_build_info)
            if actual_ticks is not None:
                print(f"Bar spec: {actual_ticks} ticks/bar")
            try:
                validate_dataset_bar_spec(
                    dataset_path=dataset_path,
                    expected_ticks_per_bar=expected_ticks,
                    metadata_required=True,
                )
                integrity = validate_dataset_integrity(
                    dataset_path=dataset_path,
                    expected_ticks_per_bar=expected_ticks,
                    metadata_required=True,
                )
                symbols = ", ".join(integrity.get("symbols", []))
                print(f"Dataset integrity: OK ({symbols})")
            except RuntimeError as exc:
                issues.append(str(exc))
            try:
                dataset_frame = pd.read_csv(dataset_path, usecols=["Symbol", "Gmt time", "Volume"])
                for symbol in sorted(dataset_frame["Symbol"].astype(str).str.upper().unique().tolist()):
                    symbol_frame = dataset_frame[dataset_frame["Symbol"].astype(str).str.upper() == symbol].copy()
                    summary = validate_symbol_bar_spec(
                        symbol_frame,
                        expected_ticks_per_bar=expected_ticks,
                        symbol=symbol,
                    )
                    partial_rows = int(summary["partial_rows"])
                    if partial_rows:
                        print(
                            f"Bar spec audit {symbol}: {summary['exact_match_rows']} exact, "
                            f"{partial_rows} partial tail row(s)"
                        )
            except RuntimeError as exc:
                issues.append(str(exc))
    except FileNotFoundError as exc:
        issues.append(str(exc))

    manifest_bundles, manifest_issues = _select_runtime_manifests()
    issues.extend(manifest_issues)
    if manifest_bundles:
        if expected_ticks is not None and expected_dataset_id is not None:
            for manifest_path, manifest in manifest_bundles:
                symbol = str(manifest.strategy_symbol).strip().upper()
                try:
                    _validate_runtime_manifest_bundle(
                        manifest_path,
                        manifest,
                        expected_ticks=expected_ticks,
                        expected_dataset_id=expected_dataset_id,
                    )
                    print(f"Runtime bundle {symbol}: OK ({_display_path(manifest_path)})")
                except Exception as exc:
                    issues.append(
                        f"Runtime bundle {symbol} failed validation from {_display_path(manifest_path)}: {exc}"
                    )
        else:
            warnings.append("Artifact manifests were found, but runtime bundle validation was skipped because the active dataset could not be resolved.")
    else:
        message = (
            "No artifact manifest found. Expected a symbol-scoped manifest like "
            "models/artifact_manifest_EURUSD.json or the compatibility alias models/artifact_manifest.json."
        )
        if strict_runtime_assets:
            issues.append(message)
        else:
            warnings.append(message)

        model_path = resolve_model_path(required=False)
        if model_path is None:
            message = f"No trained model found. Expected a symbol-scoped artifact like {DEFAULT_RUNTIME_MODEL_NAME}."
            if strict_runtime_assets:
                issues.append(message)
            else:
                warnings.append(message)
        else:
            print(f"Model:   {_display_path(model_path)}")

        scaler_path = resolve_scaler_path(required=False)
        if scaler_path is None:
            message = (
                "No scaler found. Expected a symbol-scoped scaler like "
                f"{DEFAULT_RUNTIME_SCALER_NAME} or the compatibility alias models/scaler_features.pkl."
            )
            if strict_runtime_assets:
                issues.append(message)
            else:
                warnings.append(message)
        else:
            print(f"Scaler:  {_display_path(scaler_path)}")

    if not (DATA_DIR / "FOREX_MULTI_SET.csv").exists():
        message = "Compatibility dataset alias missing: data/FOREX_MULTI_SET.csv"
        if strict_runtime_assets:
            issues.append(message)
        else:
            warnings.append(message)

    if not list(MODELS_DIR.glob("*.zip")):
        message = "No zipped model artifacts found in models/."
        if strict_runtime_assets:
            issues.append(message)
        else:
            warnings.append(message)

    return issues, warnings


def _check_rc1_assets() -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    rc1_root = MODELS_DIR / "rc1"
    if not rc1_root.exists():
        return [f"RC1 root is missing: {_display_path(rc1_root)}"], warnings

    rc_dirs = sorted(path for path in rc1_root.iterdir() if path.is_dir())
    if not rc_dirs:
        return [f"No RC1 packs found under {_display_path(rc1_root)}"], warnings

    seen_symbols: set[str] = set()
    for pack_dir in rc_dirs:
        manifest_path = pack_dir / "manifest.json"
        if not manifest_path.exists():
            issues.append(f"RC1 pack missing manifest: {_display_path(manifest_path)}")
            continue
        try:
            manifest = load_selector_manifest(manifest_path, verify_manifest_hash=True)
            validate_paper_live_candidate_manifest(manifest, verify_manifest_hash=True)
        except Exception as exc:
            issues.append(f"RC1 pack {_display_path(pack_dir)} failed manifest validation: {exc}")
            continue

        seen_symbols.add(manifest.strategy_symbol)
        for filename in RC1_REQUIRED_FILES:
            required_path = pack_dir / filename
            if not required_path.exists():
                issues.append(f"RC1 pack {_display_path(pack_dir)} missing {filename}")

        ticks_per_bar = int(manifest.ticks_per_bar or manifest.bar_construction_ticks_per_bar or 0)
        dataset_path = None
        try:
            dataset_path = resolve_dataset_path(ticks_per_bar=ticks_per_bar)
            print(f"RC1 dataset {manifest.strategy_symbol}: {_display_path(dataset_path)}")
            metadata = load_dataset_build_info(required=False, ticks_per_bar=ticks_per_bar)
            if metadata is None:
                issues.append(
                    f"Dataset build metadata missing for {manifest.strategy_symbol} {ticks_per_bar} ticks/bar."
                )
            else:
                validate_dataset_bar_spec(
                    dataset_path=dataset_path,
                    expected_ticks_per_bar=ticks_per_bar,
                    metadata_required=True,
                )
                validate_dataset_integrity(
                    dataset_path=dataset_path,
                    expected_ticks_per_bar=ticks_per_bar,
                    metadata_required=True,
                    symbol=manifest.strategy_symbol,
                )
        except Exception as exc:
            issues.append(
                f"RC1 dataset validation failed for {manifest.strategy_symbol} {ticks_per_bar} ticks/bar: {exc}"
            )
            continue

        print(
            f"RC1 pack {manifest.strategy_symbol}: OK "
            f"({_display_path(pack_dir)} @ {ticks_per_bar} ticks/bar)"
        )

    approved_pairs = {"EURUSD", "GBPUSD"}
    missing_pairs = sorted(approved_pairs - seen_symbols)
    if missing_pairs:
        issues.append("Missing approved RC1 anchor packs: " + ", ".join(missing_pairs))

    return issues, warnings


def _check_runtime_assets(
    *,
    strict_runtime_assets: bool = False,
    mode: str = "rc1",
) -> tuple[list[str], list[str]]:
    if mode == "rl":
        return _check_rl_runtime_assets(strict_runtime_assets=strict_runtime_assets)
    return _check_rc1_assets()


def _check_venv_layout() -> list[str]:
    issues: list[str] = []
    venv_dir = ROOT_DIR / ".venv"
    if not venv_dir.exists():
        issues.append("Virtual environment missing: .venv/")
        return issues

    python_exe = venv_dir / "Scripts" / "python.exe"
    if not python_exe.exists():
        issues.append("Missing interpreter: .venv/Scripts/python.exe")
        return issues

    checks = {
        "pip": "import pip",
        "pandas": "import pandas",
    }
    for name, snippet in checks.items():
        result = subprocess.run(
            [str(python_exe), "-c", snippet],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip() or "unknown error"
            issues.append(f"Broken {name} install in .venv: {details}")

    return issues


def _check_requirements() -> list[str]:
    issues: list[str] = []
    entries = _read_requirements()
    if not entries:
        return ["Requirements.txt and requirements.project.txt are missing or empty."]

    for display_name, package_key in REQUIRED_PACKAGES.items():
        if package_key.lower() not in entries:
            issues.append(f"Requirements.txt missing package: {display_name}")

    optional_missing = [
        display_name
        for display_name, package_key in OPTIONAL_PACKAGES.items()
        if package_key.lower() not in entries
    ]
    if optional_missing:
        print("Optional packages not declared: " + ", ".join(optional_missing))

    return issues


def main(*, strict_runtime_assets: bool = False, mode: str = "rc1") -> int:
    print("Trading Project Health Check")
    print("============================")

    all_issues: list[str] = []

    _print_header("File Presence")
    file_issues = _check_file_presence()
    all_issues.extend(file_issues)
    if file_issues:
        for issue in file_issues:
            print(f"[MISSING] {issue}")
    else:
        print("Core repo files are present.")

    _print_header("Runtime Assets")
    asset_issues, asset_warnings = _check_runtime_assets(
        strict_runtime_assets=strict_runtime_assets,
        mode=mode,
    )
    all_issues.extend(asset_issues)
    if asset_issues:
        for issue in asset_issues:
            print(f"[ISSUE] {issue}")
    for warning in asset_warnings:
        print(f"[WARN] {warning}")

    _print_header("Virtual Environment")
    venv_issues = _check_venv_layout()
    all_issues.extend(venv_issues)
    if venv_issues:
        for issue in venv_issues:
            print(f"[ISSUE] {issue}")
    else:
        print("Virtual environment layout looks intact.")

    _print_header("Requirements Audit")
    req_issues = _check_requirements()
    all_issues.extend(req_issues)
    if req_issues:
        for issue in req_issues:
            print(f"[ISSUE] {issue}")
    else:
        print("Requirement files cover the core runtime dependencies.")

    _print_header("Summary")
    if all_issues:
        print(f"Found {len(all_issues)} issue(s).")
        if venv_issues:
            print("Recommended next step: repair .venv, then rerun this script.")
        else:
            print("Recommended next step: address the runtime asset or dependency issues above, then rerun this script.")
    else:
        print("No obvious file or dependency issues detected.")
    if asset_warnings:
        print(f"Runtime asset warnings: {len(asset_warnings)}")
    return 1 if all_issues else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading project health checks")
    parser.add_argument(
        "--mode",
        choices=("rc1", "rl"),
        default="rc1",
        help="Select the active operational surface. Default: rc1.",
    )
    parser.add_argument(
        "--strict-runtime-assets",
        action="store_true",
        help="Treat missing model/scaler/compatibility dataset artifacts as hard issues.",
    )
    args = parser.parse_args()
    raise SystemExit(main(strict_runtime_assets=args.strict_runtime_assets, mode=args.mode))
