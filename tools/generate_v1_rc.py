from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_research import fit_baseline_alpha_gate, save_baseline_alpha_gate
from feature_engine import FEATURE_COLS, WARMUP_BARS, _compute_raw
from project_paths import resolve_dataset_path
from selector_manifest import (
    AlphaGateSpec,
    CostModel,
    RuntimeConstraints,
    ThresholdPolicy,
    _file_sha256,
    create_rule_manifest,
    load_selector_manifest,
    save_selector_manifest,
)
from train_agent import HOLDOUT_FRAC

log = logging.getLogger("generate_v1_rc")
RC_ROOT = ROOT / "models" / "rc1"

RC_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "symbol": "EURUSD",
        "ticks_per_bar": 5000,
        "name": "eurusd_5k_v1_mr_rc1",
        "rule_family": "mean_reversion",
        "rule_params": {
            "threshold": 1.5,
            "sl_value": 1.5,
            "tp_value": 3.0,
            "max_spread_z": 0.5,
            "max_time_delta_z": 2.0,
            "max_abs_ma20_slope": 0.15,
            "max_abs_ma50_slope": 0.08,
        },
        "spread_limit_pips": 1.0,
        "rollover_block_utc_hours": [21, 22, 23, 0],
        "allowed_sessions": ["London", "London/NY", "NY"],
        "alpha_gate": {
            "enabled": False,
            "model_preference": "xgboost_pair",
            "horizon_bars": 25,
            "probability_threshold": 0.53,
            "probability_margin": 0.03,
            "min_edge_pips": 0.0,
        },
    },
    {
        "symbol": "GBPUSD",
        "ticks_per_bar": 10000,
        "name": "gbpusd_10k_v1_mr_rc1",
        "rule_family": "mean_reversion",
        "rule_params": {
            "threshold": 1.75,
            "sl_value": 1.5,
            "tp_value": 3.0,
            "max_spread_z": 0.75,
            "max_time_delta_z": 2.0,
            "max_abs_ma20_slope": 0.15,
            "max_abs_ma50_slope": 0.08,
        },
        "spread_limit_pips": 1.25,
        "rollover_block_utc_hours": [21, 22, 23, 0],
        "allowed_sessions": ["London", "London/NY", "NY"],
        "alpha_gate": {
            "enabled": True,
            "model_preference": "xgboost_pair",
            "horizon_bars": 25,
            "probability_threshold": 0.61,
            "probability_margin": 0.01,
            "min_edge_pips": 0.0,
        },
    },
)


def _current_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = (result.stdout or "").strip()
    return commit or "unknown"


def _truth_snapshot(
    config: dict[str, Any],
    *,
    dataset_path: Path,
    evaluator_hash: str,
    logic_hash: str,
    holdout_start_utc: str,
    alpha_gate_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "release_name": config["name"],
        "strategy_symbol": config["symbol"],
        "rule_family": config["rule_family"],
        "rule_params": dict(config["rule_params"]),
        "ticks_per_bar": int(config["ticks_per_bar"]),
        "dataset_path": str(dataset_path),
        "holdout_start_utc": str(holdout_start_utc),
        "release_stage": "paper_live_candidate",
        "live_trading_approved": False,
        "evaluator_hash": evaluator_hash,
        "logic_hash": logic_hash,
        "alpha_gate_enabled": bool(alpha_gate_payload.get("enabled", False)),
        "alpha_gate_model_kind": alpha_gate_payload.get("model_kind"),
        "alpha_gate_fit_profit_factor": float(alpha_gate_payload.get("fit_profit_factor", 0.0) or 0.0),
        "alpha_gate_fit_expectancy_usd": float(alpha_gate_payload.get("fit_expectancy_usd", 0.0) or 0.0),
    }


def _load_symbol_feature_frame(dataset_path: Path, *, symbol: str) -> pd.DataFrame:
    raw = pd.read_csv(dataset_path, low_memory=False, parse_dates=["Gmt time"])
    raw = raw.loc[raw["Symbol"].astype(str).str.upper() == symbol.upper()].copy()
    raw["Gmt time"] = pd.to_datetime(raw["Gmt time"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["Gmt time"]).set_index("Gmt time").sort_index()
    if len(raw) <= WARMUP_BARS + 25:
        raise RuntimeError(f"Insufficient rows to build feature frame for {symbol}: {len(raw)}")
    featured = _compute_raw(raw).dropna(subset=list(FEATURE_COLS))
    if featured.empty:
        raise RuntimeError(f"Feature frame is empty for {symbol}")
    return featured


def _compute_holdout_start_utc(featured: pd.DataFrame) -> str:
    holdout_rows = int(len(featured) * float(HOLDOUT_FRAC))
    split_pos = max(len(featured) - holdout_rows, 1)
    split_pos = min(split_pos, len(featured) - 1)
    return pd.Timestamp(featured.index[split_pos]).isoformat()


def _fit_and_store_alpha_gate(
    *,
    config: dict[str, Any],
    featured: pd.DataFrame,
    holdout_start_utc: str,
    out_dir: Path,
) -> dict[str, Any]:
    alpha_cfg = dict(config.get("alpha_gate") or {})
    if not bool(alpha_cfg.get("enabled", False)):
        return {}
    holdout_start = pd.Timestamp(holdout_start_utc)
    train_frame = featured.loc[featured.index < holdout_start].copy()
    if train_frame.empty:
        return {}

    gate = fit_baseline_alpha_gate(
        symbol=str(config["symbol"]).upper(),
        train_frame=train_frame,
        feature_cols=list(FEATURE_COLS),
        horizon_bars=int(alpha_cfg.get("horizon_bars", 25)),
        commission_per_lot=7.0,
        slippage_pips=0.25,
        min_edge_pips=float(alpha_cfg.get("min_edge_pips", 0.0)),
        probability_threshold=float(alpha_cfg.get("probability_threshold", 0.55)),
        probability_margin=float(alpha_cfg.get("probability_margin", 0.05)),
        model_preference=str(alpha_cfg.get("model_preference", "auto")),
    )
    if gate is None or not bool(gate.fit_quality_passed):
        log.warning("AlphaGate fit skipped for %s (insufficient quality).", config["symbol"])
        return {}

    alpha_path = out_dir / "alpha_gate.joblib"
    save_baseline_alpha_gate(gate, alpha_path)
    return {
        "enabled": True,
        "model_path": str(alpha_path),
        "model_sha256": _file_sha256(alpha_path),
        "probability_threshold": float(alpha_cfg.get("probability_threshold", gate.probability_threshold)),
        "probability_margin": float(alpha_cfg.get("probability_margin", gate.probability_margin)),
        "min_edge_pips": float(alpha_cfg.get("min_edge_pips", gate.min_edge_pips)),
        "model_kind": str(gate.model_kind),
        "fit_profit_factor": float(gate.fit_profit_factor),
        "fit_expectancy_usd": float(gate.fit_expectancy_usd),
    }


def generate_rc_notes(manifest: dict[str, Any]) -> str:
    runtime_constraints = dict(manifest.get("runtime_constraints") or {})
    rule_params = dict(manifest.get("rule_params") or {})
    alpha_gate = dict(manifest.get("alpha_gate") or {})
    return (
        f"# Release Notes: {manifest['strategy_symbol']} {manifest['ticks_per_bar']} Tick RC1\n\n"
        "## Summary\n"
        f"* **Status**: `{manifest['release_stage']}`\n"
        "* **Architecture**: Pair-specific horizon architecture (rule-first).\n"
        f"* **Rule Family**: `{manifest.get('rule_family')}`\n"
        f"* **Ticks Per Bar**: `{manifest['ticks_per_bar']}`\n\n"
        "## Safety Contract\n"
        "* **NOT PRODUCTION READY**\n"
        "* `live_trading_approved=false` is mandatory for this pack.\n"
        "* Approved only for paper-live shadow evidence collection.\n\n"
        "## Runtime Contract\n"
        f"* **Spread Guard**: `{runtime_constraints.get('spread_sanity_max_pips')}` pips\n"
        f"* **Allowed Sessions**: `{runtime_constraints.get('allowed_sessions')}`\n"
        f"* **Rollover Block (UTC)**: `{runtime_constraints.get('rollover_block_utc_hours')}`\n"
        f"* **Daily Loss Stop**: `${runtime_constraints.get('daily_loss_stop_usd')}`\n"
        f"* **Rule Params**: `{json.dumps(rule_params, sort_keys=True)}`\n\n"
        "## AlphaGate Contract\n"
        f"* **Enabled**: `{bool(alpha_gate.get('enabled', False))}`\n"
        f"* **Model Path**: `{alpha_gate.get('model_path')}`\n"
        f"* **Model SHA256**: `{alpha_gate.get('model_sha256')}`\n"
        f"* **Threshold/Margin**: `{alpha_gate.get('probability_threshold')}` / `{alpha_gate.get('probability_margin')}`\n\n"
        "## Traceability\n"
        f"* **Evaluator Hash**: `{manifest.get('evaluator_hash')}`\n"
        f"* **Logic Hash**: `{manifest.get('logic_hash')}`\n"
        f"* **Manifest Hash**: `{manifest.get('manifest_hash')}`\n"
    )


def build_manifest(config: dict[str, Any], *, git_commit: str, evaluator_hash: str, logic_hash: str) -> Path:
    dataset_path = resolve_dataset_path(ticks_per_bar=int(config["ticks_per_bar"]))
    out_dir = RC_ROOT / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    featured = _load_symbol_feature_frame(dataset_path, symbol=str(config["symbol"]).upper())
    holdout_start_utc = _compute_holdout_start_utc(featured)
    alpha_gate_payload = _fit_and_store_alpha_gate(
        config=config,
        featured=featured,
        holdout_start_utc=holdout_start_utc,
        out_dir=out_dir,
    )
    runtime_constraints = RuntimeConstraints(
        session_filter_active=True,
        spread_sanity_max_pips=float(config["spread_limit_pips"]),
        max_concurrent_positions=1,
        daily_loss_stop_usd=100.0,
        rollover_block_utc_hours=list(config.get("rollover_block_utc_hours", [21, 22, 23, 0])),
        allowed_sessions=list(config.get("allowed_sessions", [])),
    )
    manifest = create_rule_manifest(
        strategy_symbol=config["symbol"],
        rule_family=config["rule_family"],
        rule_params=dict(config["rule_params"]),
        dataset_path=dataset_path,
        ticks_per_bar=int(config["ticks_per_bar"]),
        holdout_start_utc=holdout_start_utc,
        cost_model=CostModel(commission_per_lot=7.0, slippage_pips=0.25),
        threshold_policy=ThresholdPolicy(min_edge_pips=0.0, reject_ambiguous=True),
        runtime_constraints=runtime_constraints,
        alpha_gate=AlphaGateSpec(
            enabled=bool(alpha_gate_payload.get("enabled", False)),
            model_path=alpha_gate_payload.get("model_path"),
            model_sha256=alpha_gate_payload.get("model_sha256"),
            probability_threshold=alpha_gate_payload.get("probability_threshold"),
            probability_margin=alpha_gate_payload.get("probability_margin"),
            min_edge_pips=alpha_gate_payload.get("min_edge_pips"),
        ),
        git_commit=git_commit,
        release_stage="paper_live_candidate",
        evaluator_hash=evaluator_hash,
        logic_hash=logic_hash,
        replay_parity_reference="tools/verify_v1_rc.py",
        startup_truth_snapshot=_truth_snapshot(
            config,
            dataset_path=dataset_path,
            evaluator_hash=evaluator_hash,
            logic_hash=logic_hash,
            holdout_start_utc=holdout_start_utc,
            alpha_gate_payload=alpha_gate_payload,
        ),
    )
    manifest_path = out_dir / "manifest.json"
    save_selector_manifest(manifest, manifest_path)
    manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
    (out_dir / "release_notes_rc1.md").write_text(generate_rc_notes(manifest_dict), encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Bot v1 RC1 paper-live candidate packs.")
    parser.add_argument("--skip-verify", action="store_true", help="Generate manifests without running certification.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    evaluator_hash = _file_sha256(ROOT / "evaluate_oos.py")
    logic_hash = _file_sha256(ROOT / "strategies" / "rule_logic.py")
    git_commit = _current_git_commit()
    RC_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_paths: list[Path] = []
    for config in RC_CONFIGS:
        log.info("Generating RC1 pack %s", config["name"])
        manifest_path = build_manifest(
            config,
            git_commit=git_commit,
            evaluator_hash=evaluator_hash,
            logic_hash=logic_hash,
        )
        loaded = load_selector_manifest(manifest_path, verify_manifest_hash=True)
        log.info(
            "  -> %s [%s %s ticks hash=%s]",
            manifest_path.parent,
            loaded.strategy_symbol,
            loaded.ticks_per_bar,
            loaded.manifest_hash[:12],
        )
        manifest_paths.append(manifest_path)

    if not args.skip_verify:
        command = [sys.executable, str(ROOT / "tools" / "verify_v1_rc.py"), *[str(path) for path in manifest_paths]]
        log.info("Running RC1 certification for %d pack(s)", len(manifest_paths))
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
