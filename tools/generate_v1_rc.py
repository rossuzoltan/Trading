from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import resolve_dataset_path
from selector_manifest import (
    CostModel,
    RuntimeConstraints,
    ThresholdPolicy,
    _file_sha256,
    create_rule_manifest,
    load_selector_manifest,
    save_selector_manifest,
)

log = logging.getLogger("generate_v1_rc")
RC_ROOT = ROOT / "models" / "rc1"

RC_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "symbol": "EURUSD",
        "ticks_per_bar": 5000,
        "name": "eurusd_5k_v1_mr_rc1",
        "rule_family": "mean_reversion",
        "rule_params": {"threshold": 1.0, "sl_value": 1.5, "tp_value": 3.0},
        "spread_limit_pips": 1.5,
    },
    {
        "symbol": "GBPUSD",
        "ticks_per_bar": 10000,
        "name": "gbpusd_10k_v1_mr_rc1",
        "rule_family": "mean_reversion",
        "rule_params": {"threshold": 1.0, "sl_value": 1.5, "tp_value": 3.0},
        "spread_limit_pips": 2.5,
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


def _truth_snapshot(config: dict[str, Any], *, dataset_path: Path, evaluator_hash: str, logic_hash: str) -> dict[str, Any]:
    return {
        "release_name": config["name"],
        "strategy_symbol": config["symbol"],
        "rule_family": config["rule_family"],
        "rule_params": dict(config["rule_params"]),
        "ticks_per_bar": int(config["ticks_per_bar"]),
        "dataset_path": str(dataset_path),
        "release_stage": "paper_live_candidate",
        "live_trading_approved": False,
        "evaluator_hash": evaluator_hash,
        "logic_hash": logic_hash,
    }


def generate_rc_notes(manifest: dict[str, Any]) -> str:
    runtime_constraints = dict(manifest.get("runtime_constraints") or {})
    rule_params = dict(manifest.get("rule_params") or {})
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
        f"* **Daily Loss Stop**: `${runtime_constraints.get('daily_loss_stop_usd')}`\n"
        f"* **Rule Params**: `{json.dumps(rule_params, sort_keys=True)}`\n\n"
        "## Traceability\n"
        f"* **Evaluator Hash**: `{manifest.get('evaluator_hash')}`\n"
        f"* **Logic Hash**: `{manifest.get('logic_hash')}`\n"
        f"* **Manifest Hash**: `{manifest.get('manifest_hash')}`\n"
    )


def build_manifest(config: dict[str, Any], *, git_commit: str, evaluator_hash: str, logic_hash: str) -> Path:
    dataset_path = resolve_dataset_path(ticks_per_bar=int(config["ticks_per_bar"]))
    runtime_constraints = RuntimeConstraints(
        session_filter_active=True,
        spread_sanity_max_pips=float(config["spread_limit_pips"]),
        max_concurrent_positions=1,
        daily_loss_stop_usd=100.0,
    )
    manifest = create_rule_manifest(
        strategy_symbol=config["symbol"],
        rule_family=config["rule_family"],
        rule_params=dict(config["rule_params"]),
        dataset_path=dataset_path,
        ticks_per_bar=int(config["ticks_per_bar"]),
        holdout_start_utc=None,
        cost_model=CostModel(commission_per_lot=7.0, slippage_pips=0.25),
        threshold_policy=ThresholdPolicy(min_edge_pips=0.0, reject_ambiguous=True),
        runtime_constraints=runtime_constraints,
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
        ),
    )
    out_dir = RC_ROOT / config["name"]
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
