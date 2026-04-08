import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_oos import (
    _evaluate_policy,
    _evaluate_runtime_baselines,
    _target_direction_to_action_index,
    ReplayContext,
    _load_symbol_raw_frame
)
from feature_engine import FEATURE_COLS, _compute_raw
from project_paths import resolve_dataset_path
from run_logging import configure_run_logging
from runtime_common import ActionSpec, ActionType, serialize_action_map
from selector_manifest import load_selector_manifest, load_validated_selector_model
from validation_metrics import save_json_report

log = logging.getLogger("replay_selector")

# Default action map simulating the strict "Sparse Trading"
# 0 = HOLD, 1 = LONG, 2 = SHORT, 3 = CLOSE
SELECTOR_ACTION_MAP = [
    ActionSpec(ActionType.HOLD),
    ActionSpec(ActionType.OPEN, direction=1, sl_value=1.5, tp_value=3.0),
    ActionSpec(ActionType.OPEN, direction=-1, sl_value=1.5, tp_value=3.0),
    ActionSpec(ActionType.CLOSE),
]

def load_selector_replay_context(symbol: str) -> ReplayContext:
    model_dir = Path("models") / "selector" / f"{symbol.lower()}_10k"
    manifest_path = model_dir / "selector_manifest.json"
    
    if not manifest_path.exists():
        raise RuntimeError(f"No selector manifest found at {manifest_path}")
        
    manifest = load_selector_manifest(manifest_path)
    dataset_path = resolve_dataset_path()
    
    raw = _load_symbol_raw_frame(
        symbol=symbol, 
        dataset_path=dataset_path, 
        expected_ticks_per_bar=manifest.bar_construction_ticks_per_bar
    )
    featured = _compute_raw(raw).dropna(subset=list(FEATURE_COLS))
    
    if manifest.holdout_start_utc:
        holdout_start = pd.Timestamp(manifest.holdout_start_utc)
        replay_frame = raw.loc[holdout_start:].copy()
        holdout_feature_frame = featured.loc[featured.index >= holdout_start].copy()
        trainable_feature_frame = featured.loc[featured.index < holdout_start].copy()
        warmup_frame = raw.loc[raw.index < holdout_start].iloc[-300:].copy()
    else:
        raise RuntimeError("Selector manifest must have holdout_start_utc.")

    model = load_validated_selector_model(manifest, expected_symbol=symbol)
    
    return ReplayContext(
        symbol=symbol.upper(),
        source="selector_manifest",
        dataset_path=dataset_path,
        action_map=tuple(SELECTOR_ACTION_MAP),
        model=model,
        obs_normalizer=None,
        scaler=None,
        execution_cost_profile=manifest.cost_model,
        reward_profile={},
        warmup_frame=warmup_frame,
        replay_frame=replay_frame,
        replay_feature_frame=holdout_feature_frame,
        full_feature_frame=featured,
        trainable_feature_frame=trainable_feature_frame,
        holdout_feature_frame=holdout_feature_frame,
        holdout_start_utc=pd.Timestamp(holdout_feature_frame.index[0]).isoformat(),
        diagnostics_path=None,
        manifest_path=manifest_path,
        artifact_metadata={"manifest_path": str(manifest_path), "model_path": str(manifest.model_path)},
        runtime_options={"window_size": 1},
    )

def create_selector_provider(model, manifest):
    threshold_pips = manifest.threshold_policy.get("min_edge_pips", 0.0)
    
    # Audit log foundation
    audit_stats = Counter()
    
    def provider(*, feature_row: dict[str, float], position_direction: int, action_map, bar, **_: object) -> int:
        feature_vector = np.array([feature_row[c] for c in FEATURE_COLS]).reshape(1, -1)
        pred_expectancy = model.predict(feature_vector)[0]
        
        target_direction = 0
        if pred_expectancy > threshold_pips:
            target_direction = 1
            audit_stats["conviction_long"] += 1
        elif pred_expectancy < -threshold_pips:
            target_direction = -1
            audit_stats["conviction_short"] += 1
        else:
            audit_stats["flat_missed_threshold"] += 1
            
        return _target_direction_to_action_index(
            action_map=action_map,
            position_direction=int(position_direction or 0),
            target_direction=target_direction
        )
        
    provider.audit_stats = audit_stats
    return provider

def _sanitize_payload(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_payload(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj

def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    configure_run_logging("replay_selector")
    
    symbol = os.environ.get("REPLAY_SYMBOL", "EURUSD").upper()
    log.info(f"Loading Supervised Selector for {symbol}...")
    
    context = load_selector_replay_context(symbol)
    provider = create_selector_provider(context.model, load_selector_manifest(context.manifest_path))
    
    log.info("Running exact-runtime evaluation loop...")
    results = _evaluate_policy(
        replay_context=context,
        action_index_provider=provider
    )
    
    audit_log = provider.audit_stats
    log.info(f"No-Trade Audit Log: {dict(audit_log)}")
    
    log.info("Running Anchor Baselines...")
    baselines = _evaluate_runtime_baselines(replay_context=context)
    
    # Inject our selector into the baselines payload for exact scoreboarding
    baselines["supervised_selector"] = results
    
    flat_net = baselines["runtime_flat"]["metrics"]["net_pnl_usd"]
    selector_net = results["metrics"]["net_pnl_usd"]
    
    log.info(f"Selector Net PnL: ${selector_net:.2f} | Flat Net PnL: ${flat_net:.2f}")
    
    sanitized = _sanitize_payload(baselines)
    out_path = Path("models/selector") / f"{symbol.lower()}_10k" / "replay_report.json"
    save_json_report(sanitized, out_path)
        
    log.info(f"Saved replay report to {out_path}.")

if __name__ == "__main__":
    main()
