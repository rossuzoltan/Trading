import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

from feature_engine import FEATURE_COLS, _compute_raw
from project_paths import resolve_dataset_path
from run_logging import configure_run_logging
from selector_manifest import (
    create_selector_manifest, 
    save_selector_manifest,
    LabelDefinition, CostModel, ThresholdPolicy, RuntimeConstraints
)
from symbol_utils import price_to_pips, pip_value_for_volume
from train_agent import _split_holdout
from edge_research import _prepare_targets

log = logging.getLogger("train_selector")

def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    configure_run_logging("train_selector", run_id=f"train_selector_{uuid.uuid4().hex[:8]}")
    
    symbol = os.environ.get("TRAIN_SYMBOL", "EURUSD").upper()
    ticks_per_bar = int(os.environ.get("TRAIN_BAR_TICKS", "10000"))
    
    # Explicit Label Path A Setup
    horizon_bars = 10
    commission_per_lot = 7.0
    slippage_pips = 0.25
    min_edge_pips = 1.0

    cost_model = CostModel(
        commission_per_lot=commission_per_lot,
        slippage_pips=slippage_pips
    )
    
    threshold_policy = ThresholdPolicy(
        min_edge_pips=min_edge_pips,
        reject_ambiguous=True
    )
    
    runtime_constraints = RuntimeConstraints(
        session_filter_active=True,
        spread_sanity_max_pips=2.0,
        max_concurrent_positions=1,
        daily_loss_stop_usd=50.0
    )
    
    label_def = LabelDefinition(
        path="A",
        target_column="signed_target",
        horizon_bars=horizon_bars,
        is_classification=False
    )
    
    log.info(f"Starting Supervised Selector Training for {symbol} | {ticks_per_bar} ticks/bar")
    
    # 1. Load Data
    dataset_path = resolve_dataset_path()
    log.info(f"Loading dataset: {dataset_path}")
    raw = pd.read_csv(dataset_path, low_memory=False, parse_dates=["Gmt time"])
    raw = raw.loc[raw["Symbol"].astype(str).str.upper() == symbol].copy()
    raw["Gmt time"] = pd.to_datetime(raw["Gmt time"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["Gmt time"]).set_index("Gmt time").sort_index()

    if raw.empty:
        log.error(f"No data for {symbol}.")
        return

    # 2. Build Features
    log.info("Computing features...")
    featured = _compute_raw(raw).dropna(subset=list(FEATURE_COLS))
    log.info(f"Total featured bars: {len(featured)}")

    # 3. Holdout Segregation
    trainable_feature_frame, holdout_feature_frame = _split_holdout(featured, 0.15)
    log.info(f"Trainable bars: {len(trainable_feature_frame)}, Holdout bars: {len(holdout_feature_frame)}")
    
    holdout_start_utc_str = ""
    if not holdout_feature_frame.empty:
        holdout_start_utc_str = holdout_feature_frame.index[0].isoformat()

    # 4. Target Synthesis
    log.info("Synthesizing Continuous Expectancy Targets (signed_target)...")
    prepared_trainable = _prepare_targets(
        trainable_feature_frame,
        symbol=symbol,
        feature_cols=list(FEATURE_COLS),
        horizon_bars=horizon_bars,
        commission_per_lot=commission_per_lot,
        slippage_pips=slippage_pips,
        min_edge_pips=min_edge_pips,
    )
    
    if prepared_trainable.empty:
        log.error("Insufficient samples after target synthesis.")
        return

    x_train = prepared_trainable.loc[:, FEATURE_COLS].to_numpy(dtype=float)
    y_train = prepared_trainable["signed_target"].to_numpy(dtype=float)
    
    # 5. Train Model
    log.info("Fitting DecisionTreeRegressor pipeline...")
    # DecisionTree Challenger replacing HGB
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    
    pipeline.fit(x_train, y_train)
    score = pipeline.score(x_train, y_train)
    log.info(f"Training R2 Score: {score:.4f}")

    # 6. Save Model & Manifest
    model_dir = Path("models") / "selector" / f"{symbol.lower()}_{ticks_per_bar}_tree"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "model.joblib"
    manifest_path = model_dir / "selector_manifest.json"
    
    log.info(f"Saving model to {model_path}...")
    joblib.dump(pipeline, model_path)
    
    model_version = "1.0.0"
    
    manifest = create_selector_manifest(
        strategy_symbol=symbol,
        model_path=model_path,
        model_version=model_version,
        feature_schema=list(FEATURE_COLS),
        dataset_path=dataset_path,
        ticks_per_bar=ticks_per_bar,
        bar_construction_ticks_per_bar=ticks_per_bar,
        holdout_start_utc=holdout_start_utc_str,
        label_definition=label_def,
        cost_model=cost_model,
        threshold_policy=threshold_policy,
        runtime_constraints=runtime_constraints,
    )
    
    log.info(f"Saving manifest to {manifest_path}...")
    save_selector_manifest(manifest, manifest_path)
    
    log.info(f"Selector training complete for {symbol}. Ready for replay evaluation.")

if __name__ == "__main__":
    main()
