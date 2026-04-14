from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from symbol_utils import pip_value_for_volume, price_to_pips


EDGE_RESEARCH_PERF: dict[str, float] = {
    "prepare_targets_calls": 0,
    "prepare_targets_total_ns": 0,
}


@dataclass
class BaselineAlphaGate:
    symbol: str
    feature_cols: tuple[str, ...]
    model_kind: str
    probability_threshold: float = 0.55
    probability_margin: float = 0.05
    min_edge_pips: float = 0.0
    long_model: Any | None = None
    short_model: Any | None = None
    signed_model: Any | None = None
    fit_trade_count: float = 0.0
    fit_long_trade_count: float = 0.0
    fit_short_trade_count: float = 0.0
    fit_expectancy_usd: float = 0.0
    fit_profit_factor: float = 0.0
    fit_quality_passed: bool = False

    def score_row(self, row: pd.Series | dict[str, Any]) -> dict[str, float]:
        feature_values = [float((row.get(col, 0.0) if hasattr(row, "get") else 0.0)) for col in self.feature_cols]
        features = np.asarray([feature_values], dtype=np.float64)
        feature_frame = pd.DataFrame([feature_values], columns=list(self.feature_cols))
        scores = {
            "long_score": 0.0,
            "short_score": 0.0,
            "signed_score": 0.0,
        }
        if self.long_model is not None and self.short_model is not None:
            long_input = feature_frame if hasattr(self.long_model, "feature_names_in_") else features
            short_input = feature_frame if hasattr(self.short_model, "feature_names_in_") else features
            scores["long_score"] = float(self.long_model.predict_proba(long_input)[0, 1])
            scores["short_score"] = float(self.short_model.predict_proba(short_input)[0, 1])
        if self.signed_model is not None:
            signed_input = feature_frame if hasattr(self.signed_model, "feature_names_in_") else features
            scores["signed_score"] = float(self.signed_model.predict(signed_input)[0])
        return scores

    def allowed_directions(
        self,
        row: pd.Series | dict[str, Any],
        *,
        threshold_override: float | None = None,
        margin_override: float | None = None,
    ) -> tuple[bool, bool, dict[str, float]]:
        scores = self.score_row(row)
        if self.model_kind in {"logistic_pair", "xgboost_pair", "lightgbm_pair"}:
            long_score = float(scores["long_score"])
            short_score = float(scores["short_score"])
            effective_threshold = float(
                self.probability_threshold if threshold_override is None else threshold_override
            )
            effective_margin = float(
                self.probability_margin if margin_override is None else margin_override
            )
            allow_long = long_score >= effective_threshold and (
                (long_score - short_score) >= effective_margin
            )
            allow_short = short_score >= effective_threshold and (
                (short_score - long_score) >= effective_margin
            )
            return bool(allow_long), bool(allow_short), scores
        if self.model_kind == "ridge_signed_target":
            signed_score = float(scores["signed_score"])
            allow_long = signed_score >= float(self.min_edge_pips)
            allow_short = signed_score <= -float(self.min_edge_pips)
            return bool(allow_long), bool(allow_short), scores
        return False, False, scores


def save_baseline_alpha_gate(gate: BaselineAlphaGate, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gate, out_path)
    return out_path


def load_baseline_alpha_gate(path: str | Path) -> BaselineAlphaGate:
    gate = joblib.load(Path(path))
    if not isinstance(gate, BaselineAlphaGate):
        raise RuntimeError(f"Loaded object is not BaselineAlphaGate: {type(gate).__name__}")
    return gate


def _ensure_cost_columns(frame: pd.DataFrame) -> pd.DataFrame:
    dataset = frame.copy()
    if "Open" not in dataset.columns:
        dataset["Open"] = dataset["Close"]
    if "avg_spread" not in dataset.columns:
        dataset["avg_spread"] = 0.0
    return dataset


def _commission_pips(price: float, *, symbol: str, commission_per_lot: float) -> float:
    pip_value = pip_value_for_volume(symbol, price=price, volume_lots=1.0, account_currency="USD")
    if pip_value <= 0:
        return 0.0
    return float(commission_per_lot / pip_value)


def _prepare_targets(
    frame: pd.DataFrame,
    *,
    symbol: str,
    feature_cols: list[str],
    horizon_bars: int,
    commission_per_lot: float,
    slippage_pips: float,
    min_edge_pips: float,
) -> pd.DataFrame:
    start_ns = time.perf_counter_ns()
    dataset = _ensure_cost_columns(frame)
    required = [*feature_cols, "Close", "Open", "avg_spread"]
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise RuntimeError(f"Baseline research is missing required columns: {missing}")

    prepared = dataset.loc[:, required].copy()
    prepared["entry_open"] = prepared["Open"].shift(-1)
    prepared["exit_close"] = prepared["Close"].shift(-horizon_bars)
    prepared["entry_spread"] = prepared["avg_spread"].shift(-1).fillna(prepared["avg_spread"])
    prepared["exit_spread"] = prepared["avg_spread"].shift(-horizon_bars).fillna(prepared["avg_spread"])
    prepared = prepared.replace([np.inf, -np.inf], np.nan).dropna()
    if prepared.empty:
        EDGE_RESEARCH_PERF["prepare_targets_calls"] += 1
        EDGE_RESEARCH_PERF["prepare_targets_total_ns"] += max(time.perf_counter_ns() - start_ns, 0)
        return prepared

    raw_move_pips = prepared.apply(
        lambda row: price_to_pips(symbol, float(row["exit_close"]) - float(row["entry_open"])),
        axis=1,
    ).to_numpy(dtype=np.float64)
    cost_pips = (
        prepared.apply(lambda row: abs(price_to_pips(symbol, float(row["entry_spread"]) / 2.0)), axis=1).to_numpy(dtype=np.float64)
        + prepared.apply(lambda row: abs(price_to_pips(symbol, float(row["exit_spread"]) / 2.0)), axis=1).to_numpy(dtype=np.float64)
        + float(slippage_pips) * 2.0
        + prepared["entry_open"].apply(
            lambda price: _commission_pips(float(price), symbol=symbol, commission_per_lot=commission_per_lot)
        ).to_numpy(dtype=np.float64)
        + prepared["exit_close"].apply(
            lambda price: _commission_pips(float(price), symbol=symbol, commission_per_lot=commission_per_lot)
        ).to_numpy(dtype=np.float64)
    )
    prepared["long_net_pips"] = raw_move_pips - cost_pips
    prepared["short_net_pips"] = -raw_move_pips - cost_pips
    # NON-PARITY APPROXIMATION:
    # Baseline costs are estimated via simplified pip-math and are not execution-path
    # identical to ReplayBroker results. Absolute comparability is limited.
    prepared["long_target"] = prepared["long_net_pips"] >= min_edge_pips
    prepared["short_target"] = prepared["short_net_pips"] >= min_edge_pips
    prepared["signed_target"] = np.where(
        prepared["long_net_pips"] >= prepared["short_net_pips"],
        prepared["long_net_pips"],
        -prepared["short_net_pips"],
    )
    EDGE_RESEARCH_PERF["prepare_targets_calls"] += 1
    EDGE_RESEARCH_PERF["prepare_targets_total_ns"] += max(time.perf_counter_ns() - start_ns, 0)
    return prepared


def _simulate_signals(frame: pd.DataFrame, signals: np.ndarray, *, symbol: str, horizon_bars: int) -> dict[str, float]:
    pnl_usd: list[float] = []
    directions: list[int] = []
    index = 0
    rows = frame.reset_index(drop=True)
    while index < len(rows):
        signal = int(signals[index])
        if signal == 0:
            index += 1
            continue
        row = rows.iloc[index]
        pnl_pips = float(row["long_net_pips"]) if signal > 0 else float(row["short_net_pips"])
        pip_value = pip_value_for_volume(symbol, price=float(row["entry_open"]), volume_lots=1.0, account_currency="USD")
        pnl_usd.append(float(pnl_pips * pip_value))
        directions.append(int(signal))
        index += max(horizon_bars, 1)
    if not pnl_usd:
        return {
            "trade_count": 0.0,
            "expectancy_usd": 0.0,
            "avg_trade_usd": 0.0,
            "profit_factor": 0.0,
            "net_pnl_usd": 0.0,
            "gross_profit_usd": 0.0,
            "gross_loss_usd": 0.0,
            "win_rate": 0.0,
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0,
            "win_loss_asymmetry": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown_usd": 0.0,
            "long_trade_count": 0.0,
            "short_trade_count": 0.0,
            "avg_holding_bars": float(max(horizon_bars, 1)),
            "trades_per_bar": 0.0,
        }
    pnl_array = np.asarray(pnl_usd, dtype=np.float64)
    wins = [value for value in pnl_usd if value > 0]
    losses = [value for value in pnl_usd if value < 0]
    gross_profit = float(sum(wins))
    gross_loss = float(abs(sum(losses)))
    equity_curve = np.cumsum(pnl_array)
    running_peak = np.maximum.accumulate(np.maximum(equity_curve, 0.0))
    drawdowns = running_peak - equity_curve
    avg_loss_usd = float(np.mean(losses)) if losses else 0.0
    avg_win_usd = float(np.mean(wins)) if wins else 0.0
    pnl_std = float(np.std(pnl_array, ddof=0))
    return {
        "trade_count": float(len(pnl_usd)),
        "expectancy_usd": float(np.mean(pnl_array)),
        "avg_trade_usd": float(np.mean(pnl_array)),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "net_pnl_usd": float(np.sum(pnl_array)),
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "win_rate": float(len(wins) / len(pnl_usd)),
        "avg_win_usd": avg_win_usd,
        "avg_loss_usd": avg_loss_usd,
        "win_loss_asymmetry": float(avg_win_usd / abs(avg_loss_usd)) if avg_loss_usd < 0.0 else (float("inf") if avg_win_usd > 0.0 else 0.0),
        "sharpe_like": float((np.mean(pnl_array) / pnl_std) * np.sqrt(len(pnl_array))) if pnl_std > 0.0 else 0.0,
        "max_drawdown_usd": float(np.max(drawdowns)) if drawdowns.size else 0.0,
        "long_trade_count": float(sum(1 for direction in directions if direction > 0)),
        "short_trade_count": float(sum(1 for direction in directions if direction < 0)),
        "avg_holding_bars": float(max(horizon_bars, 1)),
        "trades_per_bar": float(len(pnl_usd) / max(len(rows), 1)),
    }


def _choose_probability_threshold(
    *,
    frame: pd.DataFrame,
    symbol: str,
    horizon_bars: int,
    long_scores: np.ndarray,
    short_scores: np.ndarray,
    probability_threshold: float,
    probability_margin: float,
) -> dict[str, Any]:
    thresholds = [probability_threshold, probability_threshold + 0.05, probability_threshold + 0.10]
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        signals = np.zeros(len(frame), dtype=np.int8)
        signals[(long_scores >= threshold) & ((long_scores - short_scores) >= probability_margin)] = 1
        signals[(short_scores >= threshold) & ((short_scores - long_scores) >= probability_margin)] = -1
        metrics = _simulate_signals(frame, signals, symbol=symbol, horizon_bars=horizon_bars)
        report = {"threshold": float(threshold), "metrics": metrics}
        if best is None or (
            float(metrics["expectancy_usd"]),
            float(metrics["profit_factor"]),
            float(metrics["trade_count"]),
        ) > (
            float(best["metrics"]["expectancy_usd"]),
            float(best["metrics"]["profit_factor"]),
            float(best["metrics"]["trade_count"]),
        ):
            best = report
    if best is None:
        return {"threshold": float(probability_threshold), "metrics": _simulate_signals(frame, np.zeros(len(frame), dtype=np.int8), symbol=symbol, horizon_bars=horizon_bars)}
    return best


def _alpha_gate_quality_report(
    metrics: dict[str, Any],
    *,
    min_trade_count: int,
    min_directional_trade_count: int,
    max_single_direction_fraction: float,
) -> dict[str, Any]:
    trade_count = int(float(metrics.get("trade_count", 0.0) or 0.0))
    long_trade_count = int(float(metrics.get("long_trade_count", 0.0) or 0.0))
    short_trade_count = int(float(metrics.get("short_trade_count", 0.0) or 0.0))
    expectancy_usd = float(metrics.get("expectancy_usd", 0.0) or 0.0)
    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    dominant_direction = max(long_trade_count, short_trade_count)
    dominant_fraction = (
        float(dominant_direction) / float(trade_count)
        if trade_count > 0
        else 1.0
    )
    directionally_balanced = (
        long_trade_count >= int(min_directional_trade_count)
        and short_trade_count >= int(min_directional_trade_count)
    )
    economically_viable = expectancy_usd > 0.0 and profit_factor > 1.0
    quality_passed = (
        trade_count >= int(min_trade_count)
        and dominant_fraction <= float(max_single_direction_fraction)
        and directionally_balanced
        and economically_viable
    )
    return {
        "trade_count": trade_count,
        "long_trade_count": long_trade_count,
        "short_trade_count": short_trade_count,
        "expectancy_usd": expectancy_usd,
        "profit_factor": profit_factor,
        "dominant_direction_fraction": float(dominant_fraction),
        "directionally_balanced": bool(directionally_balanced),
        "economically_viable": bool(economically_viable),
        "quality_passed": bool(quality_passed),
    }


def _build_pair_classifier(model_kind: str) -> Any | None:
    if model_kind == "logistic_pair":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if model_kind == "xgboost_pair":
        try:
            from xgboost import XGBClassifier
        except Exception:
            return None
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
    if model_kind == "lightgbm_pair":
        try:
            from lightgbm import LGBMClassifier
        except Exception:
            return None
        return LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )
    return None


def _fit_probability_pair(
    x_train: np.ndarray,
    frame_train: pd.DataFrame,
    *,
    model_kind: str,
) -> tuple[Any | None, Any | None]:
    if frame_train["long_target"].nunique() < 2 or frame_train["short_target"].nunique() < 2:
        return None, None
    long_classifier = _build_pair_classifier(model_kind)
    short_classifier = _build_pair_classifier(model_kind)
    if long_classifier is None or short_classifier is None:
        return None, None
    long_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", long_classifier),
        ]
    )
    short_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", short_classifier),
        ]
    )
    long_model.fit(x_train, frame_train["long_target"].to_numpy(dtype=np.int8))
    short_model.fit(x_train, frame_train["short_target"].to_numpy(dtype=np.int8))
    return long_model, short_model


def fit_baseline_alpha_gate(
    *,
    symbol: str,
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    horizon_bars: int,
    commission_per_lot: float,
    slippage_pips: float,
    min_edge_pips: float,
    probability_threshold: float,
    probability_margin: float,
    model_preference: str = "auto",
    min_trade_count: int = 20,
    min_directional_trade_count: int = 4,
    max_single_direction_fraction: float = 0.90,
) -> BaselineAlphaGate | None:
    prepared_train = _prepare_targets(
        train_frame,
        symbol=symbol,
        feature_cols=feature_cols,
        horizon_bars=horizon_bars,
        commission_per_lot=commission_per_lot,
        slippage_pips=slippage_pips,
        min_edge_pips=min_edge_pips,
    )
    if prepared_train.empty:
        return None

    x_train = prepared_train.loc[:, feature_cols].to_numpy(dtype=np.float64)
    preference = str(model_preference or "auto").strip().lower()
    pair_preferences: tuple[str, ...]
    if preference == "auto":
        pair_preferences = ("logistic_pair", "xgboost_pair", "lightgbm_pair")
    else:
        pair_preferences = (preference,)

    for pair_model_kind in pair_preferences:
        if pair_model_kind not in {"logistic_pair", "xgboost_pair", "lightgbm_pair"}:
            continue
        pair_long_model, pair_short_model = _fit_probability_pair(
            x_train,
            prepared_train,
            model_kind=pair_model_kind,
        )
        if pair_long_model is not None and pair_short_model is not None:
            long_train = np.asarray(pair_long_model.predict_proba(x_train)[:, 1], dtype=np.float64)
            short_train = np.asarray(pair_short_model.predict_proba(x_train)[:, 1], dtype=np.float64)
            selected = _choose_probability_threshold(
                frame=prepared_train,
                symbol=symbol,
                horizon_bars=horizon_bars,
                long_scores=long_train,
                short_scores=short_train,
                probability_threshold=probability_threshold,
                probability_margin=probability_margin,
            )
            quality = _alpha_gate_quality_report(
                selected["metrics"],
                min_trade_count=min_trade_count,
                min_directional_trade_count=min_directional_trade_count,
                max_single_direction_fraction=max_single_direction_fraction,
            )
            if quality["quality_passed"]:
                selected_metrics = dict(selected["metrics"] or {})
                return BaselineAlphaGate(
                    symbol=str(symbol).upper(),
                    feature_cols=tuple(feature_cols),
                    model_kind=pair_model_kind,
                    probability_threshold=float(selected["threshold"]),
                    probability_margin=float(probability_margin),
                    min_edge_pips=float(min_edge_pips),
                    long_model=pair_long_model,
                    short_model=pair_short_model,
                    fit_trade_count=float(quality["trade_count"]),
                    fit_long_trade_count=float(quality["long_trade_count"]),
                    fit_short_trade_count=float(quality["short_trade_count"]),
                    fit_expectancy_usd=float(selected_metrics.get("expectancy_usd", 0.0)),
                    fit_profit_factor=float(selected_metrics.get("profit_factor", 0.0)),
                    fit_quality_passed=True,
                )
            if preference == pair_model_kind:
                return BaselineAlphaGate(
                    symbol=str(symbol).upper(),
                    feature_cols=tuple(feature_cols),
                    model_kind=pair_model_kind,
                    probability_threshold=float(selected["threshold"]),
                    probability_margin=float(probability_margin),
                    min_edge_pips=float(min_edge_pips),
                    long_model=pair_long_model,
                    short_model=pair_short_model,
                    fit_trade_count=float(quality["trade_count"]),
                    fit_long_trade_count=float(quality["long_trade_count"]),
                    fit_short_trade_count=float(quality["short_trade_count"]),
                    fit_expectancy_usd=float(dict(selected["metrics"] or {}).get("expectancy_usd", 0.0)),
                    fit_profit_factor=float(dict(selected["metrics"] or {}).get("profit_factor", 0.0)),
                    fit_quality_passed=False,
                )
        if preference == pair_model_kind:
            return None

    if preference in {"auto", "ridge_signed_target", "ridge"}:
        ridge = Pipeline(steps=[("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        ridge.fit(x_train, prepared_train["signed_target"].to_numpy(dtype=np.float64))
        ridge_train = np.asarray(ridge.predict(x_train), dtype=np.float64)
        ridge_signals = np.zeros(len(prepared_train), dtype=np.int8)
        ridge_signals[ridge_train >= min_edge_pips] = 1
        ridge_signals[ridge_train <= -min_edge_pips] = -1
        ridge_metrics = _simulate_signals(prepared_train, ridge_signals, symbol=symbol, horizon_bars=horizon_bars)
        ridge_quality = _alpha_gate_quality_report(
            ridge_metrics,
            min_trade_count=min_trade_count,
            min_directional_trade_count=min_directional_trade_count,
            max_single_direction_fraction=max_single_direction_fraction,
        )
        if not ridge_quality["quality_passed"]:
            return None
        return BaselineAlphaGate(
            symbol=str(symbol).upper(),
            feature_cols=tuple(feature_cols),
            model_kind="ridge_signed_target",
            probability_threshold=float(probability_threshold),
            probability_margin=float(probability_margin),
            min_edge_pips=float(min_edge_pips),
            signed_model=ridge,
            fit_trade_count=float(ridge_quality["trade_count"]),
            fit_long_trade_count=float(ridge_quality["long_trade_count"]),
            fit_short_trade_count=float(ridge_quality["short_trade_count"]),
            fit_expectancy_usd=float(ridge_metrics.get("expectancy_usd", 0.0)),
            fit_profit_factor=float(ridge_metrics.get("profit_factor", 0.0)),
            fit_quality_passed=True,
        )
    return None


def _trend_rule_signals(frame: pd.DataFrame) -> np.ndarray:
    signals = np.zeros(len(frame), dtype=np.int8)
    fast = np.asarray(frame.get("ma20_slope", 0.0), dtype=np.float64)
    slow = np.asarray(frame.get("ma50_slope", 0.0), dtype=np.float64)
    signals[(fast > 0.0) & (slow > 0.0)] = 1
    signals[(fast < 0.0) & (slow < 0.0)] = -1
    return signals


def _mean_reversion_rule_signals(frame: pd.DataFrame) -> np.ndarray:
    signals = np.zeros(len(frame), dtype=np.int8)
    spread_z = np.asarray(frame.get("spread_z", 0.0), dtype=np.float64)
    # Simple MR rule: buy when spread_z is surprisingly low/negative, sell when high.
    # In reality, spread_z tracks spread abnormality, but this serves as a basic MR proxy baseline.
    signals[spread_z < -1.0] = 1
    signals[spread_z > 1.0] = -1
    return signals


def _flat_baseline_metrics() -> dict[str, float]:
    """The 'do nothing' floor."""
    return {
        "trade_count": 0.0,
        "expectancy_usd": 0.0,
        "avg_trade_usd": 0.0,
        "profit_factor": 0.0,
        "net_pnl_usd": 0.0,
        "gross_profit_usd": 0.0,
        "gross_loss_usd": 0.0,
        "win_rate": 0.0,
        "avg_win_usd": 0.0,
        "avg_loss_usd": 0.0,
        "win_loss_asymmetry": 0.0,
        "sharpe_like": 0.0,
        "max_drawdown_usd": 0.0,
        "long_trade_count": 0.0,
        "short_trade_count": 0.0,
        "avg_holding_bars": 0.0,
        "trades_per_bar": 0.0,
    }


def run_edge_baseline_research(
    *,
    symbol: str,
    trainable_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    feature_cols: list[str],
    out_path: str | Path,
    horizon_bars: int,
    commission_per_lot: float,
    slippage_pips: float,
    min_edge_pips: float,
    probability_threshold: float,
    probability_margin: float,
    min_trade_count: int,
) -> dict[str, Any]:
    effective_min_trade_count = max(int(min_trade_count) - 1, 1)
    fold_reports: list[dict[str, Any]] = []
    for fold_index, (fold_train, fold_val) in enumerate(folds):
        prepared_train = _prepare_targets(
            fold_train,
            symbol=symbol,
            feature_cols=feature_cols,
            horizon_bars=horizon_bars,
            commission_per_lot=commission_per_lot,
            slippage_pips=slippage_pips,
            min_edge_pips=min_edge_pips,
        )
        prepared_val = _prepare_targets(
            fold_val,
            symbol=symbol,
            feature_cols=feature_cols,
            horizon_bars=horizon_bars,
            commission_per_lot=commission_per_lot,
            slippage_pips=slippage_pips,
            min_edge_pips=min_edge_pips,
        )
        fold_payload: dict[str, Any] = {"fold": int(fold_index), "models": {}}
        if prepared_train.empty or prepared_val.empty:
            fold_payload["blockers"] = ["Insufficient samples after cost-adjusted target construction."]
            fold_reports.append(fold_payload)
            continue
        x_train = prepared_train.loc[:, feature_cols].to_numpy(dtype=np.float64)
        x_val = prepared_val.loc[:, feature_cols].to_numpy(dtype=np.float64)
        logistic_long, logistic_short = _fit_probability_pair(
            x_train,
            prepared_train,
            model_kind="logistic_pair",
        )
        if logistic_long is not None and logistic_short is not None:
            long_val = logistic_long.predict_proba(x_val)[:, 1]
            short_val = logistic_short.predict_proba(x_val)[:, 1]
            fold_payload["models"]["logistic_pair"] = _choose_probability_threshold(
                frame=prepared_val,
                symbol=symbol,
                horizon_bars=horizon_bars,
                long_scores=long_val,
                short_scores=short_val,
                probability_threshold=probability_threshold,
                probability_margin=probability_margin,
            )

        ridge = Pipeline(steps=[("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        ridge.fit(x_train, prepared_train["signed_target"].to_numpy(dtype=np.float64))
        ridge_val = np.asarray(ridge.predict(x_val), dtype=np.float64)
        ridge_signals = np.zeros(len(prepared_val), dtype=np.int8)
        ridge_signals[ridge_val >= min_edge_pips] = 1
        ridge_signals[ridge_val <= -min_edge_pips] = -1
        fold_payload["models"]["ridge_signed_target"] = {
            "threshold": float(min_edge_pips),
            "metrics": _simulate_signals(prepared_val, ridge_signals, symbol=symbol, horizon_bars=horizon_bars),
        }
        tree = DecisionTreeRegressor(max_depth=3, random_state=42)
        tree.fit(x_train, prepared_train["signed_target"].to_numpy(dtype=np.float64))
        tree_val = np.asarray(tree.predict(x_val), dtype=np.float64)
        tree_signals = np.zeros(len(prepared_val), dtype=np.int8)
        tree_signals[tree_val >= min_edge_pips] = 1
        tree_signals[tree_val <= -min_edge_pips] = -1
        fold_payload["models"]["tree_signed_target"] = {
            "threshold": float(min_edge_pips),
            "metrics": _simulate_signals(prepared_val, tree_signals, symbol=symbol, horizon_bars=horizon_bars),
        }
        fold_payload["models"]["trend_rule"] = {
            "threshold": 0.0,
            "metrics": _simulate_signals(
                prepared_val,
                _trend_rule_signals(prepared_val),
                symbol=symbol,
                horizon_bars=horizon_bars,
            ),
        }
        fold_payload["models"]["mean_reversion"] = {
            "threshold": 0.0,
            "metrics": _simulate_signals(
                prepared_val,
                _mean_reversion_rule_signals(prepared_val),
                symbol=symbol,
                horizon_bars=horizon_bars,
            ),
        }
        fold_payload["models"]["flat"] = {
            "threshold": 0.0,
            "metrics": _flat_baseline_metrics(),
        }
        fold_reports.append(fold_payload)

    prepared_trainable = _prepare_targets(
        trainable_frame,
        symbol=symbol,
        feature_cols=feature_cols,
        horizon_bars=horizon_bars,
        commission_per_lot=commission_per_lot,
        slippage_pips=slippage_pips,
        min_edge_pips=min_edge_pips,
    )
    prepared_holdout = _prepare_targets(
        holdout_frame,
        symbol=symbol,
        feature_cols=feature_cols,
        horizon_bars=horizon_bars,
        commission_per_lot=commission_per_lot,
        slippage_pips=slippage_pips,
        min_edge_pips=min_edge_pips,
    )
    holdout_models: dict[str, Any] = {}
    passing_models: list[str] = []
    blockers: list[str] = []
    if prepared_trainable.empty or prepared_holdout.empty:
        blockers.append("Insufficient trainable or holdout samples after cost-adjusted target construction.")
    else:
        x_trainable = prepared_trainable.loc[:, feature_cols].to_numpy(dtype=np.float64)
        x_holdout = prepared_holdout.loc[:, feature_cols].to_numpy(dtype=np.float64)
        logistic_long, logistic_short = _fit_probability_pair(
            x_trainable,
            prepared_trainable,
            model_kind="logistic_pair",
        )
        if logistic_long is not None and logistic_short is not None:
            long_holdout = logistic_long.predict_proba(x_holdout)[:, 1]
            short_holdout = logistic_short.predict_proba(x_holdout)[:, 1]
            selected = _choose_probability_threshold(
                frame=prepared_holdout,
                symbol=symbol,
                horizon_bars=horizon_bars,
                long_scores=long_holdout,
                short_scores=short_holdout,
                probability_threshold=probability_threshold,
                probability_margin=probability_margin,
            )
            holdout_models["logistic_pair"] = selected
            if (
                float(selected["metrics"]["expectancy_usd"]) > 0.0
                and int(selected["metrics"]["trade_count"]) >= effective_min_trade_count
            ):
                passing_models.append("logistic_pair")

        ridge = Pipeline(steps=[("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        ridge.fit(x_trainable, prepared_trainable["signed_target"].to_numpy(dtype=np.float64))
        ridge_holdout = np.asarray(ridge.predict(x_holdout), dtype=np.float64)
        ridge_signals = np.zeros(len(prepared_holdout), dtype=np.int8)
        ridge_signals[ridge_holdout >= min_edge_pips] = 1
        ridge_signals[ridge_holdout <= -min_edge_pips] = -1
        ridge_metrics = _simulate_signals(prepared_holdout, ridge_signals, symbol=symbol, horizon_bars=horizon_bars)
        holdout_models["ridge_signed_target"] = {
            "threshold": float(min_edge_pips),
            "metrics": ridge_metrics,
        }
        if float(ridge_metrics["expectancy_usd"]) > 0.0 and int(ridge_metrics["trade_count"]) >= effective_min_trade_count:
            passing_models.append("ridge_signed_target")

        tree = DecisionTreeRegressor(max_depth=3, random_state=42)
        tree.fit(x_trainable, prepared_trainable["signed_target"].to_numpy(dtype=np.float64))
        tree_holdout = np.asarray(tree.predict(x_holdout), dtype=np.float64)
        tree_signals = np.zeros(len(prepared_holdout), dtype=np.int8)
        tree_signals[tree_holdout >= min_edge_pips] = 1
        tree_signals[tree_holdout <= -min_edge_pips] = -1
        tree_metrics = _simulate_signals(prepared_holdout, tree_signals, symbol=symbol, horizon_bars=horizon_bars)
        holdout_models["tree_signed_target"] = {
            "threshold": float(min_edge_pips),
            "metrics": tree_metrics,
        }
        if float(tree_metrics["expectancy_usd"]) > 0.0 and int(tree_metrics["trade_count"]) >= effective_min_trade_count:
            passing_models.append("tree_signed_target")

        trend_rule_metrics = _simulate_signals(
            prepared_holdout,
            _trend_rule_signals(prepared_holdout),
            symbol=symbol,
            horizon_bars=horizon_bars,
        )
        holdout_models["trend_rule"] = {
            "threshold": 0.0,
            "metrics": trend_rule_metrics,
        }
        if (
            float(trend_rule_metrics["expectancy_usd"]) > 0.0
            and int(trend_rule_metrics["trade_count"]) >= effective_min_trade_count
        ):
            passing_models.append("trend_rule")

        mean_reversion_metrics = _simulate_signals(
            prepared_holdout,
            _mean_reversion_rule_signals(prepared_holdout),
            symbol=symbol,
            horizon_bars=horizon_bars,
        )
        holdout_models["mean_reversion"] = {
            "threshold": 0.0,
            "metrics": mean_reversion_metrics,
        }
        if (
            float(mean_reversion_metrics["expectancy_usd"]) > 0.0
            and int(mean_reversion_metrics["trade_count"]) >= effective_min_trade_count
        ):
            passing_models.append("mean_reversion")

        holdout_models["flat"] = {
            "threshold": 0.0,
            "metrics": _flat_baseline_metrics(),
        }

    report = {
        "symbol": str(symbol).upper(),
        "execution_parity": False,
        "parity_note": (
            "Research baseline economics use simplified non-parity pip math and are "
            "not execution-path identical to RuntimeEngine/ReplayBroker results."
        ),
        "target_definition": {
            "type": "cost_adjusted_tradability",
            "horizon_bars": int(horizon_bars),
            "commission_per_lot": float(commission_per_lot),
            "slippage_pips_per_side": float(slippage_pips),
            "min_edge_pips": float(min_edge_pips),
        },
        "thresholds": {
            "probability_threshold": float(probability_threshold),
            "probability_margin": float(probability_margin),
            "min_trade_count": int(min_trade_count),
            "effective_min_trade_count": int(effective_min_trade_count),
        },
        "fold_metrics": fold_reports,
        "holdout_metrics": {
            "models": holdout_models,
            "blockers": blockers,
        },
        "passing_models": passing_models,
        "gate_passed": bool(passing_models),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
