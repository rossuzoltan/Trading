"""
feature_engine.py  –  Feature engineering for runtime and research compatibility
===============================================================================
The live/RL observation set is intentionally small:
  - log_return
  - body_size
  - candle_range
  - ma20_slope
  - ma50_slope
  - vol_norm_atr
  - spread_z
  - time_delta_z

Legacy raw indicators are still computed for compatibility and future ablation work,
but they are no longer exposed in FEATURE_COLS.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from scipy.signal import lfilter

# ── Public constants ──────────────────────────────────────────────────────────

WARMUP_BARS: int = 150
SCALER_PATH: str = "models/scaler_features.pkl"

FEATURE_COLS: list[str] = [
    "log_return",
    "body_size",
    "candle_range",
    "ma20_slope",
    "ma50_slope",
    "vol_norm_atr",      # ATR/Price -- scale-free volatility
    "spread_z",          # Rolling Z-score of bid-ask spread -- cost awareness
    "time_delta_z",      # Rolling Z-score of seconds between volume bars
    "hour_sin",          # UTC session/regime context
    "hour_cos",
    "day_sin",
    "day_cos",
]


# ── Hurst Exponent ────────────────────────────────────────────────────────────

def _hurst_single(series: np.ndarray) -> float:
    """
    Compute Hurst exponent via Rescaled Range (R/S) analysis.
    Returns:
        H > 0.5 : trending (persistence)
        H ≈ 0.5 : random walk
        H < 0.5 : mean-reverting (anti-persistence)
    """
    n = len(series)
    if n < 20:
        return 0.5   # not enough data

    # Compute R/S for multiple sub-period lengths
    lags    = np.unique(np.logspace(1, np.log10(n // 2), 10).astype(int))
    lags    = lags[lags >= 10]
    rs_vals = []

    for lag in lags:
        chunks = [series[i:i + lag] for i in range(0, n - lag, lag)]
        rs_list = []
        for chunk in chunks:
            mean  = np.mean(chunk)
            dev   = np.cumsum(chunk - mean)
            r     = np.max(dev) - np.min(dev)
            s     = np.std(chunk, ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            rs_vals.append((lag, np.mean(rs_list)))

    if len(rs_vals) < 2:
        return 0.5

    lags_arr, rs_arr = zip(*rs_vals)
    log_lags = np.log(lags_arr)
    log_rs   = np.log(rs_arr)
    # Linear fit: H = slope
    try:
        H = float(np.polyfit(log_lags, log_rs, 1)[0])
        return float(np.clip(H, 0.0, 1.0))
    except Exception:
        return 0.5

# ── Fractional Differentiation (Phase 13: Memory Preservation) ───────────────

def _frac_diff_weights(d: float, size: int) -> np.ndarray:
    """Compute weights for the fractional difference operator."""
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1], dtype=np.float64)

def _apply_frac_diff(series: np.ndarray, d: float, window: int = 50) -> np.ndarray:
    """Apply fractional differentiation with a fixed-window cutoff."""
    if len(series) < window:
        return np.full(len(series), np.nan, dtype=np.float64)
    w = _frac_diff_weights(d, window)
    res = np.convolve(series, w, mode="valid")
    # Pad with NaNs to maintain original length
    return np.concatenate([np.full(window - 1, np.nan), res])


def _rolling_hurst(close: pd.Series, window: int = 100, *, latest_only: bool = False) -> pd.Series:
    """Apply Hurst exponent over a rolling window."""
    values = close.values
    result = np.full(len(values), 0.5)
    if latest_only:
        if len(values) >= window:
            result[-1] = _hurst_single(values[-window:])
        out = pd.Series(result, index=close.index)
        out.iloc[:window - 1] = np.nan
        return out
    for i in range(window, len(values) + 1):
        result[i - 1] = _hurst_single(values[i - window:i])
    out = pd.Series(result, index=close.index)
    out.iloc[:window - 1] = np.nan
    return out


# ── Core indicator computation ────────────────────────────────────────────────

def _compute_raw(df: pd.DataFrame, *, latest_only_hurst: bool = False) -> pd.DataFrame:
    """
    Compute all indicators. Handles both time bars and volume bars.
    Input:  OHLCV DataFrame (DatetimeIndex UTC or 'Gmt time' column).
    Output: DataFrame with all indicator columns added.
    """
    df = df.copy()

    # ── UTC DatetimeIndex ────────────────────────────────────────────────────
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Gmt time" in df.columns:
            df["Gmt time"] = pd.to_datetime(df["Gmt time"], utc=True, errors="coerce")
            df = df.set_index("Gmt time")
        else:
            df.index = pd.date_range("2020-01-01", periods=len(df), freq="h", tz="UTC")
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # ── Coerce to numeric ────────────────────────────────────────────────────
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except TypeError:
            pass
    df = df.select_dtypes(include=[np.number])

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = df[col].astype(np.float64)

    # ── Temporal features (UTC cyclical) ─────────────────────────────────────
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["day_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df["day_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 5)

    # ── Log return ───────────────────────────────────────────────────────────
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # ── Realised volatility ──────────────────────────────────────────────────
    df["log_return_std"] = df["log_return"].rolling(20).std()

    # ── ATR ──────────────────────────────────────────────────────────────────
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # ── Volatility-normalised ATR: ATR / Close (scale-free) ─────────────────
    df["vol_norm_atr"] = df["atr_14"] / df["Close"].replace(0, np.nan)

    # ── Momentum ─────────────────────────────────────────────────────────────
    df["rsi_14"] = ta.rsi(df["Close"], length=14)

    macd_df = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        df["macd"]  = macd_df.iloc[:, 0]
        df["macdh"] = macd_df.iloc[:, 1]
    else:
        df["macd"] = np.nan; df["macdh"] = np.nan

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    bb_df = ta.bbands(df["Close"], length=20, std=2)
    if bb_df is not None and not bb_df.empty:
        df["bb_bw"]  = bb_df.iloc[:, 3]
        df["bb_pct"] = bb_df.iloc[:, 4]
    else:
        df["bb_bw"] = np.nan; df["bb_pct"] = np.nan

    # ── ADX ──────────────────────────────────────────────────────────────────
    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx_df is not None and not adx_df.empty:
        df["adx"] = adx_df.iloc[:, 0]
    else:
        df["adx"] = np.nan

    # ── Price action (ATR-normalised) ─────────────────────────────────────────
    atr = df["atr_14"].replace(0, np.nan)
    df["body_size"]    = (df["Close"] - df["Open"]) / atr
    df["candle_range"] = (df["High"]  - df["Low"])  / atr

    # ── MA slopes ────────────────────────────────────────────────────────────
    df["ma20"] = pd.to_numeric(ta.sma(df["Close"], length=20), errors="coerce")
    df["ma50"] = pd.to_numeric(ta.sma(df["Close"], length=50), errors="coerce")
    df["ma20_slope"] = df["ma20"].diff() / atr
    df["ma50_slope"] = df["ma50"].diff() / atr

    # ── Hurst Exponent (rolling 100 bars) ────────────────────────────────────
    # Only compute if we have enough rows (slow — uses vectorised R/S)
    if len(df) >= 100:
        df["hurst_exp"] = _rolling_hurst(df["Close"], window=100, latest_only=latest_only_hurst)
    else:
        df["hurst_exp"] = 0.5   # default: random walk assumption

    # ── Spread Z-score (from Dukascopy avg_spread column, or fallback 0) ─────
    if "avg_spread" not in df.columns:
        df["avg_spread"] = 0.0
    df["avg_spread"] = df["avg_spread"].fillna(0.0)
    # Z-score via rolling window (200 bars) so it's adaptive
    spread_roll = df["avg_spread"].rolling(200, min_periods=20)
    spread_mean = spread_roll.mean()
    spread_std  = spread_roll.std().replace(0, np.nan)
    df["spread_z"] = (df["avg_spread"] - spread_mean) / spread_std
    df["spread_z"]  = df["spread_z"].fillna(0.0)

    # ── Time-Delta Z-score (Phase 12: volume bar alpha signal) ───────────────
    # For time bars this is a constant; for volume bars it reveals liquidity.
    # A bar that formed in 400ms = massive block order. 4 hours = quiet market.
    if "time_delta_s" not in df.columns:
        df["time_delta_s"] = 3600.0   # fallback: assume 1-hour time bars
    df["time_delta_s"] = df["time_delta_s"].replace(0, np.nan).fillna(3600.0)
    td_roll = df["time_delta_s"].rolling(200, min_periods=20)
    td_mean = td_roll.mean()
    td_std  = td_roll.std().replace(0, np.nan)
    df["time_delta_z"] = ((df["time_delta_s"] - td_mean) / td_std).fillna(0.0)
    df["time_delta_z"] = df["time_delta_z"].clip(-5.0, 5.0)   # cap extremes

    # ── Fractional Differentiation (Phase 13: The Alpha Booster) ──────────────
    # Preserve memory (d=0.3) on log-prices
    log_p    = np.log(df["Close"].values)
    fd_vals  = _apply_frac_diff(log_p, d=0.3, window=30)
    fd_series = pd.Series(fd_vals, index=df.index)
    fd_roll   = fd_series.rolling(200, min_periods=20)
    fd_mean   = fd_roll.mean()
    fd_std    = fd_roll.std().replace(0, np.nan)
    df["frac_diff_z"] = ((fd_series - fd_mean) / fd_std).fillna(0.0)
    df["frac_diff_z"] = df["frac_diff_z"].clip(-5.0, 5.0)

    return df


# ── Main class ────────────────────────────────────────────────────────────────

class FeatureEngine:
    """
    Unified feature pipeline for the reduced v1 observation set.

    Training:
        engine = FeatureEngine()
        df_out, cols = engine.fit_transform(train_df)

    Live:
        engine = FeatureEngine.load("models/scaler_EURUSD.pkl")
        engine.warm_up(history_df)
        engine.push(new_bar_series)
        obs = engine.latest_observation   # shape (8,) float32
    """

    def __init__(self) -> None:
        self._scaler: StandardScaler | None = None
        self._buffer: pd.DataFrame | None   = None
        self._raw_buffer: pd.DataFrame | None = None

    def fit_transform(
        self, df: pd.DataFrame, scaler_path: str = SCALER_PATH
    ) -> tuple[pd.DataFrame, list[str]]:
        raw = _compute_raw(df)
        raw = self._drop_invalid_feature_rows(raw)

        if raw.empty:
            raise ValueError(f"DataFrame empty after indicators. Need ≥{WARMUP_BARS} rows.")

        self._scaler = StandardScaler()
        feature_block = raw.loc[:, FEATURE_COLS]
        self._scaler.fit(feature_block)
        transformed = raw.copy()
        transformed.loc[:, FEATURE_COLS] = self._scaler.transform(feature_block)

        os.makedirs(os.path.dirname(scaler_path) if os.path.dirname(scaler_path) else ".", exist_ok=True)
        joblib.dump(self._scaler, scaler_path)
        print(f"[FeatureEngine] Scaler fitted on {len(raw)} rows → {scaler_path}")
        return transformed, FEATURE_COLS

    def save_scaler(self, path: str) -> None:
        if self._scaler is None:
            raise RuntimeError("No scaler fitted.")
        joblib.dump(self._scaler, path)

    def _drop_invalid_feature_rows(self, raw: pd.DataFrame) -> pd.DataFrame:
        required = [c for c in FEATURE_COLS if c in raw.columns]
        drop_cols = [c for c in required if c in raw.columns]
        cleaned = raw.copy()
        cleaned.dropna(subset=drop_cols if drop_cols else None, inplace=True)
        return cleaned

    @classmethod
    def load(cls, path: str = SCALER_PATH) -> "FeatureEngine":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler not found at '{path}'. Run training first.")
        engine = cls()
        engine._scaler = joblib.load(path)
        print(f"[FeatureEngine] Scaler loaded from {path}")
        return engine

    @classmethod
    def from_scaler(cls, scaler: StandardScaler) -> "FeatureEngine":
        engine = cls()
        engine._scaler = scaler
        return engine

    @classmethod
    def load_for_symbol(cls, symbol: str, model_dir: str = "models") -> "FeatureEngine":
        return cls.load(os.path.join(model_dir, f"scaler_{symbol}.pkl"))

    def warm_up(self, history: pd.DataFrame) -> None:
        if len(history) < WARMUP_BARS:
            raise ValueError(f"warm_up() needs ≥{WARMUP_BARS} rows, got {len(history)}.")
        base_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "avg_spread", "time_delta_s"] if c in history.columns]
        history_rows = max(WARMUP_BARS * 3, 300)
        self._raw_buffer = history[base_cols].copy().iloc[-history_rows:]
        raw = _compute_raw(history)
        raw = self._drop_invalid_feature_rows(raw)
        if raw.empty:
            raise ValueError("warm_up() produced no valid feature rows after indicator/scaler filtering.")
        self._buffer = raw

    def push(self, bar: pd.Series) -> None:
        if self._buffer is None or self._raw_buffer is None:
            raise RuntimeError("Call warm_up() before push().")
        new_row = pd.DataFrame([bar])
        new_row.index = pd.DatetimeIndex([bar.name])
        base_cols = [
            col for col in ["Open", "High", "Low", "Close", "Volume", "avg_spread", "time_delta_s"]
            if col in self._raw_buffer.columns or col in new_row.columns
        ]
        combined = pd.concat([self._raw_buffer[base_cols], new_row.reindex(columns=base_cols)], sort=False)
        combined = combined[~combined.index.duplicated(keep="last")]
        history_rows = max(WARMUP_BARS * 3, 300)
        self._raw_buffer = combined.iloc[-history_rows:]
        raw = _compute_raw(self._raw_buffer, latest_only_hurst=True)
        raw = self._drop_invalid_feature_rows(raw)
        if raw.empty:
            raise RuntimeError("Feature buffer became empty after push().")
        self._buffer = raw.iloc[-history_rows:]

    @property
    def latest_observation(self) -> np.ndarray:
        if self._buffer is None or self._buffer.empty:
            raise RuntimeError("No observations. Call warm_up() first.")
        row = self._buffer.iloc[-1].copy()

        if self._scaler is not None:
            feature_frame = pd.DataFrame(
                [{col: float(row.get(col, 0.0)) for col in FEATURE_COLS}],
                columns=FEATURE_COLS,
            )
            obs = self._scaler.transform(feature_frame)[0].astype(np.float32)
        else:
            obs = np.array([row.get(c, 0.0) for c in FEATURE_COLS], dtype=np.float32)
        if np.any(np.isnan(obs)):
            obs = np.nan_to_num(obs, nan=0.0)
        return obs
