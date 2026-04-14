"""
strategies/rule_logic.py
-----------------------
Single source of truth for deterministic trading rules.
Ensures parity between evaluate_oos backtests and live rule_selector execution.
"""
from typing import Any, Dict


def _feature_value(features: Dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(features.get(key, default) or default)


def _regime_filter_passes(features: Dict[str, Any], params: Dict[str, Any]) -> bool:
    """
    Optional regime guardrails.
    All thresholds are disabled when set to 0 (default), so existing behavior
    is preserved unless a manifest explicitly opts in.
    """
    min_vol_norm_atr = float(params.get("min_vol_norm_atr", 0.0) or 0.0)
    max_vol_norm_atr = float(params.get("max_vol_norm_atr", 0.0) or 0.0)
    max_abs_log_return = float(params.get("max_abs_log_return", 0.0) or 0.0)
    max_abs_body_size = float(params.get("max_abs_body_size", 0.0) or 0.0)
    max_candle_range = float(params.get("max_candle_range", 0.0) or 0.0)

    vol_norm_atr = abs(_feature_value(features, "vol_norm_atr", 0.0))
    log_return = abs(_feature_value(features, "log_return", 0.0))
    body_size = abs(_feature_value(features, "body_size", 0.0))
    candle_range = abs(_feature_value(features, "candle_range", 0.0))

    if min_vol_norm_atr > 0.0 and vol_norm_atr < min_vol_norm_atr:
        return False
    if max_vol_norm_atr > 0.0 and vol_norm_atr > max_vol_norm_atr:
        return False
    if max_abs_log_return > 0.0 and log_return > max_abs_log_return:
        return False
    if max_abs_body_size > 0.0 and body_size > max_abs_body_size:
        return False
    if max_candle_range > 0.0 and candle_range > max_candle_range:
        return False
    return True

def compute_mean_reversion_direction(features: Dict[str, Any], params: Dict[str, Any]) -> int:
    """
    Computes desired direction using price extension with cost/regime guards.
    This keeps the direction tied to price dislocation while using spread and slope
    only as authorization filters.
    """
    long_threshold = float(params.get("long_threshold", -params.get("threshold", 1.5)))
    short_threshold = float(params.get("short_threshold", params.get("threshold", 1.5)))
    max_spread_z = float(params.get("max_spread_z", 0.5))
    max_time_delta_z = float(params.get("max_time_delta_z", 2.0))
    max_abs_ma20_slope = float(params.get("max_abs_ma20_slope", 0.15))
    max_abs_ma50_slope = float(params.get("max_abs_ma50_slope", 0.08))

    price_z = _feature_value(features, "price_z", 0.0)
    spread_z = _feature_value(features, "spread_z", 0.0)
    time_delta_z = _feature_value(features, "time_delta_z", 0.0)
    ma20_slope = _feature_value(features, "ma20_slope", 0.0)
    ma50_slope = _feature_value(features, "ma50_slope", 0.0)

    if not _regime_filter_passes(features, params):
        return 0

    if spread_z > max_spread_z:
        return 0
    if abs(time_delta_z) > max_time_delta_z:
        return 0
    if abs(ma20_slope) > max_abs_ma20_slope:
        return 0
    if abs(ma50_slope) > max_abs_ma50_slope:
        return 0

    if price_z <= long_threshold:
        return 1
    if price_z >= short_threshold:
        return -1
    return 0

def compute_trend_direction(features: Dict[str, Any], params: Dict[str, Any]) -> int:
    """
    Computes desired direction based on MA slope trend logic.
    """
    ma20_slope = _feature_value(features, "ma20_slope", 0.0)
    ma50_slope = _feature_value(features, "ma50_slope", 0.0)
    
    if ma20_slope > 0.0 and ma50_slope > 0.0:
        return 1
    if ma20_slope < 0.0 and ma50_slope < 0.0:
        return -1
    return 0


# ── Challenger variants (for A/B experiment — do NOT promote without evidence) ──

def compute_price_mean_reversion(features: Dict[str, Any], params: Dict[str, Any]) -> int:
    """
    Variant B: Pure price-based mean reversion.
    Uses price_z (Close - MA20) / std instead of spread_z.
    Z <= -threshold -> LONG (1)
    Z >= +threshold -> SHORT (-1)
    """
    long_threshold = float(params.get("long_threshold", -params.get("threshold", 1.0)))
    short_threshold = float(params.get("short_threshold", params.get("threshold", 1.0)))
    price_z = _feature_value(features, "price_z", 0.0)

    if price_z <= long_threshold:
        return 1
    if price_z >= short_threshold:
        return -1
    return 0


def compute_price_mr_spread_filter(features: Dict[str, Any], params: Dict[str, Any]) -> int:
    """
    Variant C: price_z trigger + spread stability filter.
    Same signal as price_mean_reversion, but only fires when
    spread_z < 0 (spread is at or below its rolling average = stable cost environment).
    """
    spread_z = _feature_value(features, "spread_z", 0.0)
    if spread_z >= 0.0:
        return 0  # spread elevated — skip
    return compute_price_mean_reversion(features, params)


def compute_combined_mr(features: Dict[str, Any], params: Dict[str, Any]) -> int:
    """
    Variant D: Both spread_z AND price_z must confirm the same direction.
    Requires agreement between the cost-based and price-based signals.
    """
    s = compute_mean_reversion_direction(features, params)   # spread_z based
    p = compute_price_mean_reversion(features, params)       # price_z based
    if s == p and s != 0:
        return s
    return 0


def compute_pro_mean_reversion(features: Dict[str, Any], params: Dict[str, Any]) -> int:
    """
    Variant E (Pro Bot): Authentic Price Action Mean Reversion
    - Only trades when ADX < adx_threshold (Ranging market)
    - Triggers when price is extended (price_z) AND momentum is exhausted (rsi)
    """
    adx_threshold = float(params.get("adx_threshold", 25.0))
    rsi_oversold = float(params.get("rsi_oversold", 35.0))
    rsi_overbought = float(params.get("rsi_overbought", 65.0))
    long_pz = float(params.get("long_pz", -params.get("price_z_threshold", 1.5)))
    short_pz = float(params.get("short_pz", params.get("price_z_threshold", 1.5)))
    
    adx = _feature_value(features, "adx", 0.0)
    rsi = _feature_value(features, "rsi_14", 50.0)
    price_z = _feature_value(features, "price_z", 0.0)

    if not _regime_filter_passes(features, params):
        return 0

    # Optional hurst filter
    hurst_filter = bool(params.get("hurst_filter", False))
    if hurst_filter:
        hurst_exp = _feature_value(features, "hurst_exp", 0.5)
        if hurst_exp >= 0.5:
            return 0  # Market is trending, skip mean reversion

    # Do not trade if market is strongly trending
    if adx > adx_threshold:
        return 0

    # LONG Condition: Price is very low compared to MA, and RSI is oversold
    if price_z <= long_pz and rsi < rsi_oversold:
        return 1
    
    # SHORT Condition: Price is very high compared to MA, and RSI is overbought
    if price_z >= short_pz and rsi > rsi_overbought:
        return -1
        
    return 0


def compute_macd_trend(features: Dict[str, Any], params: Dict[str, Any]) -> int:
    """
    MACD Trend Following Rule.
    Uses MACD histogram momentum combined with MA slope alignment and optional ADX/Hurst filters.
    """
    macd = _feature_value(features, "macd", 0.0)
    macdh = _feature_value(features, "macdh", 0.0)
    ma20_slope = _feature_value(features, "ma20_slope", 0.0)
    ma50_slope = _feature_value(features, "ma50_slope", 0.0)

    if not _regime_filter_passes(features, params):
        return 0
    
    # Optional adx trend filter
    adx_trend_threshold = float(params.get("adx_trend_threshold", 0.0))
    if adx_trend_threshold > 0.0:
        adx = _feature_value(features, "adx", 0.0)
        if adx < adx_trend_threshold:
            return 0

    # Optional hurst filter
    hurst_filter = bool(params.get("hurst_filter", False))
    if hurst_filter:
        hurst_exp = _feature_value(features, "hurst_exp", 0.5)
        if hurst_exp <= 0.5:
            return 0  # Market is mean-reverting, skip trend

    require_ma_alignment = bool(params.get("require_ma_alignment", True))
    
    # MACD Histogram threshold (looking for expanding momentum)
    macdh_threshold = float(params.get("macdh_threshold", 0.0001))

    long_ma_ok = (ma20_slope > 0.0 and ma50_slope > 0.0) if require_ma_alignment else True
    short_ma_ok = (ma20_slope < 0.0 and ma50_slope < 0.0) if require_ma_alignment else True

    # LONG
    if long_ma_ok and macdh > macdh_threshold and macd > 0:
        return 1
    
    # SHORT
    if short_ma_ok and macdh < -macdh_threshold and macd < 0:
        return -1
        
    return 0


def compute_vol_breakout(features: Dict[str, Any], params: Dict[str, Any]) -> int:
    """
    Volatility Breakout logic using Bollinger Bands.
    """
    bb_pct = _feature_value(features, "bb_pct", 0.5)
    mean_revert = bool(params.get("mean_revert", False))
    threshold_up = float(params.get("threshold_up", 1.0))
    threshold_down = float(params.get("threshold_down", 0.0))

    if not _regime_filter_passes(features, params):
        return 0
    
    direction = 0
    if bb_pct >= threshold_up:
        direction = 1
    elif bb_pct <= threshold_down:
        direction = -1
        
    return -direction if mean_revert else direction


def compute_microstructure_bounce(features: Dict[str, Any], params: Dict[str, Any]) -> int:
    """
    Microstructure Bounce Hypothesis
    Trigger: Extreme low time_delta_z (high-speed volume bars representing liquidity vacuums) 
             + Price Extension (price_z).
    Spread is used strictly as a cost filter.
    """
    td_threshold = float(params.get("td_threshold", -2.0))
    long_pz = float(params.get("long_pz", -params.get("price_z_threshold", 1.5)))
    short_pz = float(params.get("short_pz", params.get("price_z_threshold", 1.5)))
    spread_max_z = float(params.get("spread_max_z", 1.0))
    
    time_delta_z = _feature_value(features, "time_delta_z", 0.0)
    price_z = _feature_value(features, "price_z", 0.0)
    spread_z = _feature_value(features, "spread_z", 0.0)

    if not _regime_filter_passes(features, params):
        return 0
    
    # Needs to be a high-speed event (time_delta_z must be smaller/more negative than threshold)
    if time_delta_z > td_threshold:
        return 0
        
    # Cost filtering
    if spread_z > spread_max_z:
        return 0

    # Exhaustion LONG: price stretched DOWNwards
    if price_z <= long_pz:
        return 1
        
    # Exhaustion SHORT: price stretched UPwards
    if price_z >= short_pz:
        return -1
        
    return 0


# Registry of available rule families
RULE_REGISTRY = {
    "mean_reversion": compute_mean_reversion_direction,
    "trend": compute_trend_direction,
    # Challenger variants — for exact-runtime A/B only
    "price_mean_reversion": compute_price_mean_reversion,
    "price_mr_spread_filter": compute_price_mr_spread_filter,
    "combined_mr": compute_combined_mr,
    "pro_mean_reversion": compute_pro_mean_reversion,
    "macd_trend": compute_macd_trend,
    "volatility_breakout": compute_vol_breakout,
    "microstructure_bounce": compute_microstructure_bounce,
}

def compute_rule_direction(rule_family: str, features: Dict[str, Any], params: Dict[str, Any]) -> int:
    """Dispatches to the specific rule logic function."""
    func = RULE_REGISTRY.get(rule_family)
    if not func:
        raise ValueError(f"Unknown rule family: {rule_family}")
    return func(features, params)
