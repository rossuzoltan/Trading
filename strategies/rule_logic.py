"""
strategies/rule_logic.py
-----------------------
Single source of truth for deterministic trading rules.
Ensures parity between evaluate_oos backtests and live rule_selector execution.
"""
from typing import Any, Dict

def compute_mean_reversion_direction(features: Dict[str, Any], params: Dict[str, Any]) -> int:
    """
    Computes desired direction based on spread_z mean reversion logic.
    Z <= -threshold -> LONG (1)
    Z >= threshold  -> SHORT (-1)
    Else            -> FLAT (0)
    """
    threshold = float(params.get("threshold", 1.0))
    spread_z = float(features.get("spread_z", 0.0) or 0.0)
    
    if spread_z <= -threshold:
        return 1
    if spread_z >= threshold:
        return -1
    return 0

def compute_trend_direction(features: Dict[str, Any], params: Dict[str, Any]) -> int:
    """
    Computes desired direction based on MA slope trend logic.
    """
    ma20_slope = float(features.get("ma20_slope", 0.0) or 0.0)
    ma50_slope = float(features.get("ma50_slope", 0.0) or 0.0)
    
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
    threshold = float(params.get("threshold", 1.0))
    price_z = float(features.get("price_z", 0.0) or 0.0)

    if price_z <= -threshold:
        return 1
    if price_z >= threshold:
        return -1
    return 0


def compute_price_mr_spread_filter(features: Dict[str, Any], params: Dict[str, Any]) -> int:
    """
    Variant C: price_z trigger + spread stability filter.
    Same signal as price_mean_reversion, but only fires when
    spread_z < 0 (spread is at or below its rolling average = stable cost environment).
    """
    spread_z = float(features.get("spread_z", 0.0) or 0.0)
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
    - Only trades when ADX < 25 (Ranging market)
    - Triggers when price is extended (price_z) AND momentum is exhausted (rsi)
    """
    adx = float(features.get("adx", 0.0) or 0.0)
    rsi = float(features.get("rsi_14", 50.0) or 50.0)
    price_z = float(features.get("price_z", 0.0) or 0.0)

    # Do not trade if market is strongly trending (ADX > 25)
    if adx > 25.0:
        return 0

    # LONG Condition: Price is very low compared to MA, and RSI is oversold
    if price_z <= -1.5 and rsi < 35.0:
        return 1
    
    # SHORT Condition: Price is very high compared to MA, and RSI is overbought
    if price_z >= 1.5 and rsi > 65.0:
        return -1
        
    return 0


def compute_vol_breakout(features: Dict[str, Any], params: Dict[str, Any]) -> int:
    """
    Volatility Breakout logic using Bollinger Bands.
    """
    bb_pct = float(features.get("bb_pct", 0.5) or 0.5)
    mean_revert = bool(params.get("mean_revert", False))
    threshold_up = float(params.get("threshold_up", 1.0))
    threshold_down = float(params.get("threshold_down", 0.0))
    
    direction = 0
    if bb_pct >= threshold_up:
        direction = 1
    elif bb_pct <= threshold_down:
        direction = -1
        
    return -direction if mean_revert else direction


# Registry of available rule families
RULE_REGISTRY = {
    "mean_reversion": compute_mean_reversion_direction,
    "trend": compute_trend_direction,
    # Challenger variants — for exact-runtime A/B only
    "price_mean_reversion": compute_price_mean_reversion,
    "price_mr_spread_filter": compute_price_mr_spread_filter,
    "combined_mr": compute_combined_mr,
    "pro_mean_reversion": compute_pro_mean_reversion,
    "volatility_breakout": compute_vol_breakout,
}

def compute_rule_direction(rule_family: str, features: Dict[str, Any], params: Dict[str, Any]) -> int:
    """Dispatches to the specific rule logic function."""
    func = RULE_REGISTRY.get(rule_family)
    if not func:
        raise ValueError(f"Unknown rule family: {rule_family}")
    return func(features, params)

