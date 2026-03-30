import pytest
import numpy as np
import pandas as pd
from trading_env import ForexTradingEnv

def test_per_bar_bonus_accumulation():
    # Setup a dummy DF with no volatility to prevent SL/TP hits
    df = pd.DataFrame({
        "Open": [1.0] * 20,
        "High": [1.0] * 20, # No volatility
        "Low": [1.0] * 20,  # No volatility
        "Close": [1.0] * 20,
        "Volume": [100] * 20,
        "atr_14": [0.01] * 20,
    })
    
    bonus_val = 1.0
    env = ForexTradingEnv(
        df=df,
        feature_columns=["Close"],
        sl_options=[10.0], # Very far SL
        tp_options=[10.0], # Very far TP
        participation_bonus_value=bonus_val,
        random_start=False
    )
    
    env.reset()
    
    # 1. First step: Open a position
    # Action 2 is OPEN LONG in simple map
    env.step(2) 
    
    # 2. Subsequent steps: HOLD (5 times)
    for _ in range(5):
        env.step(0) # HOLD
        
    # After 5 holds, we should have 5 * (bonus_val/10) = 0.5 bonus
    bonus_sum = env.reward_stats.get("participation_bonus_sum", 0.0)
    assert bonus_sum == pytest.approx(0.5)

if __name__ == "__main__":
    pytest.main([__file__])
