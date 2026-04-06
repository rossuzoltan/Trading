import pytest
import numpy as np
import pandas as pd
from runtime_common import build_simple_action_map
from runtime.rule_based_policy import RuleBasedPolicy

class MockFeatureEngine:
    def __init__(self, buffer):
        self._buffer = buffer

def test_rule_based_policy_decisions():
    action_map = build_simple_action_map()
    
    # 1. Bullish scenario: MA20 > MA50
    # Must have >= 50 rows to pass the safety check in the policy
    buffer_bull = pd.DataFrame({
        "ma20": [1.0] * 49 + [1.1],
        "ma50": [1.0] * 50
    })
    fe_bull = MockFeatureEngine(buffer_bull)
    policy_bull = RuleBasedPolicy(fe_bull, action_map)
    
    # Flat observation (direction = 0 at index 12 / -4)
    obs_flat = np.zeros((1, 16)) 
    mask = np.ones(4, dtype=bool)
    
    idx, action = policy_bull.decide(obs_flat, mask)
    # Target direction is 1 (Long). Since flat (0), should return Open Long idx.
    # Map index 2 is OPEN LONG in build_simple_action_map.
    assert idx == 2
    assert action.direction == 1
    
    # 2. Bearish scenario: MA20 < MA50
    buffer_bear = pd.DataFrame({
        "ma20": [1.0] * 49 + [0.9],
        "ma50": [1.0] * 50
    })
    fe_bear = MockFeatureEngine(buffer_bear)
    policy_bear = RuleBasedPolicy(fe_bear, action_map)
    
    idx, action = policy_bear.decide(obs_flat, mask)
    # Target direction is -1 (Short). Map index 3 is OPEN SHORT.
    assert idx == 3
    assert action.direction == -1

if __name__ == "__main__":
    pytest.main([__file__])
