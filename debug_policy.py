from runtime.rule_based_policy import RuleBasedPolicy
import numpy as np
import pandas as pd
from runtime_common import build_simple_action_map

class MockFE:
    def __init__(self, buffer):
        self._buffer = buffer

fe = MockFE(pd.DataFrame({"ma20": [1.1], "ma50": [1.0]}))
map = build_simple_action_map()
p = RuleBasedPolicy(fe, map)

obs = np.zeros((1, 16))
# Set direction to 0 (already 0)
print(f"Obs Shape: {obs.shape}")
print(f"Obs Direction Value: {obs[-1, -4]}")

idx, action = p.decide(obs, mask=np.ones(4, dtype=bool))
print(f"Index: {idx}, Action: {action}")
