from __future__ import annotations

from typing import Any, Sequence
import numpy as np
from domain.enums import ActionType
from domain.models import ActionSpec

class RuleBasedPolicy:
    """
    Deterministic trend-following policy for baseline A/B testing.
    Uses MA20 vs MA50 crossover logic.
    """
    def __init__(self, feature_engine: Any, action_map: Sequence[ActionSpec]) -> None:
        self.feature_engine = feature_engine
        self.action_map = action_map
        
        # Identify standard actions in the map
        self.hold_idx = 0
        self.close_idx = 1
        self.long_idx = None
        self.short_idx = None
        
        for i, action in enumerate(action_map):
            if action.action_type == ActionType.OPEN:
                if action.direction > 0 and self.long_idx is None:
                    self.long_idx = i
                elif action.direction < 0 and self.short_idx is None:
                    self.short_idx = i

    def decide(self, observation: np.ndarray, mask: np.ndarray) -> tuple[int, ActionSpec]:
        """
        Decide action based on MA20 vs MA50.
        Note: We access the feature_engine buffer directly for raw MA values
        since they are not exposed in the standard observation vector.
        """
        # 1. Get raw MA values from feature engine
        # We assume the buffer is already updated by the engine before decide() is called.
        buffer = self.feature_engine._buffer
        if buffer is None or len(buffer) < 50:
            # Not enough data for MA50, stay flat
            idx = self.hold_idx if mask[self.hold_idx] else self.close_idx
            return idx, self.action_map[idx]

        last_row = buffer.iloc[-1]
        # We check if 'ma20' and 'ma50' are in the buffer. 
        # FeatureEngine computes them in _compute_raw.
        ma20 = last_row.get("ma20", 0.0)
        ma50 = last_row.get("ma50", 0.0)
        
        target_direction = 0
        if ma20 > ma50:
            target_direction = 1
        elif ma20 < ma50:
            target_direction = -1
            
        # 2. Map target direction to action index
        # Logic:
        # - If already in target direction: HOLD
        # - If in opposite direction: CLOSE
        # - If flat: OPEN target direction
        
        # We need current position. In RuntimeEngine, the policy doesn't know the position
        # except through the observation vector (state_block part).
        # Observation structure: [features...] + [direction, time, pnl_sign, last_reward]
        current_direction = int(observation[-1, -4]) if observation.ndim > 1 else int(observation[-4])
        
        if current_direction == target_direction:
            idx = self.hold_idx
        elif target_direction == 0:
            # We want to be flat
            idx = self.close_idx if current_direction != 0 else self.hold_idx
        elif current_direction == 0:
            # Flat, want to open
            idx = self.long_idx if target_direction > 0 else self.short_idx
        else:
            # In opposite direction
            idx = self.close_idx
            
        # 3. Apply mask safety
        if not mask[idx]:
            idx = self.hold_idx if mask[self.hold_idx] else self.close_idx
            
        return int(idx), self.action_map[int(idx)]
