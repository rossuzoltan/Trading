from __future__ import annotations

from typing import Any

import numpy as np

def action_mask_fn(env):
    masks: Any = env.action_masks()
    if isinstance(masks, (list, tuple)):
        try:
            return np.stack(masks).astype(bool, copy=False)
        except Exception:
            return np.asarray(masks, dtype=bool)
    return masks
