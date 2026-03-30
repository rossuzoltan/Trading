from enum import Enum


class ActionType(str, Enum):
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    OPEN = "OPEN"


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
