from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from symbol_utils import price_to_pips


STATE_FEATURE_COUNT = 4


class ActionType(str, Enum):
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    OPEN = "OPEN"


@dataclass(frozen=True)
class ActionSpec:
    action_type: ActionType
    direction: int | None = None
    sl_value: float | None = None
    tp_value: float | None = None


@dataclass
class ConfirmedPosition:
    direction: int = 0
    entry_price: float | None = None
    sl_price: float | None = None
    tp_price: float | None = None
    volume: float = 0.0
    broker_ticket: int | None = None
    order_id: int | None = None
    time_in_trade_bars: int = 0
    last_reward: float = 0.0
    last_confirmed_price: float | None = None
    last_confirmed_time_msc: int | None = None

    @property
    def is_flat(self) -> bool:
        return self.direction == 0 or self.entry_price is None or self.volume <= 0

    def reset(self) -> None:
        self.direction = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.volume = 0.0
        self.broker_ticket = None
        self.order_id = None
        self.time_in_trade_bars = 0
        self.last_confirmed_price = None
        self.last_confirmed_time_msc = None


def build_action_map(sl_options: Sequence[float], tp_options: Sequence[float]) -> tuple[ActionSpec, ...]:
    actions: list[ActionSpec] = [
        ActionSpec(ActionType.HOLD),
        ActionSpec(ActionType.CLOSE),
    ]
    for direction in (1, -1):
        for sl_value in sl_options:
            for tp_value in tp_options:
                actions.append(
                    ActionSpec(
                        action_type=ActionType.OPEN,
                        direction=direction,
                        sl_value=float(sl_value),
                        tp_value=float(tp_value),
                    )
                )
    return tuple(actions)


def serialize_action_map(action_map: Sequence[ActionSpec]) -> list[dict[str, float | int | str | None]]:
    return [
        {
            "action_type": action.action_type.value,
            "direction": action.direction,
            "sl_value": action.sl_value,
            "tp_value": action.tp_value,
        }
        for action in action_map
    ]


def unrealised_pips(position: ConfirmedPosition, current_price: float, symbol: str) -> float:
    if position.is_flat:
        return 0.0
    assert position.entry_price is not None
    price_delta = float(current_price) - float(position.entry_price)
    if position.direction < 0:
        price_delta = -price_delta
    return price_to_pips(symbol, price_delta)


def build_state_vector(
    position: ConfirmedPosition,
    *,
    current_price: float,
    symbol: str,
) -> np.ndarray:
    pnl_sign = 0.0
    if not position.is_flat:
        pnl_sign = float(np.sign(unrealised_pips(position, current_price, symbol)))
    state = np.array(
        [
            float(position.direction),
            min(float(position.time_in_trade_bars) / 1000.0, 1.0),
            pnl_sign,
            np.clip(float(position.last_reward), -1.0, 1.0),
        ],
        dtype=np.float32,
    )
    return state


def build_observation(
    feature_rows: np.ndarray,
    *,
    position: ConfirmedPosition,
    current_price: float,
    symbol: str,
    window_size: int = 1,
) -> np.ndarray:
    rows = np.asarray(feature_rows, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape[0] > window_size:
        rows = rows[-window_size:]
    if rows.shape[0] < window_size:
        pad_row = rows[0] if rows.size else np.zeros(0, dtype=np.float32)
        pad = np.tile(pad_row, (window_size - rows.shape[0], 1))
        rows = np.vstack([pad, rows]) if rows.size else pad
    state = build_state_vector(position, current_price=current_price, symbol=symbol)
    state_block = np.tile(state, (window_size, 1))
    obs = np.hstack([rows, state_block]).astype(np.float32)
    return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def build_action_mask(
    action_map: Sequence[ActionSpec],
    *,
    position: ConfirmedPosition,
    spread_z: float = 0.0,
) -> np.ndarray:
    mask = np.zeros(len(action_map), dtype=bool)
    if not action_map:
        return mask
    mask[0] = True
    if position.is_flat:
        if float(spread_z) < 1.5:
            mask[2:] = True
    else:
        mask[1] = True
    return mask


def flatten_feature_window(rows: Iterable[Sequence[float]]) -> np.ndarray:
    return np.asarray(list(rows), dtype=np.float32)


def compute_timed_sharpe(equity_curve: Sequence[float], timestamps: Sequence[pd.Timestamp]) -> float:
    if len(equity_curve) < 2 or len(timestamps) < 2:
        return 0.0
    curve = np.asarray(equity_curve, dtype=np.float64)
    times = pd.to_datetime(list(timestamps), utc=True)
    log_returns = np.diff(np.log(np.maximum(curve, 1e-6)))
    delta_years = np.diff(times.view("int64")) / 1e9 / (365.25 * 24 * 3600.0)
    valid = delta_years > 0
    if not np.any(valid):
        return 0.0
    normalized = log_returns[valid] / np.sqrt(delta_years[valid])
    if len(normalized) == 0 or np.std(normalized) == 0:
        return 0.0
    return float(np.mean(normalized) / np.std(normalized))


def compute_max_drawdown(equity_curve: Sequence[float]) -> float:
    curve = np.asarray(equity_curve, dtype=np.float64)
    if len(curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / np.maximum(peak, 1e-6)))


def compute_trade_metrics(
    trade_log: Sequence[dict],
    *,
    initial_equity: float = 1_000.0,
) -> dict[str, float]:
    if not trade_log:
        return {
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "win_rate": 0.0,
            "trade_count": 0.0,
        }

    pnl_series: list[float] = []
    prev_equity = float(initial_equity)
    for trade in trade_log:
        if "equity" in trade:
            current_equity = float(trade["equity"])
            pnl_series.append(current_equity - prev_equity)
            prev_equity = current_equity
        else:
            pnl_series.append(float(trade.get("net_pips", 0.0)))

    wins = [pnl for pnl in pnl_series if pnl > 0]
    losses = [pnl for pnl in pnl_series if pnl < 0]
    gross_profit = float(sum(wins))
    gross_loss = float(abs(sum(losses)))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    expectancy = float(np.mean(pnl_series)) if pnl_series else 0.0
    win_rate = len(wins) / len(pnl_series) if pnl_series else 0.0
    return {
        "profit_factor": float(profit_factor),
        "expectancy": expectancy,
        "win_rate": win_rate,
        "trade_count": float(len(pnl_series)),
    }
