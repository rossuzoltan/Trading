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
    if not sl_options:
        raise ValueError("sl_options must contain at least one value.")
    if not tp_options:
        raise ValueError("tp_options must contain at least one value.")
    return build_simple_action_map(sl_value=float(sl_options[0]), tp_value=float(tp_options[0]))


def build_simple_action_map(
    *,
    sl_value: float = 1.0,
    tp_value: float = 1.0,
) -> tuple[ActionSpec, ...]:
    return (
        ActionSpec(ActionType.HOLD),
        ActionSpec(ActionType.CLOSE),
        ActionSpec(ActionType.OPEN, direction=1, sl_value=float(sl_value), tp_value=float(tp_value)),
        ActionSpec(ActionType.OPEN, direction=-1, sl_value=float(sl_value), tp_value=float(tp_value)),
    )


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


def deserialize_action_map(
    raw_action_map: Sequence[dict[str, float | int | str | None]],
) -> tuple[ActionSpec, ...]:
    actions: list[ActionSpec] = []
    for raw in raw_action_map:
        action_type = ActionType(str(raw.get("action_type", ActionType.HOLD.value)))
        direction_raw = raw.get("direction")
        sl_raw = raw.get("sl_value")
        tp_raw = raw.get("tp_value")
        actions.append(
            ActionSpec(
                action_type=action_type,
                direction=int(direction_raw) if direction_raw is not None else None,
                sl_value=float(sl_raw) if sl_raw is not None else None,
                tp_value=float(tp_raw) if tp_raw is not None else None,
            )
        )
    return tuple(actions)


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
        for idx, action in enumerate(action_map):
            if (
                action.action_type == ActionType.OPEN
                and action.direction is not None
                and int(action.direction) == int(position.direction)
            ):
                mask[idx] = True
    return mask


def action_label(action: ActionSpec) -> str:
    if action.action_type == ActionType.HOLD:
        return "FLAT"
    if action.action_type == ActionType.CLOSE:
        return "CLOSE"
    if int(action.direction or 0) > 0:
        return "LONG"
    if int(action.direction or 0) < 0:
        return "SHORT"
    return action.action_type.value


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
            "expectancy_usd": 0.0,
            "expectancy_pips": 0.0,
            "win_rate": 0.0,
            "trade_count": 0.0,
            "gross_pnl_usd": 0.0,
            "net_pnl_usd": 0.0,
            "gross_profit_usd": 0.0,
            "gross_loss_usd": 0.0,
            "total_transaction_cost_usd": 0.0,
            "total_commission_usd": 0.0,
            "total_spread_slippage_cost_usd": 0.0,
            "total_spread_cost_usd": 0.0,
            "total_slippage_cost_usd": 0.0,
            "forced_close_count": 0.0,
            "avg_holding_bars": 0.0,
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0,
            "win_loss_asymmetry": 0.0,
        }

    pnl_series: list[float] = []
    net_pips_series: list[float] = []
    gross_pnl_usd = 0.0
    total_transaction_cost_usd = 0.0
    total_commission_usd = 0.0
    total_spread_slippage_cost_usd = 0.0
    total_spread_cost_usd = 0.0
    total_slippage_cost_usd = 0.0
    forced_close_count = 0
    holding_bars: list[float] = []
    prev_equity = float(initial_equity)
    for trade in trade_log:
        gross_pnl_usd += float(trade.get("gross_pnl_usd", 0.0))
        total_transaction_cost_usd += float(trade.get("transaction_cost_usd", 0.0))
        total_commission_usd += float(trade.get("commission_usd", 0.0))
        total_spread_slippage_cost_usd += float(trade.get("spread_slippage_cost_usd", 0.0))
        total_spread_cost_usd += float(trade.get("spread_cost_usd", 0.0))
        total_slippage_cost_usd += float(trade.get("slippage_cost_usd", 0.0))
        forced_close_count += int(bool(trade.get("forced_close", False)))
        if "holding_bars" in trade:
            holding_bars.append(float(trade.get("holding_bars", 0.0)))
        net_pips_series.append(float(trade.get("net_pips", 0.0)))
        if "equity" in trade:
            current_equity = float(trade["equity"])
            pnl_series.append(current_equity - prev_equity)
            prev_equity = current_equity
        elif "net_pnl_usd" in trade:
            pnl_series.append(float(trade["net_pnl_usd"]))
        else:
            pnl_series.append(float(trade.get("net_pips", 0.0)))

    wins = [pnl for pnl in pnl_series if pnl > 0]
    losses = [pnl for pnl in pnl_series if pnl < 0]
    gross_profit = float(sum(wins))
    gross_loss = float(abs(sum(losses)))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    expectancy_usd = float(np.mean(pnl_series)) if pnl_series else 0.0
    expectancy_pips = float(np.mean(net_pips_series)) if net_pips_series else 0.0
    win_rate = len(wins) / len(pnl_series) if pnl_series else 0.0
    avg_win_usd = float(np.mean(wins)) if wins else 0.0
    avg_loss_usd = float(np.mean(losses)) if losses else 0.0
    win_loss_asymmetry = avg_win_usd / abs(avg_loss_usd) if avg_loss_usd < 0 else (float("inf") if avg_win_usd > 0 else 0.0)
    net_pnl_usd = float(sum(pnl_series))
    avg_holding_bars = float(np.mean(holding_bars)) if holding_bars else 0.0
    return {
        "profit_factor": float(profit_factor),
        "expectancy": expectancy_usd,
        "expectancy_usd": expectancy_usd,
        "expectancy_pips": expectancy_pips,
        "win_rate": float(win_rate),
        "trade_count": float(len(pnl_series)),
        "gross_pnl_usd": float(gross_pnl_usd),
        "net_pnl_usd": net_pnl_usd,
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "total_transaction_cost_usd": float(total_transaction_cost_usd),
        "total_commission_usd": float(total_commission_usd),
        "total_spread_slippage_cost_usd": float(total_spread_slippage_cost_usd),
        "total_spread_cost_usd": float(total_spread_cost_usd),
        "total_slippage_cost_usd": float(total_slippage_cost_usd),
        "forced_close_count": float(forced_close_count),
        "avg_holding_bars": avg_holding_bars,
        "avg_win_usd": avg_win_usd,
        "avg_loss_usd": avg_loss_usd,
        "win_loss_asymmetry": float(win_loss_asymmetry),
    }
