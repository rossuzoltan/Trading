"""
trading_env.py  –  Production Forex Trading Environment (Phase 10)
===================================================================
Phase 10 upgrades (Top 0.1% features)
--------------------------------------
* Variable spread: widens 3x during daily rollover (21:00-22:00 UTC)
  and applies realistic noise. Static 1.0 pip is backtest fiction.
* ATR trailing stop: once in profit, stop trails at 1× ATR behind price.
  This prevents giving back large open profits.
* Sortino-shaped reward: penalises downside volatility of returns
  more than upside (asymmetric risk-awareness). Pure log-return
  treats a -10 pip loss identically to a +10 pip gain — incorrect.
* Reward ×10,000 scale preserved from Phase 9.
* pnl_sign observation (no disposition-effect bias) preserved.
* Spread round-trip at open+close preserved from Phase 8.
* initial_equity=1,000 and dynamic lot_size from Phase 8/9 preserved.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from typing import Any

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM = True
except ImportError:
    import gym  # type: ignore
    from gym import spaces  # type: ignore
    _GYM = False

from runtime_common import (
    STATE_FEATURE_COUNT,
    ConfirmedPosition,
    build_action_map,
)
from symbol_utils import pip_size_for_symbol, pip_value_for_volume


# ── Variable spread model ─────────────────────────────────────────────────────

def _dynamic_spread(base_spread: float, hour_utc: int | None) -> float:
    """
    Return the effective spread in pips for a given UTC hour.
    Rollover window 21:00-22:00 UTC: spreads routinely widen to 5-10× base.
    Asian open 23:00-01:00 UTC: slightly wider than London/NY.
    """
    if hour_utc is None:
        return base_spread
    if hour_utc in (21,):                     # daily rollover — widest
        multiplier = 3.5
    elif hour_utc in (22, 0):                 # thin liquidity
        multiplier = 2.0
    elif hour_utc in (7, 8, 13, 14):          # London + NY peak — tightest
        multiplier = 0.8
    else:
        multiplier = 1.0
    noise = np.random.uniform(0.9, 1.1)       # ±10% tick-level noise
    return base_spread * multiplier * noise


class ForexTradingEnv(gym.Env):
    """
    Position-persistent Forex environment with institutional-grade realism.

    Observation: (window_size, n_features + 4)
      [0] position            : -1 / 0 / +1
      [1] time_in_trade       : t / 1000 (clamped to [0,1])
      [2] pnl_sign            : -1 losing / 0 flat / +1 winning
      [3] last_reward         : clipped recent reward memory

    Actions:
      0       : HOLD
      1       : CLOSE  (legal when position != 0)
      2+      : OPEN(direction, sl_pips, tp_pips)  (legal when flat)

    Reward:
      R_t = ln(TotalEquity_t / TotalEquity_{t-1}) × 10,000
      Downside deviation is penalised extra (Sortino-inspired):
        if reward < 0: reward *= DOWNSIDE_PENALTY
      This makes the agent more averse to losses than it is attracted to gains.

    Spread: variable (based on UTC hour), round-trip charged at open+close.
    SL: fixed + optional ATR trailing (once in profit ≥ 1×ATR).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: Any,
        feature_columns: list[str],
        sl_options: list[float],
        tp_options: list[float],
        window_size: int = 1,
        pip_value: float = 0.0001,
        spread_pips: float = 1.0,
        max_slippage_pips: float = 2.0,
        lot_size: float = 0.01,
        swap_pips_per_day: float = 0.1,
        random_start: bool = True,
        min_episode_steps: int = 300,
        episode_max_steps: int | None = None,
        initial_equity: float = 1_000.0,
        atr_col: str = "atr_14",
        use_trailing_stop: bool = True,
        use_variable_spread: bool = True,
        atr_scaled: bool = False,   # Phase 12: use ATR multipliers for SL/TP
        vol_scaling: bool = False,  # Phase 13: dynamic lot sizing (Risk Parity)
        target_risk_pct: float = 0.01,
        symbol: str = "EURUSD",
        account_currency: str = "USD",
        reward_window: int = 128,
        reward_vol_floor: float = 1e-4,
        transaction_cost_penalty: float = 0.25,
        drawdown_penalty: float = 2.0,
        position_change_penalty: float = 0.5,
        reward_tanh_clip: bool = True,
    ) -> None:
        super().__init__()

        source_frame = df.copy()
        timestamp_index: pd.DatetimeIndex | None = None
        if isinstance(source_frame.index, pd.DatetimeIndex):
            timestamp_index = pd.DatetimeIndex(pd.to_datetime(source_frame.index, utc=True, errors="coerce"))
        elif "Gmt time" in source_frame.columns:
            timestamp_index = pd.DatetimeIndex(pd.to_datetime(source_frame["Gmt time"], utc=True, errors="coerce"))

        self.df = source_frame.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.feature_columns = list(feature_columns)
        self.sl_options = list(sl_options)
        self.tp_options = list(tp_options)
        self.window_size = int(window_size)

        self.symbol = symbol.upper()
        inferred_pip_size = pip_size_for_symbol(self.symbol)
        self.pip_value = inferred_pip_size if float(pip_value) == 0.0001 else float(pip_value)
        self.base_spread_pips = float(spread_pips)
        self.max_slippage_pips = float(max_slippage_pips)
        self.lot_size = float(lot_size)
        self.account_currency = account_currency.upper()
        initial_price = float(self.df.iloc[0]["Close"]) if len(self.df) else 1.0
        self.usd_per_pip = self._pip_value_for_volume(initial_price)
        self.swap_pips_per_day = float(swap_pips_per_day)
        self._initial_equity = float(initial_equity)

        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps

        self.atr_col = atr_col
        self.use_trailing_stop = bool(use_trailing_stop)
        self.use_variable_spread = bool(use_variable_spread)
        self.atr_scaled = bool(atr_scaled)
        self.vol_scaling = bool(vol_scaling)
        self.target_risk_pct = float(target_risk_pct)
        self.reward_window = max(int(reward_window), 2)
        self.reward_vol_floor = max(float(reward_vol_floor), 1e-8)
        self.transaction_cost_penalty = float(transaction_cost_penalty)
        self.drawdown_penalty = float(drawdown_penalty)
        self.position_change_penalty = float(position_change_penalty)
        self.reward_tanh_clip = bool(reward_tanh_clip)

        # Build action map
        # When atr_scaled=True: sl_pips/tp_pips are ATR multipliers (0.5, 1.0, 2.0, 3.0)
        # When atr_scaled=False (legacy): sl_pips/tp_pips are literal pip counts
        self._action_map = build_action_map(self.sl_options, self.tp_options)

        self.action_space = spaces.Discrete(len(self._action_map))

        n_feat = len(self.feature_columns) + STATE_FEATURE_COUNT
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, n_feat),
            dtype=np.float32,
        )

        self._close_array = self.df["Close"].to_numpy(dtype=np.float64, copy=True)
        self._high_array = self.df["High"].to_numpy(dtype=np.float64, copy=True)
        self._low_array = self.df["Low"].to_numpy(dtype=np.float64, copy=True)
        if self.feature_columns:
            self._feature_array = self.df.loc[:, self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        else:
            self._feature_array = np.empty((self.n_steps, 0), dtype=np.float32)
        if self.atr_col in self.df.columns:
            self._atr_array = self.df[self.atr_col].to_numpy(dtype=np.float64, copy=True)
        else:
            self._atr_array = np.full(self.n_steps, 20 * self.pip_value, dtype=np.float64)
        if "spread_z" in self.df.columns:
            self._spread_z_array = self.df["spread_z"].to_numpy(dtype=np.float32, copy=True)
        else:
            self._spread_z_array = np.zeros(self.n_steps, dtype=np.float32)
        self._pip_value_per_lot_array = np.asarray(
            [
                pip_value_for_volume(
                    self.symbol,
                    price=float(price),
                    volume_lots=1.0,
                    account_currency=self.account_currency,
                )
                for price in self._close_array
            ],
            dtype=np.float64,
        )
        if timestamp_index is not None and len(timestamp_index) == self.n_steps:
            self._hour_array = np.asarray(timestamp_index.hour, dtype=np.int16)
            self._timestamp_datetimes = list(timestamp_index.to_pydatetime())
        else:
            self._hour_array = None
            self._timestamp_datetimes = None

        self._reset_internal()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _reset_internal(self) -> None:
        self.current_step: int = 0
        self.steps_in_episode: int = 0
        self.terminated: bool = False
        self.truncated: bool = False

        self.position: int = 0
        self.entry_price: float | None = None
        self.sl_price: float | None = None
        self.tp_price: float | None = None
        self.trailing_sl: float | None = None   # ATR trailing stop
        self.time_in_trade: int = 0

        self.equity_usd: float = self._initial_equity
        self.prev_total_equity: float = self._initial_equity
        self.high_water_mark_equity: float = self._initial_equity
        self.last_reward: float = 0.0
        self._log_return_buffer: list[float] = []
        self._spread_cache_step: int | None = None
        self._cached_spread_pips: float = self.base_spread_pips

        self.equity_curve: list[float] = []
        self.trade_log: list[dict] = []

    @property
    def initial_equity(self) -> float:
        return self._initial_equity

    def _current_index(self) -> int:
        if self.n_steps <= 0:
            return 0
        return min(max(int(self.current_step), 0), self.n_steps - 1)

    def _current_close(self) -> float:
        return float(self._close_array[self._current_index()])

    def _current_high(self) -> float:
        return float(self._high_array[self._current_index()])

    def _current_low(self) -> float:
        return float(self._low_array[self._current_index()])

    def _confirmed_position(self) -> ConfirmedPosition:
        return ConfirmedPosition(
            direction=int(self.position),
            entry_price=self.entry_price,
            sl_price=self.sl_price,
            tp_price=self.tp_price,
            volume=float(self.lot_size if self.position != 0 else 0.0),
            time_in_trade_bars=int(self.time_in_trade),
            last_reward=float(self.last_reward),
        )

    def _pip_value_for_volume(self, price: float) -> float:
        return pip_value_for_volume(
            self.symbol,
            price=price,
            volume_lots=self.lot_size,
            account_currency=self.account_currency,
        )

    def _current_hour(self) -> int | None:
        if self._hour_array is None:
            return None
        return int(self._hour_array[self._current_index()])

    def _current_atr(self) -> float:
        return float(self._atr_array[self._current_index()])

    def _current_spread_z(self) -> float:
        return float(self._spread_z_array[self._current_index()])

    def _current_pip_value_per_lot(self) -> float:
        return float(self._pip_value_per_lot_array[self._current_index()])

    def _current_pip_value(self) -> float:
        return float(self._pip_value_per_lot_array[self._current_index()] * self.lot_size)

    def _effective_spread(self) -> float:
        """Return spread in pips for this bar (variable by time of day)."""
        if self._spread_cache_step != self.current_step:
            if self.use_variable_spread:
                self._cached_spread_pips = _dynamic_spread(self.base_spread_pips, self._current_hour())
            else:
                self._cached_spread_pips = self.base_spread_pips
            self._spread_cache_step = self.current_step
        return self._cached_spread_pips

    def _unrealised_pips(self, price: float | None = None) -> float:
        if self.position == 0 or self.entry_price is None:
            return 0.0
        price = self._current_close() if price is None else float(price)
        diff = (price - self.entry_price) if self.position == 1 \
               else (self.entry_price - price)
        return diff / self.pip_value

    def _slippage(self) -> float:
        if self.max_slippage_pips <= 0:
            return 0.0
        return float(np.random.uniform(0.5, self.max_slippage_pips))

    def _current_exposure_lots(self) -> float:
        if self.position == 0:
            return 0.0
        return float(self.position * self.lot_size)

    def _rolling_reward_std(self, log_return: float) -> float:
        self._log_return_buffer.append(float(log_return))
        window = self._log_return_buffer[-self.reward_window:]
        if len(window) < 2:
            return max(abs(float(log_return)), self.reward_vol_floor)
        return max(float(np.std(window)), self.reward_vol_floor)

    def _estimate_transaction_cost_ratio(
        self,
        turnover_lots: float,
        equity_base: float,
        *,
        pip_value_per_lot: float | None = None,
    ) -> float:
        if turnover_lots <= 0:
            return 0.0
        if pip_value_per_lot is None:
            pip_value_per_lot = self._current_pip_value_per_lot()
        per_side_cost_pips = (self._effective_spread() / 2.0) + max(self.max_slippage_pips, 0.0)
        estimated_cost_usd = turnover_lots * pip_value_per_lot * per_side_cost_pips
        return float(estimated_cost_usd / max(equity_base, 1e-6))

    def _open(
        self,
        direction: int,
        sl_param: float,
        tp_param: float,
        *,
        price: float | None = None,
        pip_value_per_lot: float | None = None,
    ) -> None:
        spread    = self._effective_spread()
        price     = self._current_close() if price is None else float(price)
        slip      = self._slippage() * self.pip_value
        half_spread = (spread / 2.0) * self.pip_value

        # Convert SL/TP to price distances
        if self.atr_scaled:
            # sl_param/tp_param are ATR multipliers (e.g. 1.5 means 1.5 * ATR)
            atr = self._current_atr()
            sl_dist = sl_param * atr
            tp_dist = tp_param * atr
        else:
            # Legacy: literal pip values
            sl_dist = sl_param * self.pip_value
            tp_dist = tp_param * self.pip_value

        if direction == 1:
            entry = price + slip + half_spread
            self.sl_price = entry - sl_dist
            self.tp_price = entry + tp_dist
        else:
            entry = price - slip - half_spread
            self.sl_price = entry + sl_dist
            self.tp_price = entry - tp_dist

        # ── Dynamic Position Sizing (Phase 13: Risk Parity) ──────────────────
        if self.vol_scaling:
            # Risk X% of equity (e.g. $10) per trade.
            risk_usd = self.equity_usd * self.target_risk_pct
            sl_pips  = sl_dist / self.pip_value
            if sl_pips > 0:
                pip_val_base = (
                    self._pip_value_for_volume(price)
                    if pip_value_per_lot is None
                    else float(pip_value_per_lot)
                )
                calc_lots = risk_usd / (sl_pips * pip_val_base)
                # Cap between 0.01 and 2.0 lots for safety
                self.lot_size = float(np.clip(round(calc_lots, 2), 0.01, 2.0))
        
        self.usd_per_pip = self._pip_value_for_volume(entry)
        self.position    = direction
        self.entry_price = entry
        self.entry_step  = self.current_step
        self.trailing_sl = None
        self.time_in_trade = 0

    def _update_trailing_stop(self, *, price: float | None = None, atr: float | None = None) -> None:
        """Advance ATR trailing stop once trade is profitable by ≥ 1×ATR."""
        if not self.use_trailing_stop or self.position == 0:
            return
        atr = self._current_atr() if atr is None else float(atr)
        price = self._current_close() if price is None else float(price)
        upnl  = self._unrealised_pips(price=price)

        # Only activate trailing once in profit by ≥ 1 ATR
        if upnl * self.pip_value < atr:
            return

        if self.position == 1:
            new_trail = price - atr
            if self.trailing_sl is None or new_trail > self.trailing_sl:
                self.trailing_sl = new_trail
                # Trailing stop overrides fixed SL if it is tighter
                if self.trailing_sl > self.sl_price:
                    self.sl_price = self.trailing_sl
        else:
            new_trail = price + atr
            if self.trailing_sl is None or new_trail < self.trailing_sl:
                self.trailing_sl = new_trail
                if self.trailing_sl < self.sl_price:
                    self.sl_price = self.trailing_sl

    def _close(self, reason: str, exit_price: float) -> float:
        spread = self._effective_spread()
        half_spread = spread / 2.0
        if self.position == 1:
            raw_pips = (exit_price - self.entry_price) / self.pip_value
        else:
            raw_pips = (self.entry_price - exit_price) / self.pip_value
        net_pips = raw_pips - half_spread
        self.equity_usd += net_pips * self._pip_value_for_volume(exit_price)
        self.trade_log.append({
            "step": self.current_step, "reason": reason,
            "position": self.position, "entry": self.entry_price,
            "exit": exit_price, "net_pips": net_pips,
            "equity": self.equity_usd,
        })
        self.position     = 0
        self.entry_price  = None
        self.sl_price     = None
        self.tp_price     = None
        self.trailing_sl  = None
        self.time_in_trade = 0
        return net_pips

    def _check_sl_tp(self, *, high: float | None = None, low: float | None = None) -> None:
        if self.position == 0:
            return
        if self.current_step >= self.n_steps:
            return
        hi = self._current_high() if high is None else float(high)
        lo = self._current_low() if low is None else float(low)
        if self.position == 1:
            if lo <= self.sl_price:
                self._close("SL", self.sl_price)
            elif hi >= self.tp_price:
                self._close("TP", self.tp_price)
        else:
            if hi >= self.sl_price:
                self._close("SL", self.sl_price)
            elif lo <= self.tp_price:
                self._close("TP", self.tp_price)

    def _get_observation(self) -> np.ndarray:
        step = self._current_index()
        end = min(step + 1, self.n_steps)
        start = max(0, end - self.window_size)
        feature_window = self._feature_array[start:end]
        if feature_window.shape[0] < self.window_size:
            feature_count = self._feature_array.shape[1]
            if feature_window.size:
                pad_rows = self.window_size - feature_window.shape[0]
                pad = np.repeat(feature_window[:1], pad_rows, axis=0)
                feature_window = np.concatenate([pad, feature_window], axis=0)
            else:
                feature_window = np.zeros((self.window_size, feature_count), dtype=np.float32)

        current_price = float(self._close_array[step])
        if self.position == 0 or self.entry_price is None:
            pnl_sign = 0.0
        else:
            pnl_sign = float(np.sign(self._unrealised_pips(price=current_price) * self.position))

        state = np.array(
            [
                float(self.position),
                min(float(self.time_in_trade) / 1000.0, 1.0),
                pnl_sign,
                float(np.clip(self.last_reward, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        obs = np.empty((self.window_size, feature_window.shape[1] + STATE_FEATURE_COUNT), dtype=np.float32)
        obs[:, :feature_window.shape[1]] = feature_window
        obs[:, feature_window.shape[1]:] = state
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_obs(self) -> np.ndarray:
        return self._get_observation()

    # ── MaskablePPO interface ────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(len(self._action_map), dtype=bool)
        if not len(mask):
            return mask
        mask[0] = True
        if self.position == 0:
            if self._current_spread_z() < 1.5:
                mask[2:] = True
        else:
            mask[1] = True
        return mask

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_internal()
        if self.random_start:
            max_start = self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            hi = max(self.window_size, max_start)
            self.current_step = int(np.random.randint(
                self.window_size, max(hi, self.window_size + 1)
            ))
        else:
            self.current_step = self.window_size
        obs = self._get_observation()
        return (obs, {}) if _GYM else obs

    def step(self, action: int):
        action = int(action)
        if not (0 <= action < len(self._action_map)):
            raise ValueError(f"Invalid action {action}")
        step_idx = self.current_step
        current_close = float(self._close_array[step_idx])
        current_high = float(self._high_array[step_idx])
        current_low = float(self._low_array[step_idx])
        current_pip_value_per_lot = float(self._pip_value_per_lot_array[step_idx])
        prev_exposure_lots = abs(self._current_exposure_lots())

        # Existing positions are managed on the current completed bar.
        # This keeps stop/TP handling causal for positions opened on prior bars.
        self._check_sl_tp(high=current_high, low=current_low)
        if self.position != 0:
            self._update_trailing_stop(price=current_close, atr=float(self._atr_array[step_idx]))

        action_spec = self._action_map[action]
        act_type = action_spec.action_type.value
        direction = action_spec.direction
        sl_pips = action_spec.sl_value
        tp_pips = action_spec.tp_value

        # Safety override
        if act_type == "OPEN"  and self.position != 0: act_type = "HOLD"
        if act_type == "CLOSE" and self.position == 0: act_type = "HOLD"
        if act_type == "OPEN" and self.current_step >= self.n_steps - 1:
            act_type = "HOLD"

        if act_type == "OPEN":
            self._open(
                direction,
                sl_pips,
                tp_pips,
                price=current_close,
                pip_value_per_lot=current_pip_value_per_lot,
            )
        elif act_type == "CLOSE":
            slip   = self._slippage() * self.pip_value
            exit_p = current_close - slip if self.position == 1 else current_close + slip
            self._close("MANUAL", exit_p)

        if step_idx >= self.n_steps - 1 and self.position != 0:
            self._close("END_OF_DATA", current_close)

        if self.position != 0:
            self.time_in_trade += 1
            current_pip_value = self._current_pip_value()
            swap_usd = (self.swap_pips_per_day / 24.0) * current_pip_value
            self.equity_usd -= swap_usd
        else:
            current_pip_value = 0.0

        unrealised_usd = self._unrealised_pips(price=current_close) * current_pip_value
        total_equity   = self.equity_usd + unrealised_usd

        if total_equity <= 0:
            log_return = -1.0
            reward_std = self.reward_vol_floor
            drawdown = 1.0
            turnover_lots = prev_exposure_lots
            transaction_penalty = 1.0
            position_penalty = self.position_change_penalty * (turnover_lots ** 2)
            reward = -1.0
            self.terminated = True
        else:
            prev = max(self.prev_total_equity, 1e-6)
            log_return = float(np.log(total_equity / prev))
            reward_std = self._rolling_reward_std(log_return)
            normalized_return = log_return / (reward_std + 1e-8)

            self.high_water_mark_equity = max(self.high_water_mark_equity, total_equity)
            drawdown = (self.high_water_mark_equity - total_equity) / max(self.high_water_mark_equity, 1e-6)

            turnover_lots = abs(self._current_exposure_lots() - prev_exposure_lots)
            transaction_penalty = (
                self.transaction_cost_penalty
                * self._estimate_transaction_cost_ratio(
                    turnover_lots,
                    prev,
                    pip_value_per_lot=current_pip_value_per_lot,
                )
                / (reward_std + 1e-8)
            )
            position_penalty = self.position_change_penalty * (turnover_lots ** 2)

            reward = normalized_return
            reward -= transaction_penalty
            reward -= self.drawdown_penalty * drawdown
            reward -= position_penalty
            if self.reward_tanh_clip:
                reward = float(np.tanh(reward))

            # ── Phase 13: Regime-Aware Reward Intelligence ───────────────────
            # Hurst > 0.6: Trending. Reward holding profitable trends.
            # Hurst < 0.4: Mean-reverting. Penalise holding long trades.
        self.last_reward = reward
        self.prev_total_equity = total_equity if total_equity > 0 else 1e-6

        self.current_step     += 1
        self.steps_in_episode += 1
        self.equity_curve.append(float(total_equity))

        if self.current_step >= self.n_steps - 1:
            self.terminated = True
        if self.episode_max_steps is not None and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        obs  = self._get_observation()
        info: dict[str, Any] = {
            "equity_usd":    float(self.equity_usd),
            "total_equity_usd": float(total_equity),
            "position":      int(self.position),
            "time_in_trade": int(self.time_in_trade),
            "timestamp_utc": (
                self._timestamp_datetimes[self.current_step - 1].isoformat()
                if self._timestamp_datetimes is not None and self.current_step > 0
                else None
            ),
            "log_return": float(log_return),
            "reward_std": float(reward_std),
            "drawdown": float(drawdown),
            "turnover_lots": float(turnover_lots),
            "transaction_cost_penalty": float(transaction_penalty),
            "position_change_penalty": float(position_penalty),
        }
        if _GYM:
            return obs, reward, self.terminated, self.truncated, info
        return obs, reward, bool(self.terminated or self.truncated), info

    def render(self) -> None:
        print(f"Step={self.current_step} | Equity=${self.equity_usd:,.2f} | "
              f"Pos={self.position} | Entry={self.entry_price}")

    # ── State persistence ────────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "equity_usd":    self.equity_usd,
            "position":      self.position,
            "entry_price":   self.entry_price,
            "sl_price":      self.sl_price,
            "tp_price":      self.tp_price,
            "trailing_sl":   self.trailing_sl,
            "time_in_trade": self.time_in_trade,
            "current_step":  self.current_step,
        }

    def set_state(self, state: dict) -> None:
        self.equity_usd      = float(state["equity_usd"])
        self.position        = int(state["position"])
        self.entry_price     = state["entry_price"]
        self.sl_price        = state["sl_price"]
        self.tp_price        = state["tp_price"]
        self.trailing_sl     = state.get("trailing_sl")
        self.time_in_trade   = int(state["time_in_trade"])
        self.current_step    = int(state["current_step"])
        self.prev_total_equity = self.equity_usd

    def save_state(self, path: str = "live_state.json") -> None:
        with open(path, "w") as f:
            json.dump(self.get_state(), f, indent=2)

    def load_state(self, path: str = "live_state.json") -> None:
        with open(path) as f:
            self.set_state(json.load(f))
