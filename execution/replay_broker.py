from __future__ import annotations

from domain.enums import ActionType
from domain.models import (
    BrokerPositionSnapshot,
    OrderIntent,
    SubmitResult,
    VolumeBar,
)
from symbol_utils import (
    pip_size_for_symbol,
    pip_value_for_volume,
    price_to_pips,
)

from .broker import BaseBroker


class ReplayBroker(BaseBroker):
    def __init__(
        self,
        *,
        symbol: str,
        initial_equity: float = 1_000.0,
        account_currency: str = "USD",
        commission_per_lot: float = 7.0,
        slippage_pips: float = 0.25,
        partial_fill_ratio: float = 1.0,
    ) -> None:
        self.symbol = symbol
        self.account_currency = account_currency
        self.equity = float(initial_equity)
        self.commission_per_lot = float(commission_per_lot)
        self.slippage_pips = float(slippage_pips)
        self.partial_fill_ratio = float(partial_fill_ratio)
        self.next_order_id = 1
        self.next_ticket = 1000
        self._pending: list[OrderIntent] = []
        self._position = BrokerPositionSnapshot(symbol=symbol)
        self._bar_index = -1
        self.trade_log: list[dict[str, float | int | str]] = []
        self.execution_log: list[dict[str, float | int | str | bool | None]] = []
        self._pip_value_cache: dict[str, float] = {}

    def submit_order(self, intent: OrderIntent) -> SubmitResult:
        self._pending.append(intent)
        order_id = self.next_order_id
        self.next_order_id += 1
        return SubmitResult(accepted=True, order_id=order_id)

    def _apply_commission(self, volume: float) -> None:
        self.equity -= self.commission_per_lot * float(volume)

    def _commission_usd(self, volume: float) -> float:
        return float(self.commission_per_lot * float(volume))

    def _slippage_price(self, direction: int) -> float:
        pip_size = pip_size_for_symbol(self.symbol)
        signed = self.slippage_pips * pip_size
        return signed if direction > 0 else -signed

    def _execution_price(
        self,
        reference_price: float,
        direction: int,
        *,
        avg_spread: float = 0.0,
        is_entry: bool,
    ) -> float:
        price = float(reference_price)
        spread_half = max(float(avg_spread), 0.0) / 2.0
        if is_entry:
            price += self._slippage_price(direction)
            if direction > 0:
                price += spread_half
            else:
                price -= spread_half
            return price
        price -= self._slippage_price(direction)
        if direction > 0:
            price -= spread_half
        else:
            price += spread_half
        return price

    def _cached_pip_value(self, price: float, volume: float) -> float:
        cache_key = f"{round(float(price), 4)}_{round(float(volume), 4)}"
        if cache_key in self._pip_value_cache:
            return self._pip_value_cache[cache_key]
        
        val = pip_value_for_volume(
            self.symbol,
            price=float(price),
            volume_lots=float(volume),
            account_currency=self.account_currency,
        )
        if len(self._pip_value_cache) > 1000:
            self._pip_value_cache.clear()
        self._pip_value_cache[cache_key] = val
        return val

    def _spread_slippage_cost_usd(
        self,
        *,
        reference_price: float,
        fill_price: float,
        volume: float,
    ) -> float:
        move_pips = abs(price_to_pips(self.symbol, float(fill_price) - float(reference_price)))
        pip_value = self._cached_pip_value(float(fill_price), float(volume))
        return float(move_pips * pip_value)

    def _spread_cost_usd(self, *, price: float, volume: float, avg_spread: float) -> float:
        spread_half_pips = abs(price_to_pips(self.symbol, max(float(avg_spread), 0.0))) / 2.0
        pip_value = self._cached_pip_value(float(price), float(volume))
        return float(spread_half_pips * pip_value)

    def _slippage_cost_usd(self, *, price: float, volume: float) -> float:
        pip_value = self._cached_pip_value(float(price), float(volume))
        return float(max(float(self.slippage_pips), 0.0) * pip_value)

    def _resolve_fill_volume(self, intent: OrderIntent, *, fill_price: float) -> float:
        planned_volume = float(intent.volume)
        if (
            intent.risk_fraction is not None
            and intent.sl_distance_price is not None
            and intent.lot_size_min is not None
            and intent.lot_size_max is not None
        ):
            sl_pips = max(abs(price_to_pips(self.symbol, float(intent.sl_distance_price))), 1e-6)
            pip_value_per_lot = self._cached_pip_value(float(fill_price), 1.0)
            raw_lots = (self.equity * float(intent.risk_fraction)) / max(sl_pips * pip_value_per_lot, 1e-6)
            planned_volume = max(float(intent.lot_size_min), min(float(intent.lot_size_max), raw_lots))
        return round(planned_volume * self.partial_fill_ratio, 2)

    def _resolve_protective_prices(
        self,
        intent: OrderIntent,
        *,
        fill_price: float,
    ) -> tuple[float | None, float | None]:
        sl_price = intent.sl_price
        tp_price = intent.tp_price
        if intent.action.direction is None:
            return sl_price, tp_price
        if intent.sl_distance_price is not None:
            sl_distance = abs(float(intent.sl_distance_price))
            sl_price = float(fill_price) - sl_distance if intent.action.direction > 0 else float(fill_price) + sl_distance
        if intent.tp_distance_price is not None:
            tp_distance = abs(float(intent.tp_distance_price))
            tp_price = float(fill_price) + tp_distance if intent.action.direction > 0 else float(fill_price) - tp_distance
        return sl_price, tp_price

    def _close_position(
        self,
        exit_price: float,
        reason: str,
        time_msc: int,
        *,
        reference_price: float | None = None,
        forced: bool = False,
    ) -> float:
        if self._position.direction == 0 or self._position.entry_price is None:
            return 0.0
        direction = int(self._position.direction)
        pip_pnl = price_to_pips(self.symbol, exit_price - self._position.entry_price)
        if direction < 0:
            pip_pnl = -pip_pnl
        pip_value = self._cached_pip_value(exit_price, self._position.volume)
        pnl = float(pip_pnl * pip_value)
        entry_reference_price = (
            float(self._position.entry_reference_price)
            if self._position.entry_reference_price is not None
            else float(self._position.entry_price)
        )
        exit_reference_price = float(reference_price) if reference_price is not None else float(exit_price)
        gross_pips = price_to_pips(self.symbol, exit_reference_price - entry_reference_price)
        if direction < 0:
            gross_pips = -gross_pips
        gross_pnl_usd = float(gross_pips * pip_value)
        exit_spread_slippage_cost_usd = self._spread_slippage_cost_usd(
            reference_price=exit_reference_price,
            fill_price=float(exit_price),
            volume=float(self._position.volume),
        )
        exit_spread_cost_usd = self._spread_cost_usd(
            price=float(exit_price),
            volume=float(self._position.volume),
            avg_spread=abs(2.0 * (float(exit_reference_price) - float(exit_price) - self._slippage_price(direction))),
        )
        exit_slippage_cost_usd = max(float(exit_spread_slippage_cost_usd - exit_spread_cost_usd), 0.0)
        exit_commission_usd = self._commission_usd(self._position.volume)
        total_transaction_cost_usd = (
            float(self._position.entry_spread_slippage_cost_usd)
            + float(self._position.entry_commission_usd)
            + exit_spread_slippage_cost_usd
            + exit_commission_usd
        )
        holding_bars = max(int(self._bar_index - int(self._position.entry_bar_index or self._bar_index)), 1)
        net_pnl_usd = float(gross_pnl_usd - total_transaction_cost_usd)
        self.equity += pnl
        self._apply_commission(self._position.volume)
        self.execution_log.append(
            {
                "event": "order_executed",
                "side": "close",
                "reason": reason,
                "direction": direction,
                "volume": float(self._position.volume),
                "reference_price": exit_reference_price,
                "fill_price": float(exit_price),
                "forced": bool(forced),
                "time_msc": int(time_msc),
                "ticket": int(self._position.broker_ticket or 0),
            }
        )
        self.trade_log.append(
            {
                "reason": reason,
                "ticket": int(self._position.broker_ticket or 0),
                "direction": direction,
                "volume": float(self._position.volume),
                "entry_price": float(self._position.entry_price),
                "exit_price": float(exit_price),
                "entry_reference_price": entry_reference_price,
                "exit_reference_price": exit_reference_price,
                "gross_pips": float(gross_pips),
                "net_pips": float(pip_pnl),
                "gross_pnl_usd": gross_pnl_usd,
                "net_pnl_usd": net_pnl_usd,
                "transaction_cost_usd": float(total_transaction_cost_usd),
                "commission_usd": float(self._position.entry_commission_usd + exit_commission_usd),
                "spread_slippage_cost_usd": float(
                    self._position.entry_spread_slippage_cost_usd + exit_spread_slippage_cost_usd
                ),
                "spread_cost_usd": float(self._position.entry_spread_cost_usd + exit_spread_cost_usd),
                "slippage_cost_usd": float(self._position.entry_slippage_cost_usd + exit_slippage_cost_usd),
                "holding_bars": int(holding_bars),
                "forced_close": bool(forced),
                "equity": float(self.equity),
            }
        )
        closed_volume = float(self._position.volume)
        self._position = BrokerPositionSnapshot(symbol=self.symbol, last_confirmed_time_msc=time_msc)
        return closed_volume

    def _fill_pending(self, bar: VolumeBar) -> float:
        if not self._pending:
            return 0.0
        intents = self._pending
        self._pending = []
        turnover_lots = 0.0
        for intent in intents:
            if intent.action.action_type == ActionType.CLOSE:
                if self._position.direction != 0:
                    close_price = self._execution_price(
                        bar.open,
                        self._position.direction,
                        avg_spread=bar.avg_spread,
                        is_entry=False,
                    )
                    turnover_lots += self._close_position(
                        close_price,
                        "MANUAL",
                        bar.start_time_msc,
                        reference_price=float(bar.open),
                    )
                continue
            if intent.action.action_type != ActionType.OPEN or intent.action.direction is None:
                continue
            if self._position.direction != 0:
                continue
            open_price = self._execution_price(
                bar.open,
                intent.action.direction,
                avg_spread=bar.avg_spread,
                is_entry=True,
            )
            fill_volume = self._resolve_fill_volume(intent, fill_price=open_price)
            if fill_volume <= 0:
                continue
            sl_price, tp_price = self._resolve_protective_prices(intent, fill_price=open_price)
            entry_spread_slippage_cost_usd = self._spread_slippage_cost_usd(
                reference_price=float(bar.open),
                fill_price=float(open_price),
                volume=float(fill_volume),
            )
            entry_spread_cost_usd = self._spread_cost_usd(
                price=float(open_price),
                volume=float(fill_volume),
                avg_spread=float(bar.avg_spread),
            )
            entry_slippage_cost_usd = max(float(entry_spread_slippage_cost_usd - entry_spread_cost_usd), 0.0)
            entry_commission_usd = self._commission_usd(fill_volume)
            self._apply_commission(fill_volume)
            self._position = BrokerPositionSnapshot(
                symbol=self.symbol,
                direction=int(intent.action.direction),
                volume=fill_volume,
                entry_price=float(open_price),
                entry_reference_price=float(bar.open),
                entry_bar_index=int(self._bar_index),
                sl_price=sl_price,
                tp_price=tp_price,
                broker_ticket=self.next_ticket,
                order_id=intent.broker_ticket,
                last_confirmed_time_msc=bar.start_time_msc,
                entry_spread_slippage_cost_usd=entry_spread_slippage_cost_usd,
                entry_spread_cost_usd=entry_spread_cost_usd,
                entry_slippage_cost_usd=entry_slippage_cost_usd,
                entry_commission_usd=entry_commission_usd,
            )
            self.execution_log.append(
                {
                    "event": "order_executed",
                    "side": "open",
                    "reason": "ENTRY",
                    "direction": int(intent.action.direction),
                    "volume": float(fill_volume),
                    "reference_price": float(bar.open),
                    "fill_price": float(open_price),
                    "forced": False,
                    "time_msc": int(bar.start_time_msc),
                    "ticket": int(self.next_ticket),
                }
            )
            self.next_ticket += 1
            turnover_lots += float(fill_volume)
        return turnover_lots

    def _mark_stops(self, bar: VolumeBar) -> float:
        position = self._position
        if position.direction == 0 or position.entry_price is None:
            return 0.0
        if position.direction > 0:
            if position.sl_price is not None and bar.low <= position.sl_price:
                return self._close_position(
                    self._execution_price(
                        float(position.sl_price),
                        position.direction,
                        avg_spread=bar.avg_spread,
                        is_entry=False,
                    ),
                    "SL",
                    bar.end_time_msc,
                    reference_price=float(position.sl_price),
                )
            elif position.tp_price is not None and bar.high >= position.tp_price:
                return self._close_position(
                    self._execution_price(
                        float(position.tp_price),
                        position.direction,
                        avg_spread=bar.avg_spread,
                        is_entry=False,
                    ),
                    "TP",
                    bar.end_time_msc,
                    reference_price=float(position.tp_price),
                )
        else:
            if position.sl_price is not None and bar.high >= position.sl_price:
                return self._close_position(
                    self._execution_price(
                        float(position.sl_price),
                        position.direction,
                        avg_spread=bar.avg_spread,
                        is_entry=False,
                    ),
                    "SL",
                    bar.end_time_msc,
                    reference_price=float(position.sl_price),
                )
            elif position.tp_price is not None and bar.low <= position.tp_price:
                return self._close_position(
                    self._execution_price(
                        float(position.tp_price),
                        position.direction,
                        avg_spread=bar.avg_spread,
                        is_entry=False,
                    ),
                    "TP",
                    bar.end_time_msc,
                    reference_price=float(position.tp_price),
                )
        return 0.0

    def advance_bar(self, bar: VolumeBar) -> float:
        self._bar_index += 1
        turnover_lots = self._fill_pending(bar)
        turnover_lots += self._mark_stops(bar)
        return float(turnover_lots)

    def force_flatten(self, bar: VolumeBar, *, reason: str = "FORCED_END_OF_PATH") -> float:
        if self._position.direction == 0 or self._position.entry_price is None:
            return 0.0
        exit_price = self._execution_price(
            float(bar.close),
            self._position.direction,
            avg_spread=bar.avg_spread,
            is_entry=False,
        )
        return self._close_position(
            exit_price,
            reason,
            bar.end_time_msc,
            reference_price=float(bar.close),
            forced=True,
        )

    def current_position(self, symbol: str) -> BrokerPositionSnapshot:
        if symbol.upper() != self.symbol.upper():
            return BrokerPositionSnapshot(symbol=symbol.upper())
        return self._position

    def current_equity(
        self,
        symbol: str,
        mark_price: float | None = None,
        *,
        avg_spread: float = 0.0,
        mark_to_liquidation: bool = True,
    ) -> float:
        if symbol.upper() != self.symbol.upper():
            return self.equity
        if self._position.direction == 0 or self._position.entry_price is None or mark_price is None:
            return self.equity
        liquidation_price = float(mark_price)
        if mark_to_liquidation:
            liquidation_price = self._execution_price(
                mark_price,
                self._position.direction,
                avg_spread=avg_spread,
                is_entry=False,
            )
        pip_pnl = price_to_pips(self.symbol, liquidation_price - float(self._position.entry_price))
        if self._position.direction < 0:
            pip_pnl = -pip_pnl
        pip_value = pip_value_for_volume(
            self.symbol,
            price=liquidation_price,
            volume_lots=self._position.volume,
            account_currency=self.account_currency,
        )
        close_commission_usd = self._commission_usd(self._position.volume) if mark_to_liquidation else 0.0
        return float(self.equity + (pip_pnl * pip_value) - close_commission_usd)
