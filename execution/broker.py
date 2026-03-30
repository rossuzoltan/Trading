from __future__ import annotations

from domain.models import (
    BrokerPositionSnapshot,
    OrderIntent,
    SubmitResult,
    VolumeBar,
)


class BaseBroker:
    def advance_bar(self, bar: VolumeBar) -> float:
        return 0.0

    def submit_order(self, intent: OrderIntent) -> SubmitResult:
        raise NotImplementedError

    def current_position(self, symbol: str) -> BrokerPositionSnapshot:
        raise NotImplementedError

    def current_equity(
        self,
        symbol: str,
        mark_price: float | None = None,
        *,
        avg_spread: float = 0.0,
        mark_to_liquidation: bool = True,
    ) -> float:
        raise NotImplementedError
