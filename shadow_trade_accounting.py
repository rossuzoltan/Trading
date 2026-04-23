from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from symbol_utils import pip_value_per_lot, price_to_pips


@dataclass(frozen=True)
class ShadowTrade:
    direction: int
    open_event_index: int | None
    close_event_index: int | None
    open_ts_utc: str | None
    close_ts_utc: str | None
    entry_fill_price: float
    exit_fill_price: float
    gross_pips: float
    commission_pips: float
    slippage_pips: float
    net_pips: float
    bars_held: int | None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if math.isfinite(numeric) else None


def _event_ts(row: dict[str, Any]) -> str | None:
    for key in ("timestamp_utc", "bar_end_ts_utc", "bar_ts", "bar_start_ts_utc"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return None


def _fill_price_from_snapshot(snapshot: dict[str, Any], direction: int, *, phase: str) -> float | None:
    fill_price = _safe_float(snapshot.get("fill_price"))
    if fill_price is not None:
        return fill_price
    if phase == "entry":
        return _safe_float(snapshot.get("ask_proxy")) if int(direction) > 0 else _safe_float(snapshot.get("bid_proxy"))
    return _safe_float(snapshot.get("bid_proxy")) if int(direction) > 0 else _safe_float(snapshot.get("ask_proxy"))


def _entry_fill_price_from_exit_snapshot(snapshot: dict[str, Any], direction: int) -> float | None:
    if int(direction) > 0:
        return _safe_float(snapshot.get("entry_ask_proxy"))
    return _safe_float(snapshot.get("entry_bid_proxy"))


def _commission_pips(
    *,
    symbol: str,
    fill_price: float | None,
    commission_per_lot: float | None,
    account_currency: str,
    issues: list[str],
    issue_context: str,
) -> float:
    if fill_price is None or commission_per_lot in (None, 0.0):
        return 0.0
    try:
        pip_value = float(pip_value_per_lot(symbol, price=float(fill_price), account_currency=account_currency))
    except Exception as exc:
        issues.append(f"{issue_context}: commission pip conversion failed: {exc}")
        return 0.0
    if pip_value <= 0:
        issues.append(f"{issue_context}: non-positive pip value")
        return 0.0
    return float(commission_per_lot) / pip_value


def summarize_shadow_trade_accounting(
    *,
    events: list[dict[str, Any]],
    symbol: str,
    commission_per_lot: float | None = None,
    slippage_pips: float | None = None,
    account_currency: str = "USD",
) -> dict[str, Any]:
    trades: list[ShadowTrade] = []
    issues: list[str] = []
    open_trade: dict[str, Any] | None = None
    close_event_count = 0
    entry_event_count = 0
    priced_close_event_count = 0
    priced_entry_event_count = 0

    for row in events:
        entry_snapshot = row.get("entry_snapshot") if isinstance(row.get("entry_snapshot"), dict) else None
        exit_snapshot = row.get("exit_snapshot") if isinstance(row.get("exit_snapshot"), dict) else None
        expects_close = bool(row.get("would_close", False)) or exit_snapshot is not None
        expects_open = bool(row.get("would_open", False)) or entry_snapshot is not None

        if expects_close:
            close_event_count += 1
        if exit_snapshot is not None:
            direction = int(exit_snapshot.get("direction_closing", 0) or 0)
            if direction == 0:
                issues.append(f"event {row.get('event_index')}: exit snapshot missing direction_closing")
            if open_trade is None and direction != 0:
                open_trade = {
                    "direction": direction,
                    "open_event_index": exit_snapshot.get("opened_at_event_index"),
                    "open_ts_utc": exit_snapshot.get("opened_at_bar_ts_utc"),
                    "entry_fill_price": _entry_fill_price_from_exit_snapshot(exit_snapshot, direction),
                }
            entry_fill_price = None if open_trade is None else _safe_float(open_trade.get("entry_fill_price"))
            if entry_fill_price is None and direction != 0:
                entry_fill_price = _entry_fill_price_from_exit_snapshot(exit_snapshot, direction)
            exit_fill_price = _fill_price_from_snapshot(exit_snapshot, direction, phase="exit") if direction != 0 else None
            if entry_fill_price is not None:
                priced_entry_event_count += 1
            if exit_fill_price is not None:
                priced_close_event_count += 1
            if direction != 0 and entry_fill_price is not None and exit_fill_price is not None:
                gross_pips = float(price_to_pips(symbol, float(exit_fill_price) - float(entry_fill_price))) * float(direction)
                trade_issues: list[str] = []
                total_commission_pips = _commission_pips(
                    symbol=symbol,
                    fill_price=entry_fill_price,
                    commission_per_lot=commission_per_lot,
                    account_currency=account_currency,
                    issues=trade_issues,
                    issue_context=f"event {row.get('event_index')}",
                )
                total_commission_pips += _commission_pips(
                    symbol=symbol,
                    fill_price=exit_fill_price,
                    commission_per_lot=commission_per_lot,
                    account_currency=account_currency,
                    issues=trade_issues,
                    issue_context=f"event {row.get('event_index')}",
                )
                total_slippage_pips = 2.0 * max(float(slippage_pips or 0.0), 0.0)
                net_pips = float(gross_pips - total_commission_pips - total_slippage_pips)
                trades.append(
                    ShadowTrade(
                        direction=direction,
                        open_event_index=None if open_trade is None else open_trade.get("open_event_index"),
                        close_event_index=int(row.get("event_index", 0) or 0) or None,
                        open_ts_utc=None if open_trade is None else open_trade.get("open_ts_utc"),
                        close_ts_utc=_event_ts(row),
                        entry_fill_price=float(entry_fill_price),
                        exit_fill_price=float(exit_fill_price),
                        gross_pips=float(gross_pips),
                        commission_pips=float(total_commission_pips),
                        slippage_pips=float(total_slippage_pips),
                        net_pips=net_pips,
                        bars_held=(
                            int(exit_snapshot.get("bars_held"))
                            if exit_snapshot.get("bars_held") not in (None, "")
                            else None
                        ),
                    )
                )
                issues.extend(trade_issues)
            else:
                issues.append(f"event {row.get('event_index')}: incomplete snapshot pricing for realized close")
            open_trade = None
        elif expects_close:
            issues.append(f"event {row.get('event_index')}: missing exit snapshot for close event")
            open_trade = None

        if expects_open:
            entry_event_count += 1
        if entry_snapshot is not None:
            direction = int(entry_snapshot.get("direction_opening", 0) or 0)
            fill_price = _fill_price_from_snapshot(entry_snapshot, direction, phase="entry") if direction != 0 else None
            if fill_price is not None:
                priced_entry_event_count += 1
            else:
                issues.append(f"event {row.get('event_index')}: incomplete snapshot pricing for entry")
            open_trade = {
                "direction": direction,
                "open_event_index": (
                    int(entry_snapshot.get("event_index"))
                    if entry_snapshot.get("event_index") not in (None, "")
                    else (int(row.get("event_index", 0) or 0) or None)
                ),
                "open_ts_utc": str(entry_snapshot.get("bar_ts_utc") or _event_ts(row) or ""),
                "entry_fill_price": fill_price,
            }
        elif expects_open:
            issues.append(f"event {row.get('event_index')}: missing entry snapshot for open event")

    net_pips_list = [float(trade.net_pips) for trade in trades]
    trade_count = len(net_pips_list)
    net_pips_total = float(sum(net_pips_list))
    avg_net_pips = float(net_pips_total / trade_count) if trade_count else None
    win_rate = float(sum(1 for value in net_pips_list if value > 0.0) / trade_count) if trade_count else None
    ci_low = None
    ci_high = None
    if trade_count == 1:
        ci_low = avg_net_pips
        ci_high = avg_net_pips
    elif trade_count > 1 and avg_net_pips is not None:
        mean = avg_net_pips
        variance = sum((value - mean) ** 2 for value in net_pips_list) / float(trade_count - 1)
        stderr = math.sqrt(max(variance, 0.0) / float(trade_count))
        ci_low = float(mean - (1.96 * stderr))
        ci_high = float(mean + (1.96 * stderr))

    realized_trade_coverage = (
        float(trade_count / close_event_count)
        if close_event_count > 0
        else 1.0
    )

    return {
        "pricing_mode": "event_snapshots",
        "trade_count": trade_count,
        "close_event_count": int(close_event_count),
        "entry_event_count": int(entry_event_count),
        "priced_close_event_count": int(priced_close_event_count),
        "priced_entry_event_count": int(priced_entry_event_count),
        "realized_trade_coverage": float(realized_trade_coverage),
        "open_position_remaining": bool(open_trade is not None),
        "net_pips": net_pips_total if trade_count else None,
        "avg_net_pips": avg_net_pips,
        "avg_net_pips_ci95_low": ci_low,
        "avg_net_pips_ci95_high": ci_high,
        "win_rate": win_rate,
        "issues": list(issues),
        "trades": [asdict(trade) for trade in trades],
    }
