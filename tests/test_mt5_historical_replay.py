from __future__ import annotations

from tools.mt5_historical_replay import build_summary


def _make_record(*, would_open: bool, signal: int, spread_pips: float = 0.5) -> dict:
    return {
        "bar_ts": "2026-04-21T00:00:00+00:00",
        "hour_utc": 12,
        "session": "London/NY",
        "signal": int(signal),
        "allow_execution": True,
        "reason": "authorized" if would_open else "no signal",
        "spread_pips": float(spread_pips),
        "price_z": 0.0,
        "spread_z": 0.0,
        "time_delta_z": 0.0,
        "ma20_slope": 0.0,
        "ma50_slope": 0.0,
        "raw_price_signal": int(signal),
        "guard_failures": {"spread": False, "time_delta": False, "ma20": False, "ma50": False},
        "would_open": bool(would_open),
        "would_close": False,
        "would_hold": False,
        "would_flat": not bool(would_open),
        "active_state": "flat",
    }


def test_build_summary_uses_expected_long_share_for_direction_drift() -> None:
    records = []
    # 100 bars, 20 opens: 2 long, 18 short => live_long_share = 0.10
    records.extend([_make_record(would_open=True, signal=1) for _ in range(2)])
    records.extend([_make_record(would_open=True, signal=-1) for _ in range(18)])
    records.extend([_make_record(would_open=False, signal=0) for _ in range(80)])

    summary = build_summary(
        records,
        symbol="EURUSD",
        days=30,
        spread_backtest_pips=0.5,
        replay_trades_per_bar=0.2,  # live opens/bar = 0.2 => density ratio 1.0
        replay_trade_count=200,
        replay_bars=1000,
        expected_long_share=0.50,
    )

    assert summary["live_long_share"] == 0.10
    assert summary["expected_long_share"] == 0.50
    assert summary["directional_delta_pp"] == 40.0
    assert summary["overall_verdict"] == "DRIFT_CRITICAL"


def test_build_summary_does_not_degrade_overall_when_direction_reference_missing() -> None:
    records = []
    records.extend([_make_record(would_open=True, signal=1) for _ in range(10)])
    records.extend([_make_record(would_open=False, signal=0) for _ in range(90)])

    summary = build_summary(
        records,
        symbol="EURUSD",
        days=30,
        spread_backtest_pips=0.5,
        replay_trades_per_bar=0.1,  # live opens/bar = 0.1 => density ratio 1.0
        replay_trade_count=100,
        replay_bars=1000,
        expected_long_share=None,
    )

    assert summary["overall_verdict"] == "OK"

