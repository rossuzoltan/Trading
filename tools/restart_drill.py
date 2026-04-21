from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path to ensure infra modules can be imported from tools/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd

from interpreter_guard import ensure_project_venv

ensure_project_venv(project_root=Path(__file__).resolve().parent, script_path=__file__)

import live_bridge
from event_pipeline import JsonStateStore, RuntimeSnapshot, VolumeBar
from trading_config import deployment_paths, resolve_bar_construction_ticks_per_bar


@dataclass(frozen=True)
class RestartDrillReport:
    symbol: str
    ticks_per_bar: int
    state_path: str
    report_path: str
    evidence_mode: str
    attestable_for_live: bool
    startup_reconcile_ok: bool
    state_restored_ok: bool
    confirmed_position_restored_ok: bool
    bars_processed_before_restart: int
    bars_processed_after_restart: int
    pre_restart_snapshot: dict[str, Any]
    post_restart_snapshot: dict[str, Any]
    notes: list[str]


@dataclass(frozen=True)
class FakeMt5Position:
    type: int = 0
    volume: float = 0.01
    price_open: float = 1.1000
    sl: float = 0.0
    tp: float = 0.0
    ticket: int = 1001
    identifier: int = 2001


class FakeMt5:
    COPY_TICKS_ALL = 0
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_FOK = 2
    ORDER_FILLING_RETURN = 3
    TRADE_RETCODE_DONE = 10009
    ACCOUNT_MARGIN_MODE_RETAIL_NETTING = 0
    ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 2

    def __init__(self, *, position: FakeMt5Position | None = None, bid: float = 1.1000, ask: float = 1.1002) -> None:
        self._position = position if position is not None else FakeMt5Position()
        self._bid = bid
        self._ask = ask
        self._next_ticket = 1002
        self.initialized = False

    def initialize(self) -> bool:
        self.initialized = True
        return True

    def login(self, login, password, server) -> bool:
        return True

    def shutdown(self) -> None:
        self.initialized = False

    def last_error(self):
        return (0, "ok")

    def terminal_info(self):
        return SimpleNamespace(connected=True, trade_allowed=True)

    def account_info(self):
        return SimpleNamespace(
            equity=1000.0,
            trade_allowed=True,
            trade_expert=True,
            margin_mode=self.ACCOUNT_MARGIN_MODE_RETAIL_NETTING,
        )

    def symbol_info(self, symbol):
        return SimpleNamespace(
            point=0.0001 if not symbol.upper().endswith("JPY") else 0.01,
            visible=True,
            volume_min=0.01,
            volume_max=1.0,
            volume_step=0.01,
            trade_stops_level=0,
            trade_freeze_level=0,
        )

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=self._bid, ask=self._ask)

    def positions_get(self, symbol=None):
        if self._position is None:
            return []
        return [self._position]

    def symbol_select(self, symbol, visible):
        return True

    def order_send(self, request):
        if request.get("position"):
            self._position = None
        else:
            order_type = request.get("type")
            if order_type in (self.ORDER_TYPE_BUY, self.ORDER_TYPE_SELL):
                self._position = FakeMt5Position(
                    type=0 if int(order_type) == self.ORDER_TYPE_BUY else 1,
                    volume=float(request.get("volume", 0.01) or 0.01),
                    price_open=float(request.get("price", self._ask if int(order_type) == self.ORDER_TYPE_BUY else self._bid) or 0.0),
                    sl=float(request.get("sl", 0.0) or 0.0),
                    tp=float(request.get("tp", 0.0) or 0.0),
                    ticket=self._next_ticket,
                    identifier=self._next_ticket + 1,
                )
                self._next_ticket += 2
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=9001, price=request.get("price", 0.0))

    def copy_ticks_from(self, symbol, start_dt, count, flags):
        return []


def _snapshot_to_dict(snapshot: RuntimeSnapshot) -> dict[str, Any]:
    return {
        "cursor": asdict(snapshot.cursor),
        "bar_builder": asdict(snapshot.bar_builder),
        "confirmed_position": asdict(snapshot.confirmed_position),
        "last_equity": snapshot.last_equity,
        "high_water_mark": snapshot.high_water_mark,
        "day_start_equity": snapshot.day_start_equity,
        "last_reset_utc_date": snapshot.last_reset_utc_date,
        "consecutive_broker_failures": snapshot.consecutive_broker_failures,
        "last_tick_time_msc": snapshot.last_tick_time_msc,
        "kill_switch_active": snapshot.kill_switch_active,
        "kill_switch_reason": snapshot.kill_switch_reason,
        "safe_mode_active": snapshot.safe_mode_active,
    }


def _build_bar(index: int, *, base_time: pd.Timestamp, base_price: float) -> VolumeBar:
    timestamp = base_time + pd.Timedelta(hours=index)
    offset = index * 0.0002
    open_price = base_price + offset
    close_price = open_price + (0.0001 if index % 2 == 0 else -0.0001)
    high_price = max(open_price, close_price) + 0.00015
    low_price = min(open_price, close_price) - 0.00015
    start_time_msc = int(timestamp.timestamp() * 1000)
    return VolumeBar(
        timestamp=timestamp.to_pydatetime(),
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=5000.0,
        avg_spread=0.0002,
        time_delta_s=3600.0 if index else 0.0,
        start_time_msc=start_time_msc,
        end_time_msc=start_time_msc + 3_600_000,
    )


def _run_with_bootstrap(
    bootstrap_fn: Callable[..., tuple[Any, Any, JsonStateStore, Any]],
    *,
    symbol: str,
    state_path: str,
    report_path: str | Path | None,
    ticks_per_bar: int,
    mt5_module: Any,
    evidence_mode: str,
    attestable_for_live: bool,
    bars_before_restart: int,
    bars_after_restart: int,
) -> RestartDrillReport:
    runtime, _builder, store, _source = bootstrap_fn(
        symbol=symbol,
        state_path=state_path,
        ticks_per_bar=ticks_per_bar,
        mt5_module=mt5_module,
    )
    startup_reconcile_ok = bool(
        getattr(runtime.confirmed_position, "is_flat", False)
        or (
            getattr(runtime.confirmed_position, "last_confirmed_time_msc", None) is not None
            and getattr(runtime.confirmed_position, "broker_ticket", None) is not None
        )
    )

    pre_restart_snapshot = _snapshot_to_dict(runtime.snapshot)
    base_time = pd.Timestamp("2024-01-01T00:00:00Z")
    base_price = 1.1000
    for index in range(bars_before_restart):
        result = runtime.process_bar(_build_bar(index, base_time=base_time, base_price=base_price))
        runtime.persist()
        if result.kill_switch_active:
            break

    persisted_snapshot = _snapshot_to_dict(runtime.snapshot)
    runtime_after_restart, _builder_after, store_after, _source_after = bootstrap_fn(
        symbol=symbol,
        state_path=state_path,
        ticks_per_bar=ticks_per_bar,
        mt5_module=mt5_module,
    )
    post_restart_snapshot = _snapshot_to_dict(runtime_after_restart.snapshot)
    confirmed_position_restored_ok = (
        persisted_snapshot["confirmed_position"] == post_restart_snapshot["confirmed_position"]
    )
    state_restored_ok = (
        persisted_snapshot["cursor"] == post_restart_snapshot["cursor"]
        and persisted_snapshot["bar_builder"] == post_restart_snapshot["bar_builder"]
        and persisted_snapshot["last_tick_time_msc"] == post_restart_snapshot["last_tick_time_msc"]
        and float(persisted_snapshot["last_equity"]) == float(post_restart_snapshot["last_equity"])
        and float(persisted_snapshot["high_water_mark"]) == float(post_restart_snapshot["high_water_mark"])
        and float(persisted_snapshot["day_start_equity"]) == float(post_restart_snapshot["day_start_equity"])
        and persisted_snapshot["last_reset_utc_date"] == post_restart_snapshot["last_reset_utc_date"]
        and bool(persisted_snapshot["kill_switch_active"]) == bool(post_restart_snapshot["kill_switch_active"])
        and bool(persisted_snapshot["safe_mode_active"]) == bool(post_restart_snapshot["safe_mode_active"])
    )

    for index in range(bars_before_restart, bars_before_restart + bars_after_restart):
        runtime_after_restart.process_bar(_build_bar(index, base_time=base_time, base_price=base_price))
        runtime_after_restart.persist()

    report = RestartDrillReport(
        symbol=symbol.upper(),
        ticks_per_bar=int(ticks_per_bar),
        state_path=str(state_path),
        report_path=str(Path(report_path) if report_path is not None else Path("models") / f"restart_drill_{symbol.lower()}.json"),
        evidence_mode=evidence_mode,
        attestable_for_live=bool(attestable_for_live),
        startup_reconcile_ok=startup_reconcile_ok,
        state_restored_ok=state_restored_ok,
        confirmed_position_restored_ok=confirmed_position_restored_ok,
        bars_processed_before_restart=int(bars_before_restart),
        bars_processed_after_restart=int(bars_after_restart),
        pre_restart_snapshot=pre_restart_snapshot,
        post_restart_snapshot=post_restart_snapshot,
        notes=[],
    )
    return report


def run_restart_drill(
    *,
    symbol: str,
    state_path: str,
    report_path: str | Path | None = None,
    ticks_per_bar: int,
    mt5_module: Any | None = None,
    use_real_mt5: bool = False,
    allow_bar_spec_mismatch: bool = False,
    bars_before_restart: int = 2,
    bars_after_restart: int = 2,
) -> RestartDrillReport:
    evidence_mode = "real_mt5" if use_real_mt5 else "fake_mt5"
    attestable_for_live = bool(use_real_mt5 and not allow_bar_spec_mismatch)
    mt5_module = mt5_module or (live_bridge.mt5 if use_real_mt5 else FakeMt5())
    if allow_bar_spec_mismatch:
        os.environ.setdefault("LIVE_ALLOW_BAR_SPEC_MISMATCH", "1")

    try:
        report = _run_with_bootstrap(
            live_bridge.bootstrap_live_runtime,
            symbol=symbol,
            state_path=state_path,
            report_path=report_path,
            ticks_per_bar=ticks_per_bar,
            mt5_module=mt5_module,
            evidence_mode=evidence_mode,
            attestable_for_live=attestable_for_live,
            bars_before_restart=bars_before_restart,
            bars_after_restart=bars_after_restart,
        )
    except Exception as exc:
        # Offline RC1 fallback: produce a structured restart report so the
        # evidence chain has an explicit artifact, even though it is not live-attestable.
        report = RestartDrillReport(
            symbol=symbol.upper(),
            ticks_per_bar=int(ticks_per_bar),
            state_path=str(state_path),
            report_path=str(Path(report_path) if report_path is not None else Path("models") / f"restart_drill_{symbol.lower()}.json"),
            evidence_mode="offline_rc1_fallback",
            attestable_for_live=False,
            startup_reconcile_ok=False,
            state_restored_ok=False,
            confirmed_position_restored_ok=False,
            bars_processed_before_restart=int(bars_before_restart),
            bars_processed_after_restart=int(bars_after_restart),
            pre_restart_snapshot={"error": str(exc)},
            post_restart_snapshot={"error": str(exc)},
            notes=[f"Fallback generated because bootstrap failed: {exc}"],
        )
    output_path = Path(report.report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


def build_rc1_restart_drill(
    *,
    manifest_path: str | Path,
    state_path: str,
    report_path: str | Path | None = None,
    mt5_module: Any | None = None,
    use_real_mt5: bool = False,
    allow_bar_spec_mismatch: bool = False,
    bars_before_restart: int = 2,
    bars_after_restart: int = 2,
) -> RestartDrillReport:
    from selector_manifest import load_selector_manifest, validate_paper_live_candidate_manifest
    from paper_live_metrics import resolve_paper_live_gate_paths

    manifest = load_selector_manifest(manifest_path, verify_manifest_hash=True)
    validate_paper_live_candidate_manifest(manifest)
    ticks_per_bar = int(manifest.ticks_per_bar or manifest.bar_construction_ticks_per_bar or 0)
    symbol = manifest.strategy_symbol.upper()

    report = run_restart_drill(
        symbol=symbol,
        state_path=state_path,
        report_path=report_path,
        ticks_per_bar=ticks_per_bar,
        mt5_module=mt5_module,
        use_real_mt5=use_real_mt5,
        allow_bar_spec_mismatch=allow_bar_spec_mismatch,
        bars_before_restart=bars_before_restart,
        bars_after_restart=bars_after_restart,
    )
    payload = asdict(report)
    payload["manifest_path"] = str(Path(manifest_path))
    payload["manifest_hash"] = manifest.manifest_hash
    payload["logic_hash"] = manifest.logic_hash
    payload["evaluator_hash"] = manifest.evaluator_hash
    output_path = Path(report_path) if report_path is not None else Path("models") / f"restart_drill_{symbol.lower()}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a restart drill over the live runtime.")
    parser.add_argument("--symbol", default=os.environ.get("TRADING_SYMBOL", "EURUSD"))
    parser.add_argument("--state-path", default=os.environ.get("LIVE_STATE_PATH", "live_state.json"))
    parser.add_argument("--report-path", default=None)
    parser.add_argument(
        "--ticks-per-bar",
        type=int,
        default=resolve_bar_construction_ticks_per_bar("BAR_SPEC_TICKS_PER_BAR", "TRADING_TICKS_PER_BAR"),
    )
    parser.add_argument("--bars-before-restart", type=int, default=2)
    parser.add_argument("--bars-after-restart", type=int, default=2)
    parser.add_argument("--use-real-mt5", action="store_true")
    parser.add_argument("--allow-bar-spec-mismatch", action="store_true")
    args = parser.parse_args()

    report = run_restart_drill(
        symbol=args.symbol,
        state_path=args.state_path,
        report_path=args.report_path,
        ticks_per_bar=args.ticks_per_bar,
        use_real_mt5=args.use_real_mt5,
        allow_bar_spec_mismatch=args.allow_bar_spec_mismatch,
        bars_before_restart=args.bars_before_restart,
        bars_after_restart=args.bars_after_restart,
    )
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.state_restored_ok and report.confirmed_position_restored_ok and report.startup_reconcile_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())


