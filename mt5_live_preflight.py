from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from interpreter_guard import ensure_project_venv

ensure_project_venv(project_root=Path(__file__).resolve().parent, script_path=__file__)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from artifact_manifest import load_manifest
from project_paths import resolve_manifest_path
from summarize_execution_audit import build_summary
from trading_config import deployment_paths, resolve_bar_construction_ticks_per_bar
from validation_metrics import load_json_report


if load_dotenv is not None:
    load_dotenv()


def _bool_attr(obj: Any, name: str) -> bool | None:
    if obj is None or not hasattr(obj, name):
        return None
    return bool(getattr(obj, name))


def _append_blocker(blockers: list[str], condition: bool, message: str) -> None:
    if condition:
        blockers.append(message)


def _strategy_positions(mt5_module: Any, symbol: str, order_magic: int) -> tuple[list[Any], list[Any]]:
    positions = list(mt5_module.positions_get(symbol=symbol) or [])
    if not positions:
        return [], []
    if not any(hasattr(position, "magic") for position in positions):
        return positions, []
    strategy = [position for position in positions if int(getattr(position, "magic", 0) or 0) == order_magic]
    foreign = [position for position in positions if int(getattr(position, "magic", 0) or 0) != order_magic]
    return strategy, foreign


def _connect_mt5(mt5_module: Any) -> tuple[bool, str | None]:
    login = int(os.environ.get("MT5_LOGIN", "0") or 0)
    password = os.environ.get("MT5_PASSWORD", "")
    server = os.environ.get("MT5_SERVER", "")
    if not login or not password or not server:
        return False, "MT5 credentials missing. Set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER."
    if not mt5_module.initialize():
        return False, f"MT5 initialize() failed: {getattr(mt5_module, 'last_error', lambda: 'unknown')()}"
    if not mt5_module.login(login, password, server):
        return False, f"MT5 login() failed: {getattr(mt5_module, 'last_error', lambda: 'unknown')()}"
    return True, None


def build_report(symbol: str, ticks_per_bar: int) -> dict[str, Any]:
    symbol = symbol.upper()
    paths = deployment_paths(symbol)
    blockers: list[str] = []
    warnings: list[str] = []
    order_magic = int(os.environ.get("TRADING_ORDER_MAGIC", "123456"))
    allow_foreign_positions = os.environ.get("LIVE_ALLOW_FOREIGN_POSITIONS", "0") == "1"
    allow_untagged_positions = os.environ.get("LIVE_ALLOW_UNTAGGED_POSITIONS", "0") == "1"

    gate = load_json_report(paths.gate_path) if paths.gate_path.exists() else None
    ops_attestation = load_json_report(paths.ops_attestation_path) if paths.ops_attestation_path.exists() else None
    manifest = load_manifest(resolve_manifest_path(symbol=symbol))
    execution_audit = build_summary(symbol)

    _append_blocker(blockers, gate is None, f"Deployment gate missing: {paths.gate_path}")
    if gate is not None:
        _append_blocker(blockers, not bool(gate.get("approved_for_live", False)), "Deployment gate is not approved for live trading.")
        for blocker in gate.get("blockers", []):
            blockers.append(f"Gate blocker: {blocker}")

    manifest_ticks = manifest.bar_construction_ticks_per_bar or manifest.ticks_per_bar
    if manifest_ticks is None:
        blockers.append("Artifact manifest does not declare bar_construction_ticks_per_bar; live bar-spec parity is unproven.")
    elif int(manifest_ticks) != int(ticks_per_bar):
        blockers.append(
            f"Manifest bar_construction_ticks_per_bar={manifest_ticks} differs from requested live bar_construction_ticks_per_bar={ticks_per_bar}."
        )

    kill_switch_path = Path(os.environ.get("LIVE_KILL_SWITCH_PATH", "live.kill"))
    _append_blocker(blockers, kill_switch_path.exists(), f"Manual kill switch exists: {kill_switch_path}")

    mt5_installed = importlib.util.find_spec("MetaTrader5") is not None
    _append_blocker(blockers, not mt5_installed, "MetaTrader5 package is not installed in the active environment.")

    terminal_info = None
    account_info = None
    symbol_info = None
    account_mode_supported = False
    if mt5_installed:
        import MetaTrader5 as mt5

        connected, error = _connect_mt5(mt5)
        if not connected:
            blockers.append(error or "Failed to connect to MetaTrader5.")
        else:
            terminal_info = mt5.terminal_info()
            account_info = mt5.account_info()
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is not None and not getattr(symbol_info, "visible", True) and hasattr(mt5, "symbol_select"):
                mt5.symbol_select(symbol, True)
                symbol_info = mt5.symbol_info(symbol)
            _append_blocker(blockers, terminal_info is None, "MT5 terminal_info() returned None.")
            _append_blocker(blockers, account_info is None, "MT5 account_info() returned None.")
            _append_blocker(blockers, symbol_info is None, f"MT5 symbol_info() returned None for {symbol}.")
            if terminal_info is not None:
                trade_allowed = _bool_attr(terminal_info, "trade_allowed")
                _append_blocker(blockers, trade_allowed is False, "MT5 terminal has trading disabled.")
                connected_flag = _bool_attr(terminal_info, "connected")
                _append_blocker(blockers, connected_flag is False, "MT5 terminal is not connected to the broker.")
            if account_info is not None:
                trade_allowed = _bool_attr(account_info, "trade_allowed")
                _append_blocker(blockers, trade_allowed is False, "MT5 account reports trading disabled.")
                margin_mode = getattr(account_info, "margin_mode", None)
                hedging_constant = getattr(mt5, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", None)
                account_mode_supported = True
                if hedging_constant is not None and margin_mode is not None and int(margin_mode) == int(hedging_constant):
                    account_mode_supported = False
                    blockers.append("MT5 account is in hedging mode. Deployment requires a netting account.")
            strategy_positions, foreign_positions = _strategy_positions(mt5, symbol, order_magic)
            raw_positions = list(mt5.positions_get(symbol=symbol) or [])
            if raw_positions and not all(hasattr(position, "magic") for position in raw_positions):
                if allow_untagged_positions:
                    warnings.append("Broker positions do not expose magic; override enabled via LIVE_ALLOW_UNTAGGED_POSITIONS=1.")
                else:
                    blockers.append(
                        "Broker positions do not expose magic. Live reconciliation requires tagged strategy positions."
                    )
            if strategy_positions:
                directions = {1 if int(getattr(position, "type", 0)) == 0 else -1 for position in strategy_positions}
                if len(directions) > 1:
                    blockers.append("Mixed-direction strategy positions already exist on the hedging account.")
            if foreign_positions:
                message = (
                    f"Found {len(foreign_positions)} non-strategy position(s) for {symbol}; "
                    "account-level risk limits would be contaminated."
                )
                if allow_foreign_positions:
                    warnings.append(message + " Override enabled via LIVE_ALLOW_FOREIGN_POSITIONS=1.")
                else:
                    blockers.append(message)
            if symbol_info is not None:
                volume_min = getattr(symbol_info, "volume_min", None)
                volume_step = getattr(symbol_info, "volume_step", None)
                stops_level = getattr(symbol_info, "trade_stops_level", None)
                if volume_min in (None, 0):
                    warnings.append("Broker did not expose volume_min; order sizing normalization may be incomplete.")
                if volume_step in (None, 0):
                    warnings.append("Broker did not expose volume_step; lot normalization may be incomplete.")
                if stops_level in (None,):
                    warnings.append("Broker did not expose trade_stops_level; stop-distance prechecks may be incomplete.")
            mt5.shutdown()

    if ops_attestation is None:
        blockers.append(f"Ops attestation missing: {paths.ops_attestation_path}")
    min_audit_fills = int(os.environ.get("LIVE_MIN_AUDIT_FILLS", "20"))
    if int(execution_audit.get("accepted_count", 0) or 0) < min_audit_fills:
        blockers.append(
            f"Execution audit has only {execution_audit.get('accepted_count', 0)} accepted fills; "
            f"need at least {min_audit_fills} to assess live-vs-replay drift."
        )

    report = {
        "symbol": symbol,
        "bar_construction_ticks_per_bar": int(ticks_per_bar),
        "ticks_per_bar": int(ticks_per_bar),
        "approved_for_live_runtime": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "gate_path": str(paths.gate_path),
        "ops_attestation_path": str(paths.ops_attestation_path),
        "manifest_path": str(resolve_manifest_path(symbol=symbol)),
        "manifest_bar_construction_ticks_per_bar": manifest_ticks,
        "manifest_ticks_per_bar": manifest.ticks_per_bar,
        "execution_audit_summary": execution_audit,
        "order_magic": order_magic,
        "account_margin_mode": None if account_info is None else getattr(account_info, "margin_mode", None),
        "account_mode_supported": bool(account_mode_supported),
        "terminal_info": None if terminal_info is None else terminal_info._asdict(),
        "account_info": None if account_info is None else account_info._asdict(),
        "symbol_info": None if symbol_info is None else symbol_info._asdict(),
    }
    paths.live_preflight_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight MT5 live deployment readiness.")
    parser.add_argument("--symbol", default=os.environ.get("TRADING_SYMBOL", "EURUSD"))
    parser.add_argument(
        "--ticks-per-bar",
        type=int,
        default=resolve_bar_construction_ticks_per_bar("BAR_SPEC_TICKS_PER_BAR", "TRADING_TICKS_PER_BAR"),
    )
    args = parser.parse_args()

    report = build_report(args.symbol, args.ticks_per_bar)
    print("=" * 80)
    print(f"MT5 Live Preflight - {report['symbol']}")
    print("=" * 80)
    for blocker in report["blockers"]:
        print(f"BLOCKER: {blocker}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Report: {deployment_paths(report['symbol']).live_preflight_path}")
    print("Verdict: " + ("BUILD NOW" if report["approved_for_live_runtime"] else "DO NOT DEPLOY"))
    return 0 if report["approved_for_live_runtime"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
