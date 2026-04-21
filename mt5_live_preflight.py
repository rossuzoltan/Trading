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
from mt5_broker_caps import describe_trade_mode, read_symbol_caps, trade_mode_allows_open
from project_paths import resolve_manifest_path, resolve_selector_manifest_path
from selector_manifest import load_selector_manifest, validate_paper_live_candidate_manifest
from summarize_execution_audit import build_summary
from trading_config import deployment_paths, resolve_bar_construction_ticks_per_bar
from validation_metrics import load_json_report


if load_dotenv is not None:
    load_dotenv()


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


def _load_live_manifest(
    *,
    symbol: str,
    preferred_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    diagnostics: list[str] = []
    selector_path = resolve_selector_manifest_path(symbol=symbol, preferred=preferred_path, required=False)
    if selector_path is not None:
        try:
            selector_manifest = load_selector_manifest(
                selector_path,
                verify_manifest_hash=True,
                strict_manifest_hash=True,
                require_component_hashes=True,
            )
            validate_paper_live_candidate_manifest(selector_manifest)
            return (
                {
                    "path": str(selector_path),
                    "engine_type": str(selector_manifest.engine_type),
                    "release_stage": str(selector_manifest.release_stage),
                    "manifest_hash": str(selector_manifest.manifest_hash),
                    "strategy_symbol": str(selector_manifest.strategy_symbol),
                    "ticks_per_bar": selector_manifest.bar_construction_ticks_per_bar or selector_manifest.ticks_per_bar,
                    "manifest": selector_manifest,
                    "source": "selector_manifest",
                },
                diagnostics,
            )
        except Exception as exc:
            diagnostics.append(f"Selector manifest load failed at {selector_path}: {exc}")

    try:
        artifact_path = resolve_manifest_path(symbol=symbol, preferred=preferred_path)
        artifact_manifest = load_manifest(artifact_path)
        return (
            {
                "path": str(artifact_path),
                "engine_type": "RL",
                "release_stage": "legacy",
                "manifest_hash": None,
                "strategy_symbol": str(artifact_manifest.strategy_symbol),
                "ticks_per_bar": artifact_manifest.bar_construction_ticks_per_bar or artifact_manifest.ticks_per_bar,
                "manifest": artifact_manifest,
                "source": "artifact_manifest",
            },
            diagnostics,
        )
    except Exception as exc:
        diagnostics.append(f"Artifact manifest load failed: {exc}")
        return None, diagnostics


def build_report(symbol: str, ticks_per_bar: int, *, manifest_path: str | Path | None = None) -> dict[str, Any]:
    symbol = symbol.upper()
    paths = deployment_paths(symbol)
    blockers: list[str] = []
    warnings: list[str] = []
    order_magic = int(os.environ.get("TRADING_ORDER_MAGIC", "123456"))
    allow_foreign_positions = os.environ.get("LIVE_ALLOW_FOREIGN_POSITIONS", "0") == "1"
    allow_untagged_positions = os.environ.get("LIVE_ALLOW_UNTAGGED_POSITIONS", "0") == "1"

    gate = load_json_report(paths.gate_path) if paths.gate_path.exists() else None
    ops_attestation = load_json_report(paths.ops_attestation_path) if paths.ops_attestation_path.exists() else None
    manifest_bundle, manifest_diagnostics = _load_live_manifest(symbol=symbol, preferred_path=manifest_path)
    execution_audit = build_summary(symbol)
    manifest = None if manifest_bundle is None else manifest_bundle["manifest"]
    resolved_manifest_path = None if manifest_bundle is None else str(manifest_bundle["path"])
    manifest_ticks = None if manifest_bundle is None else manifest_bundle["ticks_per_bar"]

    # Preflight is a *technical* readiness check (MT5 connectivity, account mode,
    # symbol caps, isolation). Profitability gates and ops sign-offs are separate
    # artifacts and should not be prerequisites for preflight success.
    if gate is None:
        warnings.append(f"Deployment gate missing: {paths.gate_path}")
    else:
        if not bool(gate.get("approved_for_live", False)):
            warnings.append("Deployment gate is not approved for live trading.")
        for blocker in gate.get("blockers", []):
            warnings.append(f"Gate blocker: {blocker}")

    if manifest_bundle is None:
        blockers.append("No live manifest could be loaded for the requested symbol.")
        blockers.extend(f"Manifest diagnostic: {entry}" for entry in manifest_diagnostics)
    elif manifest_ticks is None:
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
            caps = read_symbol_caps(symbol, symbol_info) if symbol_info is not None else None
            _append_blocker(blockers, terminal_info is None, "MT5 terminal_info() returned None.")
            _append_blocker(blockers, account_info is None, "MT5 account_info() returned None.")
            _append_blocker(blockers, symbol_info is None, f"MT5 symbol_info() returned None for {symbol}.")
            if terminal_info is not None:
                trade_allowed = bool(getattr(terminal_info, "trade_allowed")) if hasattr(terminal_info, "trade_allowed") else None
                _append_blocker(blockers, trade_allowed is False, "MT5 terminal has trading disabled.")
                connected_flag = bool(getattr(terminal_info, "connected")) if hasattr(terminal_info, "connected") else None
                _append_blocker(blockers, connected_flag is False, "MT5 terminal is not connected to the broker.")
            if account_info is not None:
                trade_allowed = bool(getattr(account_info, "trade_allowed")) if hasattr(account_info, "trade_allowed") else None
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
            if caps is not None:
                if caps.visible is False:
                    blockers.append(f"MT5 symbol {symbol} is still not visible after symbol_select.")
                if caps.volume_min in (None, 0):
                    warnings.append("Broker did not expose volume_min; order sizing normalization may be incomplete.")
                if caps.volume_max in (None, 0):
                    warnings.append("Broker did not expose volume_max; order sizing normalization may be incomplete.")
                if caps.volume_step in (None, 0):
                    warnings.append("Broker did not expose volume_step; lot normalization may be incomplete.")
                if caps.trade_stops_level is None:
                    warnings.append("Broker did not expose trade_stops_level; stop-distance prechecks may be incomplete.")
                if caps.trade_freeze_level is None:
                    warnings.append("Broker did not expose trade_freeze_level; freeze-distance prechecks may be incomplete.")
                if caps.tick_size in (None, 0):
                    warnings.append("Broker did not expose trade_tick_size; price alignment may rely on point size only.")
                if caps.tick_value in (None, 0):
                    warnings.append("Broker did not expose trade_tick_value; audit economics may be incomplete.")
                if caps.contract_size in (None, 0):
                    warnings.append("Broker did not expose trade_contract_size; symbol economics may be incomplete.")
                if caps.trade_mode is not None:
                    if not trade_mode_allows_open(mt5, caps.trade_mode, 1) and not trade_mode_allows_open(mt5, caps.trade_mode, -1):
                        blockers.append(
                            f"MT5 symbol trade mode blocks new orders: {describe_trade_mode(mt5, caps.trade_mode)}."
                        )
                    elif not (
                        trade_mode_allows_open(mt5, caps.trade_mode, 1)
                        and trade_mode_allows_open(mt5, caps.trade_mode, -1)
                    ):
                        warnings.append(
                            f"MT5 symbol trade mode is directional-only: {describe_trade_mode(mt5, caps.trade_mode)}."
                        )
            mt5.shutdown()

    if ops_attestation is None:
        warnings.append(f"Ops attestation missing: {paths.ops_attestation_path}")
    min_audit_fills = int(os.environ.get("LIVE_MIN_AUDIT_FILLS", "20"))
    if int(execution_audit.get("accepted_count", 0) or 0) < min_audit_fills:
        warnings.append(
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
        "manifest_path": resolved_manifest_path,
        "manifest_source": None if manifest_bundle is None else manifest_bundle["source"],
        "manifest_engine_type": None if manifest_bundle is None else manifest_bundle["engine_type"],
        "manifest_release_stage": None if manifest_bundle is None else manifest_bundle["release_stage"],
        "manifest_hash": None if manifest_bundle is None else manifest_bundle["manifest_hash"],
        "manifest_strategy_symbol": None if manifest_bundle is None else manifest_bundle["strategy_symbol"],
        "manifest_diagnostics": manifest_diagnostics,
        "manifest_bar_construction_ticks_per_bar": manifest_ticks,
        "manifest_ticks_per_bar": manifest_ticks,
        "execution_audit_summary": execution_audit,
        "deployment_gate": None if gate is None else {"approved_for_live": bool(gate.get("approved_for_live", False)), "path": str(paths.gate_path)},
        "ops_attestation": None if ops_attestation is None else {"approved": bool(ops_attestation.get("approved", False)), "path": str(paths.ops_attestation_path)},
        "order_magic": order_magic,
        "account_margin_mode": None if account_info is None else getattr(account_info, "margin_mode", None),
        "account_mode_supported": bool(account_mode_supported),
        "terminal_info": None if terminal_info is None else terminal_info._asdict(),
        "account_info": None if account_info is None else account_info._asdict(),
        "symbol_info": None if symbol_info is None else symbol_info._asdict(),
        "symbol_capabilities": None
        if symbol_info is None
        else {
            "visible": getattr(symbol_info, "visible", None),
            "digits": getattr(symbol_info, "digits", None),
            "point": getattr(symbol_info, "point", None),
            "trade_mode": getattr(symbol_info, "trade_mode", None),
            "trade_stops_level": getattr(symbol_info, "trade_stops_level", None),
            "trade_freeze_level": getattr(symbol_info, "trade_freeze_level", None),
            "volume_min": getattr(symbol_info, "volume_min", None),
            "volume_max": getattr(symbol_info, "volume_max", None),
            "volume_step": getattr(symbol_info, "volume_step", None),
            "trade_tick_size": getattr(symbol_info, "trade_tick_size", None),
            "trade_tick_value": getattr(symbol_info, "trade_tick_value", None),
            "trade_contract_size": getattr(symbol_info, "trade_contract_size", None),
        },
    }
    paths.live_preflight_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight MT5 live deployment readiness.")
    parser.add_argument("--symbol", default=os.environ.get("TRADING_SYMBOL", "EURUSD"))
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument(
        "--ticks-per-bar",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    ticks_per_bar = args.ticks_per_bar
    if ticks_per_bar is None:
        manifest_path = args.manifest_path or resolve_selector_manifest_path(symbol=args.symbol, required=False)
        if manifest_path is not None:
            try:
                selector_manifest = load_selector_manifest(manifest_path, verify_manifest_hash=False)
                ticks_per_bar = int(
                    selector_manifest.ticks_per_bar or selector_manifest.bar_construction_ticks_per_bar or 0
                ) or None
            except Exception:
                ticks_per_bar = None
        if ticks_per_bar is None:
            ticks_per_bar = resolve_bar_construction_ticks_per_bar("BAR_SPEC_TICKS_PER_BAR", "TRADING_TICKS_PER_BAR")

    report = build_report(args.symbol, int(ticks_per_bar), manifest_path=args.manifest_path)
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
