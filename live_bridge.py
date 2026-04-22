from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from interpreter_guard import ensure_project_venv

ensure_project_venv(project_root=Path(__file__).resolve().parent, script_path=__file__)

import pandas as pd
from dotenv import load_dotenv

from artifact_manifest import (
    dataset_id_for_path,
    load_manifest,
    load_validated_model,
    load_validated_scaler,
    load_validated_vecnormalize,
)
from domain.models import (
    BrokerPositionSnapshot,
    OrderIntent,
    SubmitResult,
)
from execution.broker import BaseBroker
from risk.risk_engine import RiskEngine, RiskLimits
from runtime.runtime_engine import (
    JsonStateStore,
    ModelPolicy,
    Mt5CursorTickSource,
    ProcessResult,
    RuntimeEngine,
    RuntimeSnapshot,
    VolumeBarBuilder,
)

from feature_engine import FEATURE_COLS, FeatureEngine, WARMUP_BARS
from mt5_broker_caps import describe_trade_mode, read_symbol_caps, trade_mode_allows_open
from project_paths import resolve_dataset_path, resolve_manifest_path, validate_dataset_bar_spec
from run_logging import configure_run_logging, set_log_context
from runtime_common import STATE_FEATURE_COUNT, ActionSpec, ActionType, build_simple_action_map, deserialize_action_map
from runtime_common import TRAINING_RUNTIME_OPTION_KEYS, runtime_options_from_training_payload
from selector_manifest import load_selector_manifest, validate_paper_live_candidate_manifest
from strategies.rule_logic import compute_rule_direction
from symbol_utils import pip_size_for_symbol
from trading_config import (
    ACTION_SL_MULTS,
    ACTION_TP_MULTS,
    deployment_paths,
    live_enforce_deployment_gate,
    resolve_bar_construction_ticks_per_bar,
)
from validation_metrics import load_json_report

load_dotenv()

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


log = logging.getLogger("live_bridge")


SYMBOL = os.environ.get("TRADING_SYMBOL", "EURUSD").upper()
TICKS_PER_BAR = resolve_bar_construction_ticks_per_bar("BAR_SPEC_TICKS_PER_BAR", "TRADING_TICKS_PER_BAR")
STATE_PATH = os.environ.get("LIVE_STATE_PATH", "live_state.json")
POLL_INTERVAL_MS = int(os.environ.get("LIVE_POLL_INTERVAL_MS", "50"))
ORDER_MAGIC = int(os.environ.get("TRADING_ORDER_MAGIC", "123456"))
MAX_ORDER_DEVIATION_PIPS = float(os.environ.get("TRADING_MAX_ORDER_DEVIATION_PIPS", "2.0"))
LIVE_KILL_SWITCH_PATH = os.environ.get("LIVE_KILL_SWITCH_PATH", "live.kill")
LIVE_SHADOW_MODE = os.environ.get("LIVE_SHADOW_MODE", "0") == "1"
LIVE_STATE_FLUSH_EVERY_TICKS = int(os.environ.get("LIVE_STATE_FLUSH_EVERY_TICKS", "250"))
MAX_DRAWDOWN_FRACTION = float(os.environ.get("TRADING_MAX_DRAWDOWN_FRACTION", "0.15"))


class ManifestRulePolicy:
    def __init__(
        self,
        feature_engine: FeatureEngine,
        action_map: list[ActionSpec] | tuple[ActionSpec, ...],
        *,
        rule_family: str,
        rule_params: dict[str, Any],
    ) -> None:
        self.feature_engine = feature_engine
        self.action_map = list(action_map)
        self.rule_family = str(rule_family)
        self.rule_params = dict(rule_params)
        self.hold_idx = 0
        self.close_idx = 1
        self.long_idx = 2
        self.short_idx = 3

    def decide(self, observation, mask) -> tuple[int, ActionSpec]:
        buffer = getattr(self.feature_engine, "_buffer", None)
        if buffer is None or len(buffer) == 0:
            idx = self.hold_idx if mask[self.hold_idx] else self.close_idx
            return int(idx), self.action_map[int(idx)]
        last_row = buffer.iloc[-1]
        target_direction = int(compute_rule_direction(self.rule_family, last_row.to_dict(), self.rule_params))
        current_direction = int(observation[-1, -4]) if getattr(observation, "ndim", 1) > 1 else int(observation[-4])

        if current_direction == target_direction:
            idx = self.hold_idx
        elif target_direction == 0:
            idx = self.close_idx if current_direction != 0 else self.hold_idx
        elif current_direction == 0:
            idx = self.long_idx if target_direction > 0 else self.short_idx
        else:
            idx = self.close_idx

        if not mask[idx]:
            idx = self.hold_idx if mask[self.hold_idx] else self.close_idx
        return int(idx), self.action_map[int(idx)]


def _resolve_execution_cost_profile(manifest) -> dict[str, float]:
    # Preferred source is the selector manifest's cost_model.
    # Keep backward-compatible fallback to legacy execution_cost_profile if present.
    cost_model = dict(getattr(manifest, "cost_model", None) or {})
    legacy_profile = dict(getattr(manifest, "execution_cost_profile", None) or {})
    profile = cost_model or legacy_profile
    return {
        "commission_per_lot": float(profile.get("commission_per_lot", 7.0)),
        "slippage_pips": float(profile.get("slippage_pips", 0.25)),
        "partial_fill_ratio": float(profile.get("partial_fill_ratio", 1.0)),
    }


def _describe_execution_cost_profile(manifest) -> tuple[dict[str, float], dict[str, str]]:
    cost_model = dict(getattr(manifest, "cost_model", None) or {})
    legacy_profile = dict(getattr(manifest, "execution_cost_profile", None) or {})
    profile = cost_model or legacy_profile
    source_label = "manifest.cost_model" if cost_model else ("manifest.execution_cost_profile" if legacy_profile else "default")

    def _source_for(key: str, default_source: str) -> str:
        if key in cost_model:
            return "manifest.cost_model"
        if key in legacy_profile:
            return "manifest.execution_cost_profile"
        return default_source

    resolved = {
        "commission_per_lot": float(profile.get("commission_per_lot", 7.0)),
        "slippage_pips": float(profile.get("slippage_pips", 0.25)),
        "partial_fill_ratio": float(profile.get("partial_fill_ratio", 1.0)),
    }
    sources = {
        "commission_per_lot": _source_for("commission_per_lot", "default(7.0)"),
        "slippage_pips": _source_for("slippage_pips", "default(0.25)"),
        "partial_fill_ratio": _source_for("partial_fill_ratio", "default(1.0)"),
        "_profile_selected": source_label,
    }
    return resolved, sources


def _resolve_reward_profile(manifest) -> dict[str, float]:
    profile = dict(getattr(manifest, "reward_profile", None) or {})
    return {
        "reward_scale": float(profile.get("reward_scale", 10_000.0)),
        "drawdown_penalty": float(profile.get("drawdown_penalty", 2.0)),
        "transaction_penalty": float(profile.get("transaction_penalty", 1.0)),
        "reward_clip_low": float(profile.get("reward_clip_low", -5.0)),
        "reward_clip_high": float(profile.get("reward_clip_high", 5.0)),
    }


def _load_training_runtime_options(diagnostics_path: Path | None, *, default_window_size: int = 1) -> dict[str, Any]:
    payload = load_json_report(diagnostics_path) if diagnostics_path is not None and diagnostics_path.exists() else {}
    return runtime_options_from_training_payload(payload, default_window_size=default_window_size)


def _selector_action_map(manifest: Any) -> tuple[ActionSpec, ...]:
    rule_params = dict(getattr(manifest, "rule_params", {}) or {})
    return build_simple_action_map(
        sl_value=float(rule_params.get("sl_value", 1.5)),
        tp_value=float(rule_params.get("tp_value", 3.0)),
    )


DAILY_LOSS_FRACTION = float(os.environ.get("TRADING_DAILY_LOSS_FRACTION", "0.05"))
STALE_FEED_MS = int(os.environ.get("TRADING_STALE_FEED_MS", "30000"))
MAX_BROKER_FAILURES = int(os.environ.get("TRADING_MAX_BROKER_FAILURES", "3"))
RISK_PER_TRADE_FRACTION = float(os.environ.get("TRADING_RISK_PER_TRADE_FRACTION", "0.01"))
LOT_SIZE_MIN = float(os.environ.get("TRADING_LOT_SIZE_MIN", "0.01"))
LOT_SIZE_MAX = float(os.environ.get("TRADING_LOT_SIZE_MAX", "0.10"))
LIVE_ALLOW_BAR_SPEC_MISMATCH = os.environ.get("LIVE_ALLOW_BAR_SPEC_MISMATCH", "0") == "1"
LIVE_ALLOW_WARMUP_DATASET_FALLBACK = os.environ.get("LIVE_ALLOW_WARMUP_DATASET_FALLBACK", "0") == "1"
LIVE_ALLOW_FOREIGN_POSITIONS = os.environ.get("LIVE_ALLOW_FOREIGN_POSITIONS", "0") == "1"
LIVE_ALLOW_UNTAGGED_POSITIONS = os.environ.get("LIVE_ALLOW_UNTAGGED_POSITIONS", "0") == "1"
LIVE_FOREX_SESSION_AWARE_STALE_FEED = os.environ.get("LIVE_FOREX_SESSION_AWARE_STALE_FEED", "1") != "0"

ATR_MULT_SL = list(ACTION_SL_MULTS)
ATR_MULT_TP = list(ACTION_TP_MULTS)

_MT5_LOGIN = int(os.environ.get("MT5_LOGIN", "0"))
_MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "")
_MT5_SERVER = os.environ.get("MT5_SERVER", "")


def _connect_mt5(mt5_module: Any) -> None:
    if not mt5_module.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {_mt5_last_error(mt5_module)}")
    if not _MT5_LOGIN or not _MT5_PASSWORD or not _MT5_SERVER:
        raise RuntimeError("MT5 credentials missing. Set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER.")
    if not mt5_module.login(_MT5_LOGIN, _MT5_PASSWORD, _MT5_SERVER):
        raise RuntimeError(f"MT5 login() failed: {_mt5_last_error(mt5_module)}")


def _mt5_last_error(mt5_module: Any) -> str:
    if hasattr(mt5_module, "last_error"):
        try:
            return str(mt5_module.last_error())
        except Exception:
            return "last_error() unavailable"
    return "unknown"


def _safe_mt5_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _account_uses_hedging(mt5_module: Any, account_info: Any) -> bool:
    margin_mode = _safe_mt5_attr(account_info, "margin_mode", None)
    hedging_constant = _safe_mt5_attr(mt5_module, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", None)
    if hedging_constant is None or margin_mode is None:
        return False
    return int(margin_mode) == int(hedging_constant)


def _load_warmup_bars(symbol: str, ticks_per_bar: int) -> pd.DataFrame:
    pair_path = Path("data") / f"{symbol}_volbars_{ticks_per_bar}.csv"
    if pair_path.exists():
        frame = pd.read_csv(pair_path, parse_dates=["Gmt time"])
    else:
        if not LIVE_ALLOW_WARMUP_DATASET_FALLBACK:
            raise RuntimeError(
                f"Missing warmup bars for exact bar spec: {pair_path}. "
                "Refusing to warm the live feature engine on a potentially mismatched dataset. "
                "Set LIVE_ALLOW_WARMUP_DATASET_FALLBACK=1 only if you have separately verified bar-spec parity."
            )
        dataset_path = resolve_dataset_path()
        validate_dataset_bar_spec(
            dataset_path=dataset_path,
            expected_ticks_per_bar=ticks_per_bar,
            metadata_required=True,
        )
        frame = pd.read_csv(dataset_path, low_memory=False, parse_dates=["Gmt time"])
        if "Symbol" in frame.columns:
            frame = frame[frame["Symbol"].astype(str).str.upper() == symbol.upper()].copy()
    if frame.empty:
        raise RuntimeError(f"No warmup bars available for {symbol}.")
    frame["Gmt time"] = pd.to_datetime(frame["Gmt time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["Gmt time"]).set_index("Gmt time").sort_index()
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Warmup data missing required columns: {missing}")
    if "avg_spread" not in frame.columns:
        frame["avg_spread"] = 0.0
    if "time_delta_s" not in frame.columns:
        frame["time_delta_s"] = frame.index.to_series().diff().dt.total_seconds().fillna(0.0)
    if len(frame) < WARMUP_BARS:
        raise RuntimeError(f"Need at least {WARMUP_BARS} warmup bars for {symbol}, got {len(frame)}.")
    return frame.iloc[-WARMUP_BARS:].copy()


def _build_risk_limits() -> RiskLimits:
    return RiskLimits(
        max_drawdown_fraction=MAX_DRAWDOWN_FRACTION,
        daily_loss_fraction=DAILY_LOSS_FRACTION,
        stale_feed_ms=STALE_FEED_MS,
        max_broker_failures=MAX_BROKER_FAILURES,
        risk_per_trade_fraction=RISK_PER_TRADE_FRACTION,
        lot_size_min=LOT_SIZE_MIN,
        lot_size_max=LOT_SIZE_MAX,
        safe_mode_on_kill=True,
    )


def _likely_forex_market_closed(now_utc: datetime) -> bool:
    timestamp = pd.Timestamp(now_utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    weekday = int(timestamp.weekday())
    hour_fraction = float(timestamp.hour) + (float(timestamp.minute) / 60.0)
    if weekday == 5:
        return True
    if weekday == 6 and hour_fraction < 22.0:
        return True
    if weekday == 4 and hour_fraction >= 22.0:
        return True
    return False


def _live_readiness_blockers(symbol: str) -> list[str]:
    if not live_enforce_deployment_gate():
        return []
    paths = deployment_paths(symbol)
    blockers: list[str] = []

    if not paths.gate_path.exists():
        blockers.append(f"Deployment gate missing: {paths.gate_path}")
    else:
        gate = load_json_report(paths.gate_path)
        if not bool(gate.get("approved_for_live", False)):
            blockers.append("Deployment gate is not approved for live trading.")
            blockers.extend(f"Gate blocker: {blocker}" for blocker in gate.get("blockers", []))

    if not paths.live_preflight_path.exists():
        blockers.append(f"Live preflight missing: {paths.live_preflight_path}")
    else:
        preflight = load_json_report(paths.live_preflight_path)
        if not bool(preflight.get("approved_for_live_runtime", False)):
            blockers.append("Live preflight is not approved.")
            blockers.extend(f"Preflight blocker: {blocker}" for blocker in preflight.get("blockers", []))

    if not paths.ops_attestation_path.exists():
        blockers.append(f"Ops attestation missing: {paths.ops_attestation_path}")
    else:
        attestation = load_json_report(paths.ops_attestation_path)
        if not bool(attestation.get("approved", False)):
            blockers.append("Ops attestation is not approved.")
            blockers.extend(f"Ops blocker: {blocker}" for blocker in attestation.get("blockers", []))

    return blockers


def _emergency_flatten(runtime: RuntimeEngine, reason: str) -> SubmitResult | None:
    position = runtime.broker.current_position(runtime.symbol)
    if position.direction == 0 or position.broker_ticket is None:
        return None
    intent = OrderIntent(
        symbol=runtime.symbol,
        action=ActionSpec(ActionType.CLOSE),
        volume=float(position.volume),
        submitted_time_msc=int(datetime.now(tz=timezone.utc).timestamp() * 1000),
        requested_price=float(position.entry_price or 0.0),
        broker_ticket=position.broker_ticket,
    )
    result = runtime.broker.submit_order(intent)
    if result.accepted:
        runtime.startup_reconcile()
    log.critical("Emergency flatten %s: accepted=%s error=%s", reason, result.accepted, result.error)
    return result


class LiveMt5Broker(BaseBroker):
    def __init__(
        self,
        mt5_module: Any,
        *,
        symbol: str,
        commission_per_lot: float = 7.0,
        slippage_pips: float = 0.25,
        partial_fill_ratio: float = 1.0,
    ) -> None:
        self.mt5 = mt5_module
        self.symbol = symbol.upper()
        self.paths = deployment_paths(self.symbol)
        self.commission_per_lot = float(commission_per_lot)
        self.slippage_pips = float(slippage_pips)
        self.partial_fill_ratio = float(partial_fill_ratio)

    def _raw_positions(self, symbol: str) -> list[Any]:
        return list(self.mt5.positions_get(symbol=symbol) or [])

    def _classify_positions(self, symbol: str) -> tuple[list[Any], list[Any]]:
        positions = self._raw_positions(symbol)
        if not positions:
            return [], []
        if not all(hasattr(position, "magic") for position in positions):
            if LIVE_ALLOW_UNTAGGED_POSITIONS:
                return positions, []
            raise RuntimeError(
                f"Broker positions for {symbol} do not expose magic. "
                "Refusing to reconcile an untagged account. Set LIVE_ALLOW_UNTAGGED_POSITIONS=1 to override."
            )
        strategy = [
            position
            for position in positions
            if int(getattr(position, "magic", 0) or 0) == ORDER_MAGIC
        ]
        foreign = [
            position
            for position in positions
            if int(getattr(position, "magic", 0) or 0) != ORDER_MAGIC
        ]
        return strategy, foreign

    def isolation_blocker(self, symbol: str) -> str | None:
        try:
            _strategy, foreign = self._classify_positions(symbol)
        except RuntimeError as exc:
            return str(exc)
        if foreign and not LIVE_ALLOW_FOREIGN_POSITIONS:
            return (
                f"Found {len(foreign)} non-strategy position(s) for {symbol}. "
                "Account-level equity and risk limits would be contaminated. "
                "Use an isolated account or set LIVE_ALLOW_FOREIGN_POSITIONS=1 only with full awareness of the risk."
            )
        return None

    def _strategy_positions(self, symbol: str) -> list[Any]:
        strategy, _foreign = self._classify_positions(symbol)
        return strategy

    def _aggregate_strategy_position(self, symbol: str) -> BrokerPositionSnapshot:
        positions = self._strategy_positions(symbol)
        if not positions:
            return BrokerPositionSnapshot(symbol=symbol.upper())

        directions = [1 if int(position.type) == 0 else -1 for position in positions]
        if len(set(directions)) > 1:
            raise RuntimeError(
                f"Mixed-direction strategy positions found for {symbol}. "
                "Manual intervention is required before the runtime can continue safely."
            )

        direction = directions[0]
        total_volume = float(sum(float(position.volume) for position in positions))
        if total_volume <= 0:
            return BrokerPositionSnapshot(symbol=symbol.upper())
        weighted_entry = sum(float(position.volume) * float(position.price_open) for position in positions) / total_volume

        sl_values = {round(float(getattr(position, "sl", 0.0) or 0.0), 10) for position in positions}
        tp_values = {round(float(getattr(position, "tp", 0.0) or 0.0), 10) for position in positions}
        aggregate_sl = None if len(sl_values) != 1 or next(iter(sl_values)) == 0.0 else next(iter(sl_values))
        aggregate_tp = None if len(tp_values) != 1 or next(iter(tp_values)) == 0.0 else next(iter(tp_values))
        primary = max(positions, key=lambda position: int(getattr(position, "time_msc", getattr(position, "time", 0)) or 0))
        return BrokerPositionSnapshot(
            symbol=symbol.upper(),
            direction=direction,
            volume=total_volume,
            entry_price=float(weighted_entry),
            sl_price=float(aggregate_sl) if aggregate_sl is not None else None,
            tp_price=float(aggregate_tp) if aggregate_tp is not None else None,
            broker_ticket=int(getattr(primary, "ticket", 0) or 0),
            order_id=int(getattr(primary, "identifier", 0) or 0),
            last_confirmed_time_msc=int(getattr(primary, "time_msc", getattr(primary, "time", 0)) or 0),
        )

    def _normalize_volume(self, symbol_info: Any, requested_volume: float) -> float:
        caps = read_symbol_caps(self.symbol, symbol_info)
        return caps.normalize_volume(requested_volume)

    def _normalize_price(self, symbol_info: Any, price: float | None) -> float | None:
        caps = read_symbol_caps(self.symbol, symbol_info)
        return caps.normalize_price(price)

    def _trade_permissions_blocker(
        self,
        *,
        symbol: str,
        opening: bool,
        direction: int | None = None,
        symbol_info: Any | None = None,
    ) -> str | None:
        if hasattr(self.mt5, "terminal_info"):
            terminal_info = self.mt5.terminal_info()
            if terminal_info is not None and hasattr(terminal_info, "trade_allowed") and not bool(getattr(terminal_info, "trade_allowed")):
                return "MT5 terminal has trading disabled."
        if hasattr(self.mt5, "account_info"):
            account_info = self.mt5.account_info()
            if account_info is not None and hasattr(account_info, "trade_allowed") and not bool(getattr(account_info, "trade_allowed")):
                return "MT5 account reports trading disabled."
        symbol_info = symbol_info if symbol_info is not None else self.mt5.symbol_info(symbol)
        if symbol_info is None:
            return f"symbol_info returned None for {symbol}"
        caps = read_symbol_caps(symbol, symbol_info)
        if opening and not trade_mode_allows_open(self.mt5, caps.trade_mode, direction):
            return f"Broker symbol trade mode does not allow this open request ({describe_trade_mode(self.mt5, caps.trade_mode)})."
        return None

    def _stops_valid(
        self,
        symbol_info: Any,
        price: float,
        sl_price: float | None,
        tp_price: float | None,
        *,
        direction: int | None = None,
    ) -> tuple[bool, str | None]:
        caps = read_symbol_caps(self.symbol, symbol_info)
        min_distance = caps.minimum_stop_distance
        if direction is not None:
            if direction > 0:
                if sl_price is not None and float(sl_price) >= float(price):
                    return False, "SL must be below entry price for long positions."
                if tp_price is not None and float(tp_price) <= float(price):
                    return False, "TP must be above entry price for long positions."
            if direction < 0:
                if sl_price is not None and float(sl_price) <= float(price):
                    return False, "SL must be above entry price for short positions."
                if tp_price is not None and float(tp_price) >= float(price):
                    return False, "TP must be below entry price for short positions."
        for label, target in (("sl", sl_price), ("tp", tp_price)):
            if target is None:
                continue
            if abs(float(price) - float(target)) < min_distance:
                return False, f"{label.upper()} violates broker stops/freeze level (min distance {min_distance:.8f})."
        return True, None

    def _audit_order(
        self,
        *,
        intent: OrderIntent,
        request: dict[str, Any],
        result: SubmitResult,
        symbol_info: Any,
        tick_info: Any | None = None,
    ) -> None:
        point = float(_safe_mt5_attr(symbol_info, "point", 0.0) or 0.0)
        requested_price = float(intent.requested_price or 0.0)
        sent_price = float(request.get("price", 0.0) or 0.0)
        fill_price = float(result.fill_price or 0.0)
        fill_delta_price_requested = fill_price - requested_price if fill_price else 0.0
        fill_delta_price_sent = fill_price - sent_price if fill_price else 0.0
        # Prefer measuring drift vs the transmitted price (sent_price). requested_price may be a
        # higher-level intent (or a legacy placeholder) and can inflate drift metrics.
        drift_basis = "sent_price" if sent_price else "requested_price"
        fill_delta_price = fill_delta_price_sent if sent_price else fill_delta_price_requested
        pip_size = pip_size_for_symbol(intent.symbol)
        fill_delta_pips = fill_delta_price / pip_size if pip_size else 0.0
        fill_delta_pips_requested = fill_delta_price_requested / pip_size if pip_size else 0.0
        fill_delta_pips_sent = fill_delta_price_sent / pip_size if pip_size else 0.0
        bid = _safe_mt5_attr(tick_info, "bid", None) if tick_info is not None else None
        ask = _safe_mt5_attr(tick_info, "ask", None) if tick_info is not None else None
        spread_price = None
        spread_pips = None
        if bid is not None and ask is not None:
            spread_price = float(ask) - float(bid)
            spread_pips = float(spread_price / pip_size) if pip_size else 0.0
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": intent.symbol,
            "action": intent.action.action_type.value,
            "direction": intent.action.direction,
            "requested_price": requested_price,
            "sent_price": sent_price,
            "fill_price": fill_price,
            "fill_delta_basis": drift_basis,
            "fill_delta_price": fill_delta_price,
            "fill_delta_pips": fill_delta_pips,
            "fill_delta_price_requested": fill_delta_price_requested,
            "fill_delta_pips_requested": fill_delta_pips_requested,
            "fill_delta_price_sent": fill_delta_price_sent,
            "fill_delta_pips_sent": fill_delta_pips_sent,
            "volume": float(request.get("volume", 0.0) or 0.0),
            "sl_price": request.get("sl"),
            "tp_price": request.get("tp"),
            "broker_ticket": intent.broker_ticket,
            "retcode": result.retcode,
            "accepted": bool(result.accepted),
            "error": result.error,
            "shadow_mode": bool(LIVE_SHADOW_MODE),
            "order_id": result.order_id,
            "request_magic": request.get("magic"),
            "request_comment": request.get("comment"),
            "request_deviation_points": request.get("deviation"),
            "request_type_filling": request.get("type_filling"),
            "request_type_time": request.get("type_time"),
            "tick_bid": bid,
            "tick_ask": ask,
            "tick_spread_price": spread_price,
            "tick_spread_pips": spread_pips,
            "broker_point": point,
            "broker_tick_size": _safe_mt5_attr(symbol_info, "trade_tick_size"),
            "broker_tick_value": _safe_mt5_attr(symbol_info, "trade_tick_value"),
            "broker_contract_size": _safe_mt5_attr(symbol_info, "trade_contract_size"),
            "broker_stops_level_points": _safe_mt5_attr(symbol_info, "trade_stops_level"),
            "broker_freeze_level_points": _safe_mt5_attr(symbol_info, "trade_freeze_level"),
        }
        _append_jsonl(self.paths.execution_audit_path, payload)

    def _send_request(
        self,
        *,
        intent: OrderIntent,
        request: dict[str, Any],
        symbol_info: Any,
    ) -> SubmitResult:
        tick_info = None
        if hasattr(self.mt5, "symbol_info_tick"):
            try:
                tick_info = self.mt5.symbol_info_tick(intent.symbol)
            except Exception:
                tick_info = None
        if LIVE_SHADOW_MODE:
            result = SubmitResult(
                accepted=True,
                order_id=0,
                retcode=0,
                fill_price=float(request.get("price", 0.0) or 0.0),
            )
            self._audit_order(intent=intent, request=request, result=result, symbol_info=symbol_info, tick_info=tick_info)
            return result

        result = self.mt5.order_send(request)
        if result is not None and result.retcode == 10030:
            request["type_filling"] = self.mt5.ORDER_FILLING_FOK
            result = self.mt5.order_send(request)
        if result is not None and result.retcode == 10030 and hasattr(self.mt5, "ORDER_FILLING_RETURN"):
            request["type_filling"] = self.mt5.ORDER_FILLING_RETURN
            result = self.mt5.order_send(request)
        if result is None:
            submit_result = SubmitResult(accepted=False, error="order_send returned None")
            self._audit_order(intent=intent, request=request, result=submit_result, symbol_info=symbol_info, tick_info=tick_info)
            return submit_result
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            submit_result = SubmitResult(
                accepted=False,
                error=f"retcode={result.retcode}",
                retcode=int(result.retcode),
                fill_price=float(getattr(result, "price", 0.0) or 0.0),
            )
            self._audit_order(intent=intent, request=request, result=submit_result, symbol_info=symbol_info, tick_info=tick_info)
            return submit_result
        submit_result = SubmitResult(
            accepted=True,
            order_id=int(getattr(result, "order", 0) or 0),
            retcode=int(getattr(result, "retcode", 0) or 0),
            fill_price=float(getattr(result, "price", 0.0) or 0.0),
        )
        self._audit_order(intent=intent, request=request, result=submit_result, symbol_info=symbol_info, tick_info=tick_info)
        return submit_result

    def submit_order(self, intent: OrderIntent) -> SubmitResult:
        if intent.action.action_type == ActionType.OPEN and Path(LIVE_KILL_SWITCH_PATH).exists():
            return SubmitResult(accepted=False, error="Manual kill switch is active.")

        tick = self.mt5.symbol_info_tick(intent.symbol)
        if tick is None:
            return SubmitResult(accepted=False, error="symbol_info_tick returned None")
        symbol_info = self.mt5.symbol_info(intent.symbol)
        if symbol_info is None:
            return SubmitResult(accepted=False, error=f"symbol_info returned None for {intent.symbol}")
        if not getattr(symbol_info, "visible", True) and hasattr(self.mt5, "symbol_select"):
            if not self.mt5.symbol_select(intent.symbol, True):
                return SubmitResult(accepted=False, error=f"symbol_select failed for {intent.symbol}")
            symbol_info = self.mt5.symbol_info(intent.symbol)
            if symbol_info is None:
                return SubmitResult(accepted=False, error=f"symbol_info returned None for {intent.symbol} after symbol_select")
        permission_blocker = self._trade_permissions_blocker(
            symbol=intent.symbol,
            opening=intent.action.action_type == ActionType.OPEN,
            direction=intent.action.direction,
            symbol_info=symbol_info,
        )
        if permission_blocker is not None:
            return SubmitResult(accepted=False, error=permission_blocker)
        point = float(getattr(symbol_info, "point", 0.0) or 0.0)
        deviation_points = 0
        if point > 0:
            pip_size = pip_size_for_symbol(intent.symbol)
            deviation_points = max(1, int(round((MAX_ORDER_DEVIATION_PIPS * pip_size) / point)))

        if intent.action.action_type == ActionType.CLOSE:
            positions = self._strategy_positions(intent.symbol)
            if not positions:
                return SubmitResult(accepted=False, error="No broker position to close.")
            results: list[SubmitResult] = []
            errors: list[str] = []
            for position_row in positions:
                position_direction = 1 if int(position_row.type) == 0 else -1
                order_type = self.mt5.ORDER_TYPE_SELL if position_direction > 0 else self.mt5.ORDER_TYPE_BUY
                price = self._normalize_price(symbol_info, tick.bid if position_direction > 0 else tick.ask)
                volume = self._normalize_volume(symbol_info, float(position_row.volume))
                request = {
                    "action": self.mt5.TRADE_ACTION_DEAL,
                    "symbol": intent.symbol,
                    "volume": volume,
                    "type": order_type,
                    "position": int(getattr(position_row, "ticket", 0) or 0),
                    "price": price,
                    "magic": ORDER_MAGIC,
                    "comment": "Bot close",
                    "type_time": self.mt5.ORDER_TIME_GTC,
                    "type_filling": self.mt5.ORDER_FILLING_IOC,
                    "deviation": deviation_points,
                }
                result = self._send_request(intent=intent, request=request, symbol_info=symbol_info)
                results.append(result)
                if not result.accepted:
                    errors.append(result.error or "close failed")
            if errors:
                return SubmitResult(
                    accepted=False,
                    error="; ".join(errors),
                    retcode=next((result.retcode for result in results if result.retcode is not None), None),
                    fill_price=next((result.fill_price for result in results if result.fill_price is not None), None),
                )
            return SubmitResult(
                accepted=True,
                order_id=results[-1].order_id if results else None,
                retcode=results[-1].retcode if results else None,
                fill_price=results[-1].fill_price if results else None,
            )
        else:
            if intent.action.direction is None:
                return SubmitResult(accepted=False, error="OPEN intent missing direction.")
            order_type = self.mt5.ORDER_TYPE_BUY if intent.action.direction > 0 else self.mt5.ORDER_TYPE_SELL
            price = self._normalize_price(symbol_info, tick.ask if intent.action.direction > 0 else tick.bid)
            normalized_volume = self._normalize_volume(symbol_info, intent.volume)
            normalized_sl = self._normalize_price(symbol_info, intent.sl_price)
            normalized_tp = self._normalize_price(symbol_info, intent.tp_price)
            stops_ok, stops_error = self._stops_valid(
                symbol_info,
                float(price),
                normalized_sl,
                normalized_tp,
                direction=intent.action.direction,
            )
            if not stops_ok:
                result = SubmitResult(accepted=False, error=stops_error)
                self._audit_order(
                    intent=intent,
                    request={"price": price, "volume": normalized_volume, "sl": normalized_sl, "tp": normalized_tp},
                    result=result,
                    symbol_info=symbol_info,
                )
                return result
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": intent.symbol,
                "volume": normalized_volume,
                "type": order_type,
                "price": price,
                "sl": normalized_sl,
                "tp": normalized_tp,
                "magic": ORDER_MAGIC,
                "comment": "Bot open",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
                "deviation": deviation_points,
            }
        if LIVE_SHADOW_MODE:
            log.warning("SHADOW MODE: suppressed order_send for %s %s", intent.symbol, intent.action.action_type.value)
        return self._send_request(intent=intent, request=request, symbol_info=symbol_info)

    def current_position(self, symbol: str) -> BrokerPositionSnapshot:
        return self._aggregate_strategy_position(symbol)

    def current_equity(
        self,
        symbol: str,
        mark_price: float | None = None,
        *,
        avg_spread: float = 0.0,
        mark_to_liquidation: bool = True,
    ) -> float:
        isolation_blocker = self.isolation_blocker(symbol)
        if isolation_blocker is not None:
            raise RuntimeError(isolation_blocker)
        account = self.mt5.account_info()
        if account is None:
            raise RuntimeError("MT5 account_info() returned None.")
        return float(account.equity)


def bootstrap_live_runtime(
    *,
    symbol: str = SYMBOL,
    state_path: str = STATE_PATH,
    ticks_per_bar: int = TICKS_PER_BAR,
    manifest_path: str | Path | None = None,
    mt5_module: Any | None = None,
) -> tuple[RuntimeEngine, VolumeBarBuilder, JsonStateStore, Mt5CursorTickSource]:
    symbol = symbol.upper()
    if manifest_path is not None:
        selector_manifest = load_selector_manifest(
            manifest_path,
            verify_manifest_hash=True,
            strict_manifest_hash=True,
            require_component_hashes=True,
        )
        if selector_manifest.engine_type == "RULE":
            validate_paper_live_candidate_manifest(selector_manifest)
            return _bootstrap_rule_live_runtime(
                manifest=selector_manifest,
                manifest_path=Path(manifest_path),
                symbol=symbol,
                state_path=state_path,
                ticks_per_bar=ticks_per_bar,
                mt5_module=mt5_module,
            )

    dataset_path = resolve_dataset_path()
    resolved_manifest_path = resolve_manifest_path(symbol=symbol, preferred=manifest_path)
    manifest = load_manifest(resolved_manifest_path)
    execution_cost_profile, execution_cost_sources = _describe_execution_cost_profile(manifest)
    reward_profile = _resolve_reward_profile(manifest)
    log.info(
        "Runtime cost profile resolved=%s sources=%s",
        execution_cost_profile,
        execution_cost_sources,
    )
    manifest_ticks = manifest.bar_construction_ticks_per_bar or manifest.ticks_per_bar
    if manifest_ticks is not None and Path(dataset_path).exists():
        validate_dataset_bar_spec(
            dataset_path=dataset_path,
            expected_ticks_per_bar=int(manifest_ticks),
            metadata_required=True,
        )
    elif manifest_ticks is not None:
        log.warning(
            "Dataset path %s does not exist during bootstrap; skipping dataset bar-spec validation.",
            dataset_path,
        )
    if manifest_ticks is not None and int(manifest_ticks) != int(ticks_per_bar) and not LIVE_ALLOW_BAR_SPEC_MISMATCH:
        raise RuntimeError(
            f"Manifest bar_construction_ticks_per_bar={manifest_ticks} does not match live bar_construction_ticks_per_bar={ticks_per_bar}. "
            "Retrain/rebuild manifests or set LIVE_ALLOW_BAR_SPEC_MISMATCH=1 for an explicit override."
        )
    dataset_id = dataset_id_for_path(dataset_path)
    diagnostics_path = Path(manifest.training_diagnostics_path) if manifest.training_diagnostics_path else None

    action_map = deserialize_action_map(manifest.action_map)
    observation_shape = list(getattr(manifest, "observation_shape", None) or [1, len(FEATURE_COLS) + STATE_FEATURE_COUNT])
    model = load_validated_model(
        manifest,
        expected_symbol=symbol,
        expected_action_map=action_map,
        expected_observation_shape=observation_shape,
        expected_dataset_id=dataset_id,
    )
    obs_normalizer = load_validated_vecnormalize(
        manifest,
        expected_symbol=symbol,
        expected_action_map=action_map,
        expected_observation_shape=observation_shape,
        expected_dataset_id=dataset_id,
    )
    scaler = load_validated_scaler(
        manifest,
        expected_symbol=symbol,
        expected_action_map=action_map,
        expected_observation_shape=observation_shape,
        expected_dataset_id=dataset_id,
    )
    runtime_options = _load_training_runtime_options(
        diagnostics_path,
        default_window_size=int(observation_shape[0] if observation_shape else 1),
    )
    if diagnostics_path is None:
        log.warning(
            "Training diagnostics path missing in manifest for %s; using default runtime guard settings.",
            symbol,
        )
    elif not diagnostics_path.exists():
        log.warning(
            "Training diagnostics file missing for %s at %s; using default runtime guard settings.",
            symbol,
            diagnostics_path,
        )
    else:
        diagnostics_payload = load_json_report(diagnostics_path)
        missing_keys = [key for key in TRAINING_RUNTIME_OPTION_KEYS if key not in diagnostics_payload]
        if missing_keys:
            log.warning(
                "Training diagnostics for %s are incomplete; default runtime guard values used for keys: %s",
                symbol,
                ", ".join(missing_keys),
            )
    feature_engine = FeatureEngine.from_scaler(scaler)  # validated above
    feature_engine.warm_up(_load_warmup_bars(symbol, ticks_per_bar))

    if not MT5_AVAILABLE and mt5_module is None:
        raise RuntimeError("MetaTrader5 is not available in this environment.")
    broker_module = mt5_module or mt5
    _connect_mt5(broker_module)
    broker = LiveMt5Broker(
        broker_module,
        symbol=symbol,
        commission_per_lot=execution_cost_profile["commission_per_lot"],
        slippage_pips=execution_cost_profile["slippage_pips"],
        partial_fill_ratio=execution_cost_profile["partial_fill_ratio"],
    )
    account_info = broker_module.account_info()
    if account_info is None:
        raise RuntimeError("MT5 account_info() returned None during live bootstrap.")
    if account_info is not None and _account_uses_hedging(broker_module, account_info):
        raise RuntimeError(
            "MT5 account is in hedging mode. Deployment requires a netting account; "
            "hedging accounts are not a supported live runtime target."
        )
    initial_equity = float(getattr(account_info, "equity", 0.0) or 0.0)

    state_store = JsonStateStore(state_path, ticks_per_bar=ticks_per_bar)
    snapshot = state_store.load()
    blockers = _live_readiness_blockers(symbol)
    isolation_blocker = broker.isolation_blocker(symbol)
    if isolation_blocker is not None:
        blockers.append(isolation_blocker)
    if blockers:
        snapshot.safe_mode_active = True
        log.critical("Live readiness blockers present for %s. Starting in safe mode.", symbol)
        for blocker in blockers:
            log.critical("LIVE BLOCKER: %s", blocker)
    if snapshot.bar_builder.ticks_per_bar == 0:
        snapshot.bar_builder.ticks_per_bar = ticks_per_bar
    builder = VolumeBarBuilder(ticks_per_bar=ticks_per_bar, state=snapshot.bar_builder)
    risk_engine = RiskEngine(_build_risk_limits(), snapshot=snapshot, initial_equity=initial_equity)
    runtime = RuntimeEngine(
        symbol=symbol,
        feature_engine=feature_engine,
        policy=ModelPolicy(model, action_map, obs_normalizer=obs_normalizer),
        broker=broker,
        action_map=action_map,
        risk_engine=risk_engine,
        state_store=state_store,
        snapshot=snapshot,
        reward_scale=reward_profile["reward_scale"],
        reward_drawdown_penalty=reward_profile["drawdown_penalty"],
        reward_transaction_penalty=reward_profile["transaction_penalty"],
        reward_clip_low=reward_profile["reward_clip_low"],
        reward_clip_high=reward_profile["reward_clip_high"],
        window_size=int(runtime_options.get("window_size", observation_shape[0] if observation_shape else 1)),
        churn_min_hold_bars=int(runtime_options.get("churn_min_hold_bars", 0)),
        churn_action_cooldown=int(runtime_options.get("churn_action_cooldown", 0)),
        entry_spread_z_limit=float(runtime_options.get("entry_spread_z_limit", 1.5)),
    )
    runtime.startup_reconcile()
    source = Mt5CursorTickSource(broker_module)
    log.info(f"Live runtime ready using manifest {resolved_manifest_path}")
    return runtime, builder, state_store, source


def _bootstrap_rule_live_runtime(
    *,
    manifest: Any,
    manifest_path: Path,
    symbol: str,
    state_path: str,
    ticks_per_bar: int,
    mt5_module: Any | None,
) -> tuple[RuntimeEngine, VolumeBarBuilder, JsonStateStore, Mt5CursorTickSource]:
    resolved_symbol = str(getattr(manifest, "strategy_symbol", symbol)).upper()
    if resolved_symbol != symbol.upper():
        raise RuntimeError(f"Selector manifest symbol mismatch: manifest={resolved_symbol} runtime={symbol.upper()}.")

    execution_cost_profile, execution_cost_sources = _describe_execution_cost_profile(manifest)
    reward_profile = _resolve_reward_profile(manifest)
    log.info(
        "Runtime cost profile resolved=%s sources=%s",
        execution_cost_profile,
        execution_cost_sources,
    )
    manifest_ticks = manifest.bar_construction_ticks_per_bar or manifest.ticks_per_bar
    if manifest_ticks is not None and int(manifest_ticks) != int(ticks_per_bar) and not LIVE_ALLOW_BAR_SPEC_MISMATCH:
        raise RuntimeError(
            f"Selector manifest bar_construction_ticks_per_bar={manifest_ticks} does not match live bar_construction_ticks_per_bar={ticks_per_bar}. "
            "Align the runtime bar spec or set LIVE_ALLOW_BAR_SPEC_MISMATCH=1 for an explicit override."
        )

    feature_engine = FeatureEngine()
    feature_engine.warm_up(_load_warmup_bars(resolved_symbol, ticks_per_bar))
    action_map = list(_selector_action_map(manifest))
    policy = ManifestRulePolicy(
        feature_engine,
        action_map,
        rule_family=str(manifest.rule_family or ""),
        rule_params=dict(getattr(manifest, "rule_params", {}) or {}),
    )
    runtime_options = runtime_options_from_training_payload({}, default_window_size=1)

    if not MT5_AVAILABLE and mt5_module is None:
        raise RuntimeError("MetaTrader5 is not available in this environment.")
    broker_module = mt5_module or mt5
    _connect_mt5(broker_module)
    broker = LiveMt5Broker(
        broker_module,
        symbol=resolved_symbol,
        commission_per_lot=execution_cost_profile["commission_per_lot"],
        slippage_pips=execution_cost_profile["slippage_pips"],
        partial_fill_ratio=execution_cost_profile["partial_fill_ratio"],
    )
    account_info = broker_module.account_info()
    if account_info is None:
        raise RuntimeError("MT5 account_info() returned None during selector live bootstrap.")
    if _account_uses_hedging(broker_module, account_info):
        raise RuntimeError(
            "MT5 account is in hedging mode. Deployment requires a netting account; "
            "hedging accounts are not a supported live runtime target."
        )
    initial_equity = float(getattr(account_info, "equity", 0.0) or 0.0)

    state_store = JsonStateStore(state_path, ticks_per_bar=ticks_per_bar)
    snapshot = state_store.load()
    blockers = _live_readiness_blockers(resolved_symbol)
    isolation_blocker = broker.isolation_blocker(resolved_symbol)
    if isolation_blocker is not None:
        blockers.append(isolation_blocker)
    if blockers:
        snapshot.safe_mode_active = True
        log.critical("Live readiness blockers present for %s. Starting in safe mode.", resolved_symbol)
        for blocker in blockers:
            log.critical("LIVE BLOCKER: %s", blocker)
    if snapshot.bar_builder.ticks_per_bar == 0:
        snapshot.bar_builder.ticks_per_bar = ticks_per_bar
    builder = VolumeBarBuilder(ticks_per_bar=ticks_per_bar, state=snapshot.bar_builder)
    risk_engine = RiskEngine(_build_risk_limits(), snapshot=snapshot, initial_equity=initial_equity)
    runtime = RuntimeEngine(
        symbol=resolved_symbol,
        feature_engine=feature_engine,
        policy=policy,
        broker=broker,
        action_map=action_map,
        risk_engine=risk_engine,
        state_store=state_store,
        snapshot=snapshot,
        reward_scale=reward_profile["reward_scale"],
        reward_drawdown_penalty=reward_profile["drawdown_penalty"],
        reward_transaction_penalty=reward_profile["transaction_penalty"],
        reward_clip_low=reward_profile["reward_clip_low"],
        reward_clip_high=reward_profile["reward_clip_high"],
        window_size=int(runtime_options.get("window_size", 1)),
        churn_min_hold_bars=int(runtime_options.get("churn_min_hold_bars", 0)),
        churn_action_cooldown=int(runtime_options.get("churn_action_cooldown", 0)),
        entry_spread_z_limit=float(runtime_options.get("entry_spread_z_limit", 1.5)),
    )
    runtime.policy_mode = "RULE"
    runtime.startup_reconcile()
    source = Mt5CursorTickSource(broker_module)
    log.info(
        "Selector live runtime ready using manifest %s (manifest_hash=%s)",
        manifest_path,
        getattr(manifest, "manifest_hash", ""),
    )
    return runtime, builder, state_store, source


def _persist_live_checkpoint(runtime: RuntimeEngine, builder: VolumeBarBuilder) -> None:
    runtime.snapshot.bar_builder = builder.state
    runtime.persist()


def _log_result(result: ProcessResult) -> None:
    log.info(
        "Bar %s O=%.5f H=%.5f L=%.5f C=%.5f action=%s dir=%s equity=%.2f pos=%s",
        result.bar.timestamp.isoformat(),
        result.bar.open,
        result.bar.high,
        result.bar.low,
        result.bar.close,
        result.action.action_type.value,
        result.action.direction,
        result.equity,
        result.position_direction,
    )
    if result.submit_result is not None and not result.submit_result.accepted:
        log.error("Broker submission failed: %s", result.submit_result.error)


def run_live_loop(
    *,
    symbol: str = SYMBOL,
    state_path: str = STATE_PATH,
    ticks_per_bar: int = TICKS_PER_BAR,
    poll_interval_ms: int = POLL_INTERVAL_MS,
    max_loops: int | None = None,
    mt5_module: Any | None = None,
) -> None:
    runtime, builder, _store, source = bootstrap_live_runtime(
        symbol=symbol,
        state_path=state_path,
        ticks_per_bar=ticks_per_bar,
        mt5_module=mt5_module,
    )
    loops = 0
    broker_module = mt5_module or mt5
    dirty_ticks_since_flush = 0

    try:
        while max_loops is None or loops < max_loops:
            if Path(LIVE_KILL_SWITCH_PATH).exists():
                runtime.risk_engine.trigger_kill_switch("Manual kill switch file detected.")
                _persist_live_checkpoint(runtime, builder)
                _emergency_flatten(runtime, "manual kill switch")
                raise RuntimeError("Manual kill switch active.")

            if isinstance(runtime.broker, LiveMt5Broker):
                isolation_blocker = runtime.broker.isolation_blocker(symbol)
                if isolation_blocker is not None:
                    runtime.risk_engine.trigger_kill_switch(isolation_blocker)
                    _persist_live_checkpoint(runtime, builder)
                    _emergency_flatten(runtime, isolation_blocker)
                    raise RuntimeError(isolation_blocker)

            now_utc = pd.Timestamp.utcnow().to_pydatetime()
            stale_ok, stale_reason = runtime.risk_engine.check_stale_feed(
                now_utc=now_utc,
                last_tick_time_msc=runtime.snapshot.last_tick_time_msc,
            )
            if not stale_ok:
                if LIVE_FOREX_SESSION_AWARE_STALE_FEED and _likely_forex_market_closed(now_utc):
                    log.warning("Ignoring stale feed while forex market is likely closed: %s", stale_reason)
                    time.sleep(poll_interval_ms / 1000.0)
                    loops += 1
                    continue
                _emergency_flatten(runtime, stale_reason)
                raise RuntimeError(stale_reason)

            try:
                ticks, cursor = source.fetch(symbol.upper(), runtime.snapshot.cursor)
            except Exception as exc:
                log.error("Tick fetch failed: %s", exc)
                if broker_module is None:
                    raise
                broker_module.shutdown()
                _connect_mt5(broker_module)
                _emergency_flatten(runtime, "tick fetch failure")
                raise RuntimeError("MT5 reconnect attempted and execution stopped fail-closed.") from exc

            runtime.snapshot.cursor = cursor
            if not ticks:
                time.sleep(poll_interval_ms / 1000.0)
                loops += 1
                continue

            for tick in ticks:
                runtime.snapshot.last_tick_time_msc = tick.time_msc
                bar = builder.push_tick(tick)
                dirty_ticks_since_flush += 1
                if LIVE_STATE_FLUSH_EVERY_TICKS > 0 and dirty_ticks_since_flush >= LIVE_STATE_FLUSH_EVERY_TICKS:
                    _persist_live_checkpoint(runtime, builder)
                    dirty_ticks_since_flush = 0
                if bar is None:
                    continue
                result = runtime.process_bar(bar)
                _persist_live_checkpoint(runtime, builder)
                dirty_ticks_since_flush = 0
                _log_result(result)
                if result.kill_switch_active:
                    _emergency_flatten(runtime, result.kill_switch_reason or "kill switch")
                    raise RuntimeError(result.kill_switch_reason or "Kill switch active.")
            loops += 1
    finally:
        if broker_module is not None:
            broker_module.shutdown()


def main() -> None:
    log_config = configure_run_logging(
        "live_bridge",
        symbol=SYMBOL,
        capture_print=True,
        extra_text_log_paths=[Path("live_bot.log")],
    )
    set_log_context(symbol=SYMBOL)
    log.info(
        "Live bridge logging ready",
        extra={
            "event": "live_bridge_logging_ready",
            "text_log_path": log_config.text_log_path,
            "jsonl_log_path": log_config.jsonl_log_path,
            "legacy_text_log_path": Path("live_bot.log"),
        },
    )
    run_live_loop()


if __name__ == "__main__":
    main()

