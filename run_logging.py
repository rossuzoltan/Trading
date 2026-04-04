from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from project_paths import LOGS_DIR, ensure_runtime_dirs


_STANDARD_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}
_DEFAULT_CONTEXT = {
    "component": "-",
    "symbol": "-",
    "run_id": "-",
    "event": "-",
    "stream": "-",
}
_START_MONOTONIC = time.perf_counter()
_SEQUENCE = 0
_SEQUENCE_LOCK = threading.Lock()
_CONTEXT_LOCK = threading.Lock()
_CURRENT_CONTEXT: dict[str, str] = dict(_DEFAULT_CONTEXT)
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_ORIGINAL_EXCEPTHOOK = sys.excepthook
_ORIGINAL_THREADING_EXCEPTHOOK = getattr(threading, "excepthook", None)
_INSTALLED_STDOUT: _LoggerWriter | None = None
_INSTALLED_STDERR: _LoggerWriter | None = None


class _SafeStream(io.TextIOBase):
    """Best-effort text stream wrapper.

    Windows consoles and some redirected streams can throw UnicodeEncodeError
    when log messages contain non-ASCII characters. Logging should never crash
    long training/eval runs, so we fall back to writing UTF-8 bytes with
    replacement when needed.
    """

    def __init__(self, stream: TextIO) -> None:
        super().__init__()
        self._stream = stream

    def write(self, s: str) -> int:  # type: ignore[override]
        try:
            return int(self._stream.write(s))
        except UnicodeEncodeError:
            buffer = getattr(self._stream, "buffer", None)
            if buffer is not None:
                buffer.write(str(s).encode("utf-8", errors="replace"))
                return len(s)
            safe = str(s).encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            return int(self._stream.write(safe))

    def flush(self) -> None:  # type: ignore[override]
        try:
            self._stream.flush()
        except Exception:
            pass


@dataclass(frozen=True)
class RunLoggingConfig:
    component: str
    text_log_path: Path
    jsonl_log_path: Path
    logger_name: str


class _MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = int(max_level)

    def filter(self, record: logging.LogRecord) -> bool:
        return int(record.levelno) <= self.max_level


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "sequence") and hasattr(record, "elapsed_s"):
            return True
        global _SEQUENCE
        with _SEQUENCE_LOCK:
            _SEQUENCE += 1
            sequence = _SEQUENCE

        with _CONTEXT_LOCK:
            current_context = dict(_CURRENT_CONTEXT)

        for key, fallback in _DEFAULT_CONTEXT.items():
            value = getattr(record, key, current_context.get(key, fallback))
            setattr(record, key, fallback if value in (None, "") else str(value))

        record.sequence = sequence
        record.elapsed_s = max(time.perf_counter() - _START_MONOTONIC, 0.0)
        return True


class _HumanFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return timestamp.isoformat(timespec="milliseconds")

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        context_bits = [
            f"symbol={record.symbol}" if record.symbol != "-" else "",
            f"run={record.run_id}" if record.run_id != "-" else "",
            f"event={record.event}" if record.event != "-" else "",
            f"stream={record.stream}" if record.stream != "-" else "",
        ]
        context = " ".join(bit for bit in context_bits if bit)
        prefix = (
            f"{self.formatTime(record)} | +{record.elapsed_s:8.3f}s | "
            f"#{record.sequence:06d} | {record.levelname:<7} | {record.component}"
        )
        if context:
            return f"{prefix} | {context} | {message}"
        return f"{prefix} | {message}"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp_utc": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "elapsed_s": round(float(record.elapsed_s), 6),
            "sequence": int(record.sequence),
            "level": record.levelname,
            "logger": record.name,
            "component": record.component,
            "message": record.getMessage(),
            "pid": int(record.process),
            "thread": record.threadName,
        }
        for key in ("symbol", "run_id", "event", "stream"):
            value = getattr(record, key, "-")
            if value != "-":
                payload[key] = value

        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_KEYS or key in payload or key.startswith("_"):
                continue
            payload[key] = _json_safe(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = record.stack_info
        return json.dumps(payload, sort_keys=True)


class _LoggerWriter(io.TextIOBase):
    def __init__(self, logger: logging.Logger, level: int, stream_name: str) -> None:
        super().__init__()
        self._logger = logger
        self._level = int(level)
        self._stream_name = str(stream_name)
        self._buffer = ""

    @property
    def encoding(self) -> str:
        return "utf-8"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line.rstrip("\r"))
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer.rstrip("\r"))
            self._buffer = ""

    def _emit(self, line: str) -> None:
        if not line:
            return
        self._logger.log(self._level, line, extra={"stream": self._stream_name})


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _resolve_level(raw_level: str | int | None) -> int:
    if raw_level is None:
        raw_level = os.environ.get("TRADING_LOG_LEVEL", "INFO")
    if isinstance(raw_level, int):
        return raw_level
    name = str(raw_level).strip().upper()
    return int(getattr(logging, name, logging.INFO))


def _sanitize_token(value: str | None, fallback: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return fallback
    chars = []
    for character in raw:
        if character.isalnum() or character in {"-", "_"}:
            chars.append(character.lower())
        else:
            chars.append("_")
    sanitized = "".join(chars).strip("_")
    return sanitized or fallback


def _build_default_log_paths(
    *,
    component: str,
    symbol: str | None,
    run_id: str | None,
) -> tuple[Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = [_sanitize_token(component, "run")]
    if symbol:
        parts.append(_sanitize_token(symbol, "symbol"))
    if run_id:
        parts.append(_sanitize_token(run_id, "run"))
    else:
        parts.append(timestamp.lower())
    stem = "_".join(parts)
    base_dir = LOGS_DIR / _sanitize_token(component, "run")
    return base_dir / f"{stem}.log", base_dir / f"{stem}.jsonl"


def _detach_existing_handlers(root_logger: logging.Logger) -> None:
    for handler in list(root_logger.handlers):
        if getattr(handler, "_trading_run_logging", False):
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def _install_exception_hooks(logger: logging.Logger) -> None:
    def _log_uncaught_exception(exc_type: type[BaseException], exc_value: BaseException, exc_traceback: Any) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback), extra={"event": "uncaught_exception"})

    sys.excepthook = _log_uncaught_exception

    if _ORIGINAL_THREADING_EXCEPTHOOK is not None:
        def _log_thread_exception(args: Any) -> None:
            if issubclass(args.exc_type, KeyboardInterrupt):
                _ORIGINAL_THREADING_EXCEPTHOOK(args)
                return
            logger.critical(
                "Uncaught thread exception",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
                extra={"event": "thread_exception", "thread_name": getattr(args.thread, "name", None)},
            )

        threading.excepthook = _log_thread_exception


def configure_run_logging(
    component: str,
    *,
    symbol: str | None = None,
    run_id: str | None = None,
    logger_name: str | None = None,
    text_log_path: str | Path | None = None,
    jsonl_log_path: str | Path | None = None,
    extra_text_log_paths: list[str | Path] | None = None,
    capture_print: bool = True,
    level: str | int | None = None,
) -> RunLoggingConfig:
    ensure_runtime_dirs()
    # Default to "production-style" logging behavior: handler failures should not
    # spam stderr nor risk recursive logging loops when stdout/stderr are captured.
    # Opt-in to noisy handler exceptions via TRADING_LOG_RAISE_EXCEPTIONS=1.
    if os.environ.get("TRADING_LOG_RAISE_EXCEPTIONS", "0") != "1":
        logging.raiseExceptions = False

    # Best-effort: make console streams tolerant of Unicode to prevent
    # UnicodeEncodeError inside StreamHandler on Windows.
    for stream in (_ORIGINAL_STDOUT, _ORIGINAL_STDERR):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
        except Exception:
            pass
    component_value = str(component).strip() or "run"
    logger_name_value = str(logger_name or component_value)
    resolved_level = _resolve_level(level)
    default_text_log_path, default_jsonl_log_path = _build_default_log_paths(
        component=component_value,
        symbol=symbol,
        run_id=run_id,
    )
    resolved_text_log_path = Path(text_log_path) if text_log_path is not None else default_text_log_path
    resolved_jsonl_log_path = Path(jsonl_log_path) if jsonl_log_path is not None else default_jsonl_log_path

    for path in [resolved_text_log_path, resolved_jsonl_log_path, *(Path(item) for item in (extra_text_log_paths or []))]:
        path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)
    root_logger.propagate = False
    _detach_existing_handlers(root_logger)

    context_filter = _ContextFilter()
    context_filter._trading_run_logging = True  # type: ignore[attr-defined]
    human_formatter = _HumanFormatter()
    json_formatter = _JsonFormatter()

    stdout_handler = logging.StreamHandler(_SafeStream(_ORIGINAL_STDOUT))
    stdout_handler.setLevel(resolved_level)
    stdout_handler.addFilter(_MaxLevelFilter(logging.WARNING))
    stdout_handler.addFilter(context_filter)
    stdout_handler.setFormatter(human_formatter)
    stdout_handler._trading_run_logging = True  # type: ignore[attr-defined]
    root_logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(_SafeStream(_ORIGINAL_STDERR))
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.addFilter(context_filter)
    stderr_handler.setFormatter(human_formatter)
    stderr_handler._trading_run_logging = True  # type: ignore[attr-defined]
    root_logger.addHandler(stderr_handler)

    text_handler = logging.FileHandler(resolved_text_log_path, encoding="utf-8")
    text_handler.setLevel(resolved_level)
    text_handler.addFilter(context_filter)
    text_handler.setFormatter(human_formatter)
    text_handler._trading_run_logging = True  # type: ignore[attr-defined]
    root_logger.addHandler(text_handler)

    jsonl_handler = logging.FileHandler(resolved_jsonl_log_path, encoding="utf-8")
    jsonl_handler.setLevel(resolved_level)
    jsonl_handler.addFilter(context_filter)
    jsonl_handler.setFormatter(json_formatter)
    jsonl_handler._trading_run_logging = True  # type: ignore[attr-defined]
    root_logger.addHandler(jsonl_handler)

    for extra_path in extra_text_log_paths or []:
        extra_handler = logging.FileHandler(Path(extra_path), encoding="utf-8")
        extra_handler.setLevel(resolved_level)
        extra_handler.addFilter(context_filter)
        extra_handler.setFormatter(human_formatter)
        extra_handler._trading_run_logging = True  # type: ignore[attr-defined]
        root_logger.addHandler(extra_handler)

    logging.captureWarnings(True)
    set_log_context(component=component_value, symbol=symbol, run_id=run_id)

    logger = logging.getLogger(logger_name_value)
    _install_exception_hooks(logger)

    global _INSTALLED_STDOUT, _INSTALLED_STDERR
    if capture_print:
        _INSTALLED_STDOUT = _LoggerWriter(logging.getLogger(f"{logger_name_value}.stdout"), logging.INFO, "stdout")
        _INSTALLED_STDERR = _LoggerWriter(logging.getLogger(f"{logger_name_value}.stderr"), logging.ERROR, "stderr")
        sys.stdout = _INSTALLED_STDOUT
        sys.stderr = _INSTALLED_STDERR

    logger.info(
        "Logging configured",
        extra={
            "event": "logging_configured",
            "text_log_path": resolved_text_log_path,
            "jsonl_log_path": resolved_jsonl_log_path,
        },
    )
    return RunLoggingConfig(
        component=component_value,
        text_log_path=resolved_text_log_path,
        jsonl_log_path=resolved_jsonl_log_path,
        logger_name=logger_name_value,
    )


def set_log_context(**fields: Any) -> None:
    with _CONTEXT_LOCK:
        for key, value in fields.items():
            if key not in _DEFAULT_CONTEXT:
                continue
            _CURRENT_CONTEXT[key] = _DEFAULT_CONTEXT[key] if value in (None, "") else str(value)


def shutdown_run_logging() -> None:
    global _INSTALLED_STDOUT, _INSTALLED_STDERR, _SEQUENCE
    if _INSTALLED_STDOUT is not None:
        _INSTALLED_STDOUT.flush()
    if _INSTALLED_STDERR is not None:
        _INSTALLED_STDERR.flush()
    sys.stdout = _ORIGINAL_STDOUT
    sys.stderr = _ORIGINAL_STDERR
    _INSTALLED_STDOUT = None
    _INSTALLED_STDERR = None
    logging.captureWarnings(False)
    sys.excepthook = _ORIGINAL_EXCEPTHOOK
    if _ORIGINAL_THREADING_EXCEPTHOOK is not None:
        threading.excepthook = _ORIGINAL_THREADING_EXCEPTHOOK
    root_logger = logging.getLogger()
    _detach_existing_handlers(root_logger)
    _SEQUENCE = 0
    with _CONTEXT_LOCK:
        _CURRENT_CONTEXT.clear()
        _CURRENT_CONTEXT.update(_DEFAULT_CONTEXT)
