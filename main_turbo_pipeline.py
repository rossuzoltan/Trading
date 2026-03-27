from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from project_paths import ensure_runtime_dirs
from run_logging import configure_run_logging
from trading_config import resolve_bar_construction_ticks_per_bar


ROOT = Path(__file__).resolve().parent
DEFAULT_SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY")
log = logging.getLogger("main_turbo_pipeline")


def resolve_python() -> str:
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def run_step(args: list[str], desc: str, *, critical: bool = False, env: dict[str, str] | None = None) -> None:
    started_at = time.perf_counter()
    log.info(
        "Starting pipeline step",
        extra={
            "event": "pipeline_step_start",
            "step_description": desc,
            "command": args,
            "critical": critical,
        },
    )
    completed = subprocess.run(args, cwd=str(ROOT), env=env, check=False)
    if completed.returncode != 0:
        log.error(
            "Pipeline step failed",
            extra={
                "event": "pipeline_step_failed",
                "step_description": desc,
                "command": args,
                "critical": critical,
                "returncode": int(completed.returncode),
                "duration_s": round(time.perf_counter() - started_at, 3),
            },
        )
        if critical:
            raise SystemExit(completed.returncode)
    else:
        log.info(
            "Pipeline step completed",
            extra={
                "event": "pipeline_step_complete",
                "step_description": desc,
                "command": args,
                "returncode": int(completed.returncode),
                "duration_s": round(time.perf_counter() - started_at, 3),
            },
        )


def enable_windows_stay_awake() -> None:
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 | 0x00000040)
        log.info("PC stay-awake mode activated")
    except Exception as exc:  # pragma: no cover - best effort helper
        log.warning("Could not set stay-awake mode: %s", exc)


def main() -> None:
    ensure_runtime_dirs()
    log_config = configure_run_logging("main_turbo_pipeline", capture_print=True)
    log.info(
        "Turbo pipeline logging ready",
        extra={
            "event": "pipeline_logging_ready",
            "text_log_path": log_config.text_log_path,
            "jsonl_log_path": log_config.jsonl_log_path,
        },
    )
    enable_windows_stay_awake()
    start_time = time.time()

    python = resolve_python()
    ticks_per_bar = resolve_bar_construction_ticks_per_bar("BAR_SPEC_TICKS_PER_BAR", "TRADING_TICKS_PER_BAR")
    symbols = [
        symbol.strip().upper()
        for symbol in os.environ.get("PIPELINE_SYMBOLS", ",".join(DEFAULT_SYMBOLS)).split(",")
        if symbol.strip()
    ]
    days = int(os.environ.get("PIPELINE_DAYS", "1095"))

    run_step(
        [
            python,
            "-u",
            "download_dukascopy.py",
            "--pairs",
            *symbols,
            "--days",
            str(days),
            "--bar-volume",
            str(ticks_per_bar),
        ],
        "Data ingestion (Dukascopy ticks)",
        critical=True,
    )
    run_step(
        [python, "-u", "build_volume_bars.py", "--ticks-per-bar", str(ticks_per_bar)],
        "Build consolidated volume-bar dataset",
        critical=True,
    )

    for symbol in symbols:
        train_env = dict(os.environ)
        train_env["TRAIN_SYMBOL"] = symbol
        run_step([python, "-u", "train_agent.py"], f"Train model ({symbol})", env=train_env)

    for symbol in symbols:
        eval_env = dict(os.environ)
        eval_env["EVAL_SYMBOL"] = symbol
        run_step([python, "-u", "evaluate_oos.py"], f"OOS evaluation ({symbol})", env=eval_env)

    duration_h = (time.time() - start_time) / 3600
    log.info(
        "All pipeline phases complete",
        extra={
            "event": "pipeline_complete",
            "duration_hours": round(duration_h, 4),
            "symbols": symbols,
            "ticks_per_bar": int(ticks_per_bar),
        },
    )


if __name__ == "__main__":
    main()
