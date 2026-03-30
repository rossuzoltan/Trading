from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_project_python(project_root: Path) -> str:
    try:
        from interpreter_guard import project_venv_python

        candidate = project_venv_python(project_root)
        if candidate is not None and candidate.exists():
            return str(candidate)
    except Exception:
        pass
    return sys.executable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive launcher for train_agent.py (env-var based config).")
    parser.add_argument("--symbol", default=os.environ.get("TRAIN_SYMBOL", "EURUSD"), help="e.g. EURUSD")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=int(os.environ.get("TRAIN_TOTAL_TIMESTEPS", "3000000")),
        help="Total PPO timesteps (TRAIN_TOTAL_TIMESTEPS).",
    )
    parser.add_argument(
        "--adaptive-tune",
        action="store_true",
        default=os.environ.get("TRAIN_ADAPTIVE_TUNE", "false").lower() == "true",
        help="Run TRAIN_ADAPTIVE_TUNE=true benchmark to pick TRAIN_NUM_ENVS before training.",
    )
    parser.add_argument(
        "--target-cpu-pct",
        type=int,
        default=int(os.environ.get("TRAIN_TARGET_CPU_PCT", "90")),
        help="Adaptive tune target CPU percent (TRAIN_TARGET_CPU_PCT).",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Start training in a child process, redirect logs to train_run.log/train_err.log, write train_pid.txt, then exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = _resolve_project_root()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    symbol = (args.symbol or "EURUSD").strip().upper()
    timesteps = max(int(args.timesteps or 0), 1)
    target_cpu = min(max(int(args.target_cpu_pct or 90), 10), 99)

    os.environ["TRAIN_SYMBOL"] = symbol
    os.environ["TRAIN_TOTAL_TIMESTEPS"] = str(timesteps)
    os.environ["TRAIN_ADAPTIVE_TUNE"] = "true" if bool(args.adaptive_tune) else "false"
    os.environ["TRAIN_TARGET_CPU_PCT"] = str(target_cpu)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    python_exe = _resolve_project_python(project_root)
    train_script = project_root / "train_agent.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Missing training entrypoint: {train_script}")

    cmd = [python_exe, "-u", str(train_script)]

    print("--- Adaptive Training Launcher ---")
    print(f"python: {python_exe}")
    print(f"symbol: {symbol}")
    print(f"timesteps: {timesteps}")
    print(f"adaptive_tune: {os.environ['TRAIN_ADAPTIVE_TUNE']} (target_cpu_pct={target_cpu})")
    print(f"cwd: {project_root}")
    print("Starting train_agent.py...\n")
    try:
        sys.stdout.flush()
    except Exception:
        pass

    env = os.environ.copy()
    if args.detach:
        stdout_path = project_root / "train_run.log"
        stderr_path = project_root / "train_err.log"
        pid_path = project_root / "train_pid.txt"
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)

        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr:
            process = subprocess.Popen(cmd, cwd=str(project_root), env=env, stdout=stdout, stderr=stderr)
        pid_path.write_text(str(process.pid), encoding="utf-8")
        print(f"Started training process with PID: {process.pid}")
        print(f"Stdout: {stdout_path}")
        print(f"Stderr: {stderr_path}")
        return 0

    return subprocess.call(cmd, cwd=str(project_root), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
