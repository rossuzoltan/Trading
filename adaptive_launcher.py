from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
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


def _read_current_training_run(project_root: Path) -> dict | None:
    path = project_root / "checkpoints" / "current_training_run.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_training_pid_from_current_run(
    *,
    project_root: Path,
    symbol: str,
    launch_started_utc: datetime,
    timeout_s: float = 15.0,
) -> int | None:
    deadline = time.time() + max(float(timeout_s), 0.1)
    symbol = (symbol or "").strip().upper()
    while time.time() < deadline:
        ctx = _read_current_training_run(project_root)
        if ctx:
            ctx_symbol = str(ctx.get("symbol", "")).strip().upper()
            if ctx_symbol == symbol:
                started_raw = str(ctx.get("process_started_utc", "")).strip()
                try:
                    started_utc = datetime.fromisoformat(started_raw)
                except Exception:
                    started_utc = None
                # If the file is fresh (started after we launched), trust its PID.
                if started_utc is None or started_utc >= launch_started_utc:
                    pid_raw = ctx.get("pid")
                    try:
                        return int(pid_raw)
                    except Exception:
                        pass
        time.sleep(0.25)
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive launcher for train_agent.py (env-var based config).")
    parser.add_argument("--symbol", default=os.environ.get("TRAIN_SYMBOL", "EURUSD"), help="e.g. EURUSD")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=int(os.environ.get("TRAIN_TOTAL_TIMESTEPS", "3000000")),
        help="Total PPO timesteps (TRAIN_TOTAL_TIMESTEPS).",
    )
    adaptive_tune_default = os.environ.get("TRAIN_ADAPTIVE_TUNE", "false").lower() == "true"
    parser.add_argument(
        "--adaptive-tune",
        dest="adaptive_tune",
        action="store_true",
        help="Run TRAIN_ADAPTIVE_TUNE=true benchmark to pick TRAIN_NUM_ENVS before training.",
    )
    parser.add_argument(
        "--no-adaptive-tune",
        dest="adaptive_tune",
        action="store_false",
        help="Disable adaptive tuning even if TRAIN_ADAPTIVE_TUNE is set in the shell.",
    )
    parser.set_defaults(adaptive_tune=adaptive_tune_default)
    parser.add_argument(
        "--target-cpu-pct",
        type=int,
        default=int(os.environ.get("TRAIN_TARGET_CPU_PCT", "90")),
        help="Adaptive tune target CPU percent (TRAIN_TARGET_CPU_PCT).",
    )
    force_dummy_default_env = os.environ.get("TRAIN_FORCE_DUMMY_VEC", "").strip()
    if force_dummy_default_env:
        force_dummy_default = force_dummy_default_env == "1"
    else:
        # Windows SubprocVecEnv historically had issues, but for performance (200+ SPS),
        # we should favor SubprocVecEnv and only fall back to dummy if explicitly forced.
        force_dummy_default = False
    parser.add_argument(
        "--force-dummy-vec",
        dest="force_dummy_vec",
        action="store_true",
        help="Set TRAIN_FORCE_DUMMY_VEC=1 (DummyVecEnv; no multiprocessing).",
    )
    parser.add_argument(
        "--no-force-dummy-vec",
        dest="force_dummy_vec",
        action="store_false",
        help="Set TRAIN_FORCE_DUMMY_VEC=0 (allow SubprocVecEnv on Windows).",
    )
    parser.set_defaults(force_dummy_vec=force_dummy_default)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=int(os.environ.get("TRAIN_NUM_ENVS", "8")),
        help="TRAIN_NUM_ENVS (used to pick env_workers / vector size).",
    )
    parser.add_argument(
        "--experiment-profile",
        default=os.environ.get("TRAIN_EXPERIMENT_PROFILE", ""),
        help="TRAIN_EXPERIMENT_PROFILE (e.g. reward_strip_hard_churn_alpha_gate).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=int(os.environ.get("TRAIN_WINDOW_SIZE", "8")),
        help="TRAIN_WINDOW_SIZE (observation window).",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=int(os.environ.get("TRAIN_EVAL_FREQ", "200000")),
        help="TRAIN_EVAL_FREQ (timesteps; internally scaled by env_workers).",
    )
    parser.add_argument(
        "--heartbeat-every",
        type=int,
        default=int(os.environ.get("TRAIN_HEARTBEAT_EVERY_STEPS", "2048")),
        help="TRAIN_HEARTBEAT_EVERY_STEPS.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=int(os.environ.get("TRAIN_N_FOLDS", "3")),
        help="Number of purged walk-forward folds (TRAIN_N_FOLDS).",
    )
    resume_latest_default = os.environ.get("TRAIN_RESUME_LATEST", "0") == "1"
    parser.add_argument(
        "--resume-latest",
        dest="resume_latest",
        action="store_true",
        help="Set TRAIN_RESUME_LATEST=1 to resume from checkpoints/current_training_run.json when possible.",
    )
    parser.add_argument(
        "--no-resume-latest",
        dest="resume_latest",
        action="store_false",
        help="Set TRAIN_RESUME_LATEST=0 (start fresh) even if TRAIN_RESUME_LATEST is set in the shell.",
    )
    parser.set_defaults(resume_latest=resume_latest_default)
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Start training in a child process, redirect logs to train_run.log/train_err.log, write train_pid.txt, then exit.",
    )
    parser.add_argument(
        "--kill-existing",
        action="store_true",
        help="If any train_agent.py from this repo is already running, terminate it before launching a new one.",
    )
    return parser.parse_args()


def _repo_train_agent_pids(project_root: Path) -> list[int]:
    try:
        import psutil  # type: ignore
    except Exception:
        return []

    root = str(project_root).lower()
    pids: list[int] = []
    for proc in psutil.process_iter(attrs=["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmd = " ".join(cmdline).lower()
        except Exception:
            continue
        if "train_agent.py" not in cmd:
            continue
        if root not in cmd:
            continue
        pids.append(int(proc.info["pid"]))
    return sorted(set(pids))


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
    folds = max(int(args.folds or 0), 1)
    num_envs = max(int(args.num_envs or 0), 1)
    window_size = max(int(args.window_size or 0), 1)
    eval_freq = max(int(args.eval_freq or 0), 1)
    heartbeat_every = max(int(args.heartbeat_every or 0), 1)
    experiment_profile = str(args.experiment_profile or "").strip().lower()

    os.environ["TRAIN_SYMBOL"] = symbol
    os.environ["TRAIN_TOTAL_TIMESTEPS"] = str(timesteps)
    os.environ["TRAIN_ADAPTIVE_TUNE"] = "true" if bool(args.adaptive_tune) else "false"
    os.environ["TRAIN_TARGET_CPU_PCT"] = str(target_cpu)
    os.environ["TRAIN_FORCE_DUMMY_VEC"] = "1" if bool(args.force_dummy_vec) else "0"
    os.environ["TRAIN_NUM_ENVS"] = str(num_envs)
    os.environ["TRAIN_EXPERIMENT_PROFILE"] = experiment_profile
    os.environ["TRAIN_WINDOW_SIZE"] = str(window_size)
    os.environ["TRAIN_EVAL_FREQ"] = str(eval_freq)
    os.environ["TRAIN_HEARTBEAT_EVERY_STEPS"] = str(heartbeat_every)
    os.environ["TRAIN_N_FOLDS"] = str(folds)
    os.environ["TRAIN_RESUME_LATEST"] = "1" if bool(args.resume_latest) else "0"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    # Stable defaults for long runs (allow manual overrides from caller env).
    os.environ.setdefault("TRAIN_PROGRESS_VERBOSE", "1")
    os.environ.setdefault("TRAIN_HEARTBEAT_EVERY_STEPS", "2048")

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
    print(f"force_dummy_vec: {os.environ['TRAIN_FORCE_DUMMY_VEC']}")
    print(f"num_envs: {num_envs}")
    print(f"experiment_profile: {experiment_profile or '(default)'}")
    print(f"window_size: {window_size}")
    print(f"eval_freq: {eval_freq}")
    print(f"heartbeat_every: {heartbeat_every}")
    print(f"folds: {folds}")
    print(f"resume_latest: {os.environ['TRAIN_RESUME_LATEST']}")
    print(f"cwd: {project_root}")
    print("Starting train_agent.py...\n")
    try:
        sys.stdout.flush()
    except Exception:
        pass

    env = os.environ.copy()
    if args.detach:
        launch_started_utc = datetime.now(timezone.utc)
        existing = _repo_train_agent_pids(project_root)
        if existing:
            if not args.kill_existing:
                print(f"[ABORT] train_agent.py already running for this repo: {existing}")
                print("Stop it first, or re-run with --kill-existing.")
                return 2
            for pid in existing:
                try:
                    import psutil  # type: ignore

                    psutil.Process(pid).terminate()
                except Exception:
                    pass

        stdout_path = project_root / "train_run.log"
        stderr_path = project_root / "train_err.log"
        pid_path = project_root / "train_pid.txt"

        try:
            stdout = stdout_path.open("w", encoding="utf-8", errors="replace")
            stderr = stderr_path.open("w", encoding="utf-8", errors="replace")
        except PermissionError:
            # Windows often denies delete/overwrite when a log is being tailed/opened.
            # Fall back to unique filenames instead of failing the launch.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            stdout_path = project_root / f"train_run_{stamp}.log"
            stderr_path = project_root / f"train_err_{stamp}.log"
            stdout = stdout_path.open("w", encoding="utf-8", errors="replace")
            stderr = stderr_path.open("w", encoding="utf-8", errors="replace")

        creationflags = 0
        if os.name == "nt":
            creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
            creationflags |= getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)

        try:
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(project_root),
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=creationflags,
                )
            except OSError:
                # Fall back to a plain detached group if job breakaway isn't allowed.
                fallback_flags = 0
                if os.name == "nt":
                    fallback_flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    fallback_flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
                process = subprocess.Popen(
                    cmd,
                    cwd=str(project_root),
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=fallback_flags,
                )
        finally:
            stdout.close()
            stderr.close()

        resolved_pid = _resolve_training_pid_from_current_run(
            project_root=project_root,
            symbol=symbol,
            launch_started_utc=launch_started_utc,
            timeout_s=15.0,
        )
        effective_pid = int(resolved_pid or process.pid)
        pid_path.write_text(str(effective_pid), encoding="utf-8")
        print(f"Started training process with PID: {effective_pid}")
        if resolved_pid and int(resolved_pid) != int(process.pid):
            print(f"[INFO] Initial PID {process.pid} re-execed/spawned into PID {resolved_pid}; pid file updated.")
        print(f"Stdout: {stdout_path}")
        print(f"Stderr: {stderr_path}")
        return 0

    return subprocess.call(cmd, cwd=str(project_root), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
