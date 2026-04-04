from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil


REPO_ROOT = Path(__file__).resolve().parent.parent
CURRENT_RUN_CONTEXT = REPO_ROOT / "checkpoints" / "current_training_run.json"
BENCH_ROOT = REPO_ROOT / "checkpoints" / "bench_sps"


@dataclass
class BenchResult:
    num_envs: int
    n_steps: int
    n_epochs: int
    batch_size: int
    total_timesteps: int
    returncode: int
    timed_out: bool
    duration_seconds: float
    state: str | None
    run_id: str | None
    checkpoints_root: str | None
    heartbeat_path: str | None
    steps_per_second_window: float | None
    steps_per_second_mean: float | None
    cpu_pct_mean: float | None
    cpu_pct_peak: float | None
    ram_pct_mean: float | None
    ram_pct_peak: float | None
    gpu_pct_mean: float | None
    gpu_pct_peak: float | None
    vram_pct_mean: float | None
    vram_pct_peak: float | None
    stderr_log: str
    stdout_log: str


def _parse_csv_ints(value: str) -> list[int]:
    parsed = []
    for item in str(value).split(","):
        token = item.strip()
        if token:
            parsed.append(int(token))
    if not parsed:
        raise ValueError(f"Expected at least one integer in {value!r}.")
    return parsed


def _resolve_checkpoint_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _stop_repo_training() -> None:
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(REPO_ROOT / "tools" / "stop_repo_training.ps1"),
            "-RepoRoot",
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _gpu_metrics() -> tuple[float | None, float | None]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None, None
    if not output:
        return None, None
    first_line = output.splitlines()[0]
    parts = [part.strip() for part in first_line.split(",", maxsplit=1)]
    if len(parts) != 2:
        return None, None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None, None


def _latest_heartbeat_for_context(context: dict[str, Any] | None) -> Path | None:
    if not context:
        return None
    checkpoints_root = _resolve_checkpoint_path(context.get("checkpoints_root"))
    if checkpoints_root is None or not checkpoints_root.exists():
        return None
    candidates = list(checkpoints_root.glob("fold_*/training_heartbeat.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def _default_batch_size(num_envs: int, n_steps: int) -> int:
    rollout = max(int(num_envs) * int(n_steps), 1)
    return min(max(rollout // 2, 256), 4096)


def _run_single_benchmark(
    *,
    symbol: str,
    total_timesteps: int,
    num_envs: int,
    n_steps: int,
    n_epochs: int,
    batch_size: int,
    heartbeat_every_steps: int,
    timeout_seconds: int,
    output_dir: Path,
) -> BenchResult:
    label = f"env{num_envs}_steps{n_steps}_epochs{n_epochs}_batch{batch_size}"
    stdout_log = output_dir / f"{label}.stdout.log"
    stderr_log = output_dir / f"{label}.stderr.log"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "FEATURE_ENGINE_FAST": "1",
            "TRAIN_BENCH_SPS": "1",
            "TRAIN_SYMBOL": symbol.upper(),
            "TRAIN_TOTAL_TIMESTEPS": str(total_timesteps),
            "TRAIN_NUM_ENVS": str(num_envs),
            "TRAIN_PPO_N_STEPS": str(n_steps),
            "TRAIN_PPO_BATCH_SIZE": str(batch_size),
            "TRAIN_PPO_N_EPOCHS": str(n_epochs),
            "TRAIN_EVAL_FREQ": "0",
            "TRAIN_HEARTBEAT_EVERY_STEPS": str(heartbeat_every_steps),
            "TRAIN_LOG_INTERVAL": "20",
            "TRAIN_REDUCE_LOGGING": "1",
            "TRAIN_ASYNC_EVAL": "0",
            "TRAIN_RESUME_LATEST": "0",
            "TRAIN_DEBUG_ALLOW_BASELINE_BYPASS": "1",
            "TRAIN_ADAPTIVE_KL_LR": "0",
        }
    )

    _stop_repo_training()
    start_time = time.perf_counter()
    cpu_samples: list[float] = []
    ram_samples: list[float] = []
    gpu_samples: list[float] = []
    vram_samples: list[float] = []
    timed_out = False
    returncode = -1

    with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            [str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"), "-u", "train_agent.py"],
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        try:
            while True:
                returncode = process.poll()
                if returncode is not None:
                    break
                cpu_samples.append(float(psutil.cpu_percent(interval=1.0)))
                ram_samples.append(float(psutil.virtual_memory().percent))
                gpu_pct, vram_pct = _gpu_metrics()
                if gpu_pct is not None:
                    gpu_samples.append(float(gpu_pct))
                if vram_pct is not None:
                    vram_samples.append(float(vram_pct))
                if (time.perf_counter() - start_time) >= timeout_seconds:
                    timed_out = True
                    break
        finally:
            if timed_out and process.poll() is None:
                _stop_repo_training()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
            returncode = process.wait(timeout=30) if process.poll() is None else int(process.returncode)

    duration_seconds = max(time.perf_counter() - start_time, 0.0)
    context = _load_json(CURRENT_RUN_CONTEXT)
    heartbeat_path = _latest_heartbeat_for_context(context)
    heartbeat = _load_json(heartbeat_path)

    def _mean(values: list[float]) -> float | None:
        return (sum(values) / len(values)) if values else None

    def _peak(values: list[float]) -> float | None:
        return max(values) if values else None

    return BenchResult(
        num_envs=num_envs,
        n_steps=n_steps,
        n_epochs=n_epochs,
        batch_size=batch_size,
        total_timesteps=total_timesteps,
        returncode=int(returncode),
        timed_out=bool(timed_out),
        duration_seconds=float(duration_seconds),
        state=str((context or {}).get("state")) if context else None,
        run_id=str((context or {}).get("run_id")) if context else None,
        checkpoints_root=str((context or {}).get("checkpoints_root")) if context else None,
        heartbeat_path=str(heartbeat_path) if heartbeat_path is not None else None,
        steps_per_second_window=float(heartbeat.get("steps_per_second_window")) if heartbeat and heartbeat.get("steps_per_second_window") is not None else None,
        steps_per_second_mean=float(heartbeat.get("steps_per_second")) if heartbeat and heartbeat.get("steps_per_second") is not None else None,
        cpu_pct_mean=_mean(cpu_samples),
        cpu_pct_peak=_peak(cpu_samples),
        ram_pct_mean=_mean(ram_samples),
        ram_pct_peak=_peak(ram_samples),
        gpu_pct_mean=_mean(gpu_samples),
        gpu_pct_peak=_peak(gpu_samples),
        vram_pct_mean=_mean(vram_samples),
        vram_pct_peak=_peak(vram_samples),
        stderr_log=str(stderr_log),
        stdout_log=str(stdout_log),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark training SPS across PPO worker/update configurations.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--total-timesteps", type=int, default=120000)
    parser.add_argument("--num-envs", default="8,12,16,20")
    parser.add_argument("--n-steps", default="512,1024,2048")
    parser.add_argument("--n-epochs", default="1,2,3")
    parser.add_argument("--heartbeat-every-steps", type=int, default=20000)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--max-cpu", type=float, default=95.0)
    parser.add_argument("--output", default=str(BENCH_ROOT / "latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_envs_values = _parse_csv_ints(args.num_envs)
    n_steps_values = _parse_csv_ints(args.n_steps)
    n_epochs_values = _parse_csv_ints(args.n_epochs)

    results: list[BenchResult] = []
    for num_envs, n_steps, n_epochs in itertools.product(num_envs_values, n_steps_values, n_epochs_values):
        batch_size = _default_batch_size(num_envs, n_steps)
        result = _run_single_benchmark(
            symbol=args.symbol,
            total_timesteps=args.total_timesteps,
            num_envs=num_envs,
            n_steps=n_steps,
            n_epochs=n_epochs,
            batch_size=batch_size,
            heartbeat_every_steps=args.heartbeat_every_steps,
            timeout_seconds=args.timeout_seconds,
            output_dir=output_path.parent,
        )
        results.append(result)

    eligible = [
        item
        for item in results
        if item.returncode == 0
        and not item.timed_out
        and item.steps_per_second_window is not None
        and (item.cpu_pct_peak is None or item.cpu_pct_peak < args.max_cpu)
    ]
    best = max(
        eligible,
        key=lambda item: (
            float(item.steps_per_second_window or 0.0),
            float(item.steps_per_second_mean or 0.0),
        ),
        default=None,
    )

    payload = {
        "symbol": str(args.symbol).upper(),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_rule": {
            "metric": "steps_per_second_window",
            "max_cpu_peak": float(args.max_cpu),
        },
        "results": [asdict(item) for item in results],
        "best": asdict(best) if best is not None else None,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Benchmark results -> {output_path}")
    for item in results:
        print(
            f"{item.num_envs:>2} env | n_steps={item.n_steps:<4} | epochs={item.n_epochs:<2} | "
            f"batch={item.batch_size:<4} | sps_window={item.steps_per_second_window} | "
            f"cpu_peak={item.cpu_pct_peak} | state={item.state} | rc={item.returncode}"
        )
    if best is None:
        print("No eligible benchmark result met the selection rule.")
        return 1
    print(
        "Selected best config: "
        f"envs={best.num_envs}, n_steps={best.n_steps}, epochs={best.n_epochs}, "
        f"batch={best.batch_size}, sps_window={best.steps_per_second_window}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
