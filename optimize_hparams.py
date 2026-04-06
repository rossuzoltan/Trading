from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna

from interpreter_guard import ensure_project_venv

ensure_project_venv(project_root=Path(__file__).resolve().parent, script_path=__file__)


@dataclass(frozen=True)
class TrialResult:
    score: float
    per_symbol: dict[str, dict[str, Any]]
    artifacts_dir: Path


def _python_exe() -> Path:
    return Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"


def _run(cmd: list[str], *, env: dict[str, str], cwd: Path, timeout_s: int) -> None:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed (code={completed.returncode}): {' '.join(cmd)}\n{completed.stdout}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _objective(
    trial: optuna.Trial,
    *,
    symbols: list[str],
    project_root: Path,
    train_timesteps: int,
    eval_max_bars: int,
    timeout_train_s: int,
    timeout_eval_s: int,
) -> float:
    lr = trial.suggest_float("ppo_learning_rate", 1e-5, 2e-3, log=True)
    min_lr = trial.suggest_float("ppo_min_learning_rate", 1e-6, 5e-4, log=True)
    ent = trial.suggest_float("ppo_ent_coef", 1e-4, 1e-1, log=True)
    n_steps = trial.suggest_categorical("ppo_n_steps", [512, 1024, 2048, 4096])
    n_epochs = trial.suggest_int("ppo_n_epochs", 5, 15)

    # Keep PPO batch constraints simple and safe (batch_size <= n_steps * n_envs).
    trial.set_user_attr("ppo_batch_size", int(n_steps))

    artifacts_dir = project_root / "models" / "optuna" / (trial.study.study_name or "study") / f"trial_{trial.number:05d}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    env_base = dict(os.environ)
    opt_num_envs = int(os.environ.get("OPT_TRAIN_NUM_ENVS", "1") or "1")
    opt_use_amp = os.environ.get("OPT_USE_AMP", "0") == "1"
    opt_experiment_profile = os.environ.get("OPT_TRAIN_EXPERIMENT_PROFILE", "").strip()
    env_base.update(
        {
            "TRAIN_MODEL_DIR": str(artifacts_dir),
            "TRAIN_NUM_ENVS": str(int(max(opt_num_envs, 1))),
            "TRAIN_TOTAL_TIMESTEPS": str(int(train_timesteps)),
            "TRAIN_EXPORT_BEST_FOLD": "1",
            "TRAIN_PPO_LEARNING_RATE": str(float(lr)),
            "TRAIN_PPO_MIN_LEARNING_RATE": str(float(min_lr)),
            "TRAIN_PPO_ENT_COEF": str(float(ent)),
            "TRAIN_PPO_N_STEPS": str(int(n_steps)),
            "TRAIN_PPO_BATCH_SIZE": str(int(n_steps)),
            "TRAIN_PPO_N_EPOCHS": str(int(n_epochs)),
            "TRAIN_REDUCE_LOGGING": "1",
            "TRAIN_ASYNC_EVAL": "0",
        }
    )
    if opt_use_amp:
        env_base["TRAIN_USE_AMP"] = "1"
    if opt_experiment_profile:
        env_base["TRAIN_EXPERIMENT_PROFILE"] = opt_experiment_profile

    per_symbol: dict[str, dict[str, Any]] = {}
    score_total = 0.0

    for symbol in symbols:
        symbol = symbol.strip().upper()
        if not symbol:
            continue

        env_train = dict(env_base)
        env_train["TRAIN_SYMBOL"] = symbol

        _run(
            [str(_python_exe()), "-u", "train_agent.py"],
            env=env_train,
            cwd=project_root,
            timeout_s=timeout_train_s,
        )

        manifest_path = artifacts_dir / f"artifact_manifest_{symbol}.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Training did not produce a promoted manifest for {symbol}: {manifest_path}")

        env_eval = dict(os.environ)
        env_eval.update(
            {
                "EVAL_SYMBOL": symbol,
                "EVAL_MANIFEST_PATH": str(manifest_path),
                "EVAL_OUTPUT_DIR": str(artifacts_dir),
                "EVAL_MAX_BARS": str(int(eval_max_bars)),
                "EVAL_SKIP_PLOT": "1",
            }
        )
        _run(
            [str(_python_exe()), "-u", "evaluate_oos.py"],
            env=env_eval,
            cwd=project_root,
            timeout_s=timeout_eval_s,
        )

        report_path = artifacts_dir / f"replay_report_{symbol.lower()}.json"
        report = _read_json(report_path)
        metrics = dict(report.get("replay_metrics", {}) or {})

        net_pnl = float(metrics.get("net_pnl_usd", 0.0))
        max_dd = float(metrics.get("max_drawdown", 1.0))
        sharpe = float(metrics.get("timed_sharpe", 0.0))
        trade_count = int(metrics.get("trade_count", 0))

        # Score: net PnL minus equity-scaled drawdown penalty, plus modest Sharpe bonus.
        # Equity baseline is $1,000 in evaluate_oos.py.
        score = net_pnl - (max_dd * 1000.0) + (sharpe * 25.0)
        if trade_count < 10:
            score -= float((10 - trade_count) * 10.0)

        per_symbol[symbol] = {
            "score": float(score),
            "net_pnl_usd": float(net_pnl),
            "max_drawdown": float(max_dd),
            "timed_sharpe": float(sharpe),
            "trade_count": int(trade_count),
            "replay_report_path": str(report_path),
        }
        score_total += float(score)

    avg_score = float(score_total / max(len(per_symbol), 1))
    trial.set_user_attr("artifacts_dir", str(artifacts_dir))
    trial.set_user_attr("per_symbol", per_symbol)
    return avg_score


def main() -> None:
    project_root = Path(__file__).resolve().parent

    symbols = [s.strip().upper() for s in os.environ.get("OPT_SYMBOLS", "EURUSD,GBPUSD,USDJPY").split(",") if s.strip()]
    n_trials = int(os.environ.get("OPT_TRIALS", "10") or "10")
    train_timesteps = int(os.environ.get("OPT_TRAIN_TIMESTEPS", "200000") or "200000")
    eval_max_bars = int(os.environ.get("OPT_EVAL_MAX_BARS", "5000") or "5000")
    timeout_train_s = int(os.environ.get("OPT_TRAIN_TIMEOUT_S", "7200") or "7200")
    timeout_eval_s = int(os.environ.get("OPT_EVAL_TIMEOUT_S", "1800") or "1800")

    study_name = os.environ.get("OPT_STUDY_NAME", "hpo").strip() or "hpo"
    storage_path = os.environ.get("OPT_STORAGE", str(project_root / "checkpoints" / f"optuna_{study_name}.db"))
    storage = f"sqlite:///{storage_path}"

    (project_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    print(f"[Optuna] study={study.study_name} storage={storage}")
    print(f"[Optuna] symbols={symbols} trials={n_trials} train_timesteps={train_timesteps} eval_max_bars={eval_max_bars}")

    study.optimize(
        lambda t: _objective(
            t,
            symbols=symbols,
            project_root=project_root,
            train_timesteps=train_timesteps,
            eval_max_bars=eval_max_bars,
            timeout_train_s=timeout_train_s,
            timeout_eval_s=timeout_eval_s,
        ),
        n_trials=n_trials,
    )

    best = study.best_trial
    best_summary = {
        "study": study.study_name,
        "best_value": float(best.value),
        "best_params": dict(best.params),
        "user_attrs": dict(best.user_attrs),
    }
    out_path = project_root / "models" / "optuna" / study.study_name / "best_trial.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best_summary, indent=2), encoding="utf-8")

    print(f"[Optuna] best_value={best.value}")
    print(f"[Optuna] best_params={best.params}")
    print(f"[Optuna] saved -> {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
