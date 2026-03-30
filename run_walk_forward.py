"""run_walk_forward.py
======================
Phase 6 Reproducible Walk-Forward orchestrator.

Orchestrates sequential execution for a given seed and config:
1. Calls train_agent.py (with enforced clean run)
2. Runs validation_metrics.py to compute holdout metrics.
3. Automatically evaluates OOS via evaluate_oos.py and logs trades.
4. Generates a baseline comparison (tools/compare_oos_baselines.py).
5. Runs the Phase 3 loss diagnostic tool (tools/diagnose_losses.py).

Produces a unified output report in the `models/` directory.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

def _run_cmd(cmd: list[str], cwd: str | Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"\n[{cmd[0]}] {' '.join(cmd)}")
    import os
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=False, env=run_env)
    if result.returncode != 0:
        print(f"\nERROR: Command failed with exit code {result.returncode}")
        print(f"Failed cmd: {' '.join(cmd)}")
        sys.exit(result.returncode)

def main() -> int:
    parser = argparse.ArgumentParser(description="Reproducible Walk-Forward Pipeline")
    parser.add_argument("--symbol", default="EURUSD", help="Target pair")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timesteps", type=int, default=3_000_000, help="Total curriculum timesteps")
    args = parser.parse_args()
    
    symbol = str(args.symbol).upper()
    
    print(f"=== Starting Turbo Walk-Forward Pipeline for {symbol} ===")
    print(f"Seed: {args.seed} | Timesteps: {args.timesteps}")
    
    root_dir = Path(__file__).resolve().parent

    print("\n\n--- 1. Training Agent ---")
    _run_cmd([
        sys.executable, str(root_dir / "train_agent.py"),
        "--symbol", symbol,
        "--total-timesteps", str(args.timesteps)
    ], cwd=root_dir)
    
    print("\n\n--- 2. End-to-End Evaluation (Holdout) ---")
    _run_cmd([
        sys.executable, str(root_dir / "evaluate_oos.py"),
        "--symbol", symbol,
        "--render-gui", "False",
    ], cwd=root_dir)
    
    print("\n\n--- 3. Baseline Comparison ---")
    _run_cmd([
        sys.executable, str(root_dir / "tools" / "compare_oos_baselines.py"),
        "--symbol", symbol
    ], cwd=root_dir)
    
    print("\n\n--- 4. Loss Diagnostics ---")
    _run_cmd([
        sys.executable, str(root_dir / "tools" / "diagnose_losses.py"),
        "--symbol", symbol
    ], cwd=root_dir)
    
    print("\n\n--- 5. Cost-Stress Scenario (2x Slippage, 1.5x Commission) ---")
    stress_env = {
        "TRADING_SLIPPAGE_PIPS": "4.0",  # assuming 2.0 is baseline
        "TRADING_COMMISSION_PER_LOT": "12.0", # assuming 8.0 is baseline
    }
    _run_cmd([
        sys.executable, str(root_dir / "evaluate_oos.py"),
        "--symbol", symbol,
        "--render-gui", "False"
    ], cwd=root_dir, env=stress_env)
    
    print("\n=== Walk-Forward Pipeline Complete ===")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
