import subprocess
import time
import os
import sys

def run_step(cmd, desc):
    print(f"\n{'='*60}")
    print(f"STEP: {desc}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}\n")
    
    # Use -u for unbuffered output to ensure we see progress in logs
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    
    if process.returncode != 0:
        print(f"\n[ERROR] Step failed with code {process.returncode}: {desc}")
        # We continue anyway to try and save what we can, or we could exit.
        # For a "Turbo" run, we'll exit if a critical data step fails.
        if "Download" in desc or "Build" in desc:
            sys.exit(1)
    else:
        print(f"\n[SUCCESS] Completed: {desc}")

def main():
    # ── PC STAY-AWAKE LOGIC ──
    # This prevents Windows from sleeping while the script is running
    try:
        import ctypes
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 | 0x00000040)
        print("[INFO] PC 'Stay-Awake' mode activated. (System will not sleep, but screen can be turned off)")
    except Exception as e:
        print(f"[WARN] Could not set Stay-Awake mode: {e}")

    start_time = time.time()
    
    # 1. Turbo Ingestion (3-year history for 3 pairs)
    # Using the optimized 2000-tick bar volume
    run_step(
        "./.venv/Scripts/python -u download_dukascopy.py --pairs EURUSD GBPUSD USDJPY --days 1095 --bar-volume 2000",
        "Phase 12: Turbo Data Ingestion (Dukascopy Ticks)"
    )

    # 2. Build Unified Training Set
    # Consolidates all pairs into DATA_CLEAN_VOLUME.csv
    run_step(
        "./.venv/Scripts/python -u build_volume_bars.py --ticks-per-bar 2000",
        "Phase 13: Building Consolidated Institutional Dataset"
    )

    # 3. Training Run (Phase 13: 3M step MaskablePPO)
    # Includes FracDiff, Risk-Parity, and Regime-Aware logic
    run_step(
        "./.venv/Scripts/python -u train_agent.py",
        "Phase 14: 3.0M Step Institutional Training"
    )

    # 4. Final OOS Validation
    # Generates final performance report and equity curves
    run_step(
        "./.venv/Scripts/python -u evaluate_oos.py",
        "Phase 15: Out-of-Sample Final Audit"
    )

    end_time = time.time()
    duration_h = (end_time - start_time) / 3600
    print(f"\n{'#'*60}")
    print(f"ALL PHASES COMPLETE! Total duration: {duration_h:.2f} hours")
    print("Check 'models/' for the best agent and equity-curve outputs.")
    print(f"{'#'*60}")

if __name__ == "__main__":
    main()
