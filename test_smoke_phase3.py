import os
import time
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from runtime_gym_env import RuntimeGymEnv, RuntimeGymConfig
from runtime_common import build_simple_action_map
from feature_engine import FEATURE_COLS

def make_env(rank, df, bars, scaler, action_map):
    def _init():
        env = RuntimeGymEnv(
            symbol="EURUSD",
            bars_frame=df,
            bars=bars,
            scaler=scaler,
            action_map=action_map,
            config=RuntimeGymConfig(
                reward_scale=10000.0,
                random_start=True,
                churn_min_hold_bars=5,
                churn_action_cooldown=10
            )
        )
        return env
    return _init

def run_smoke_test(num_envs=8, use_memmap=True):
    print(f"\n--- Phase 3 Smoke Test (envs={num_envs}, memmap={use_memmap}) ---")
    
    # 1. Create dummy data
    n_bars = 2000
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="1h")
    df = pd.DataFrame({
        "Open": np.random.randn(n_bars).cumsum() + 1.10,
        "High": np.random.randn(n_bars).cumsum() + 1.12,
        "Low": np.random.randn(n_bars).cumsum() + 1.08,
        "Close": np.random.randn(n_bars).cumsum() + 1.10,
        "Volume": np.random.randint(100, 1000, n_bars),
        "avg_spread": np.random.uniform(0.0001, 0.0002, n_bars),
        "time_delta_s": np.full(n_bars, 3600.0),
    }, index=dates)
    
    # Add dummy feature cols for FeatureEngine buffer
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # 2. Prepare bars
    bars_arr = RuntimeGymEnv._frame_to_bars(df)
    
    if use_memmap:
        import tempfile
        mmap_path = os.path.join(tempfile.gettempdir(), "smoke_bars.dat")
        mmap_arr = np.memmap(mmap_path, dtype=bars_arr.dtype, mode='w+', shape=bars_arr.shape)
        mmap_arr[:] = bars_arr[:]
        mmap_arr.flush()
        bars_to_use = mmap_arr
        print(f"Using shared memmap at {mmap_path}")
    else:
        bars_to_use = bars_arr
        print("Using per-worker data copy (High RAM)")

    # 3. Dummy scaler & action map
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLS])
    action_map = build_simple_action_map(sl_value=1.0, tp_value=1.0)

    # 4. Spawn VecEnv
    print(f"Spawning {num_envs} workers...")
    env_fns = [make_env(i, df, bars_to_use, scaler, action_map) for i in range(num_envs)]
    
    start_time = time.time()
    try:
        venv = SubprocVecEnv(env_fns)
        print(f"VecEnv spawned in {time.time() - start_time:.2f}s")
        
        # 5. Run steps
        print("Running 100 steps...")
        obs = venv.reset()
        for i in range(100):
            actions = np.random.randint(0, len(action_map), size=num_envs)
            obs, rewards, dones, infos = venv.step(actions)
            if i % 20 == 0:
                print(f"  Step {i} OK")
        
        print("Smoke Test PASSED!")
    except Exception as e:
        print(f"Smoke Test FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'venv' in locals():
            venv.close()

if __name__ == "__main__":
    # Test with memmap
    run_smoke_test(num_envs=4, use_memmap=True)
