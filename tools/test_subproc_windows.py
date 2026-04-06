import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from runtime_gym_env import RuntimeGymEnv, RuntimeGymConfig
from runtime_common import build_simple_action_map
from feature_engine import FeatureEngine

def make_env(rank, df, bars, scaler, action_map):
    def _init():
        # Each worker needs a unique rank for observation window size variety
        return RuntimeGymEnv(
            symbol="EURUSD",
            bars_frame=df,
            bars=bars,
            scaler=scaler,
            action_map=action_map,
            config=RuntimeGymConfig(window_size=1)
        )

    return _init

if __name__ == "__main__":
    # Windows requires __main__ guard for SubprocVecEnv (spawn)
    
    data_path = "data/EURUSD_volbars_2000.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    print("Loading small slice of data...")
    df = pd.read_csv(data_path).iloc[:1000]
    
    print("Preparing FeatureEngine...")
    engine = FeatureEngine()
    # Mock/fit for test
    df_feat, _ = engine.fit_transform(df)
    
    print("Pre-converting bars...")
    bars = RuntimeGymEnv._frame_to_bars(df_feat)
    action_map = build_simple_action_map(sl_value=1.0, tp_value=1.0)
    
    num_envs = 2
    env_fns = [make_env(i, df_feat, bars, engine._scaler, action_map) for i in range(num_envs)]

    
    print(f"Initializing {num_envs} DummyVecEnv workers (Single-process)...")
    try:
        venv = DummyVecEnv(env_fns)
        venv.reset()
        venv.close()
        print("DummyVecEnv OK.")
    except Exception as e:
        print(f"DummyVecEnv Failure: {e}")
        sys.exit(1)

    print(f"Initializing {num_envs} SubprocVecEnv workers on Windows...")

    try:
        venv = SubprocVecEnv(env_fns)
        print("Resetting...")
        obs = venv.reset()
        print(f"Reset success! Obs shape: {obs.shape}")
        
        print("Stepping 10 times...")
        for _ in range(10):
            actions = [venv.action_space.sample() for _ in range(num_envs)]
            venv.step(actions)
        
        print("Closing...")
        venv.close()
        print("\n" + "="*40)
        print("SUCCESS: SubprocVecEnv is working on Windows!")
        print("="*40)
        
    except Exception as e:
        print("\n" + "!"*40)
        print(f"FAILURE: SubprocVecEnv error: {e}")
        import traceback
        traceback.print_exc()
        print("!"*40)
