import time
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from runtime_gym_env import RuntimeGymEnv, RuntimeGymConfig
from feature_engine import FeatureEngine
from runtime_common import build_simple_action_map

def profile_env(steps=5000):
    data_path = "data/EURUSD_volbars_2000.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Preparing FeatureEngine...")
    engine = FeatureEngine()
    # Mocking a fit_transform for profiling
    df_feat, _ = engine.fit_transform(df)
    
    print("Initializing RuntimeGymEnv...")
    action_map = build_simple_action_map(sl_value=1.0, tp_value=1.0)
    config = RuntimeGymConfig(window_size=1)
    
    env = RuntimeGymEnv(
        symbol="EURUSD",
        bars_frame=df_feat,
        scaler=engine._scaler,
        action_map=action_map,
        config=config
    )
    
    print(f"Starting profile of {steps} steps...")
    env.reset()
    
    # Warmup
    for _ in range(100):
        env.step(env.action_space.sample())
    
    start_time = time.perf_counter()
    for i in range(steps):
        action = env.action_space.sample()
        # Handle both Gym versions
        result = env.step(action)
        if len(result) == 5:
            obs, reward, term, trunc, info = result
            done = term or trunc
        else:
            obs, reward, done, info = result
            
        if done:
            env.reset()
            
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    sps = steps / duration
    print("\n" + "="*30)
    print(f"Baseline Profiling Results")
    print("="*30)
    print(f"Total Steps: {steps}")
    print(f"Duration:    {duration:.4f}s")
    print(f"SPS:         {sps:.2f}")
    print("="*30)

if __name__ == "__main__":
    profile_env(5000)
