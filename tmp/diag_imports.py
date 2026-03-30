import sys
import os
import traceback

print("Testing imports...")
try:
    print("Importing numpy...")
    import numpy as np
    print("NumPy OK")
    
    print("Importing pandas...")
    import pandas as pd
    print("Pandas OK")
    
    print("Importing domain.models...")
    import domain.models
    print("Domain Models OK")
    
    print("Importing BAR_DTYPE from domain.models...")
    from domain.models import BAR_DTYPE
    print("BAR_DTYPE OK:", BAR_DTYPE)
    
    print("Importing runtime_common...")
    import runtime_common
    print("Runtime Common OK")
    
    print("Importing feature_engine...")
    import feature_engine
    print("Feature Engine OK")
    
    print("Importing RuntimeGymEnv from runtime_gym_env...")
    from runtime_gym_env import RuntimeGymEnv
    print("RuntimeGymEnv OK")
    
    print("ALL IMPORTS SUCCESSFUL!")
except Exception as e:
    print("\nFAIL!")
    traceback.print_exc()
    sys.exit(1)
