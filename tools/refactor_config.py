import os

def main():
    agent_path = r'c:\dev\trading\train_agent.py'
    config_path = r'c:\dev\trading\train_config.py'
    
    with open(agent_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # We want to extract lines starting from `def _resolve_training_experiment_profile...` 
    # to the line before `class WindowsStayAwakeGuard:`
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if line.startswith('def _resolve_training_experiment_profile('):
            start_idx = i
        elif line.startswith('class WindowsStayAwakeGuard:'):
            # The previous lines contain blanks. Let's step back a bit.
            end_idx = i
            break
            
    if start_idx == -1 or end_idx == -1:
        print("Could not find extraction boundaries!")
        return
        
    extraction = lines[start_idx:end_idx]
    
    # Check if there is anything that needs fixing in the extracted lines
    # e.g., imports
    config_content = [
        "from __future__ import annotations\n",
        "import os\n",
        "import copy\n",
        "from pathlib import Path\n",
        "from typing import Any\n",
        "from datetime import datetime, timezone\n",
        "from trading_config import (\n",
        "    DEFAULT_MIN_LEARNING_RATE,\n",
        "    DEFAULT_TARGET_KL,\n",
        "    DEFAULT_SLIPPAGE_START_PIPS,\n",
        "    DEFAULT_SLIPPAGE_END_PIPS,\n",
        "    DEFAULT_CHURN_MIN_HOLD_BARS,\n",
        "    DEFAULT_CHURN_ACTION_COOLDOWN,\n",
        "    DEFAULT_CHURN_PENALTY_USD,\n",
        "    DEFAULT_ENTRY_SPREAD_Z_LIMIT,\n",
        "    DEFAULT_REWARD_DOWNSIDE_RISK_COEF,\n",
        "    DEFAULT_REWARD_TURNOVER_COEF,\n",
        "    DEFAULT_REWARD_DRAWDOWN_COEF,\n",
        "    DEFAULT_REWARD_NET_RETURN_COEF,\n",
        "    resolve_bar_construction_ticks_per_bar,\n",
        ")\n\n"
    ]
    config_content.extend(extraction)
    
    # Look for the RECOVERY_CONFIG block after WindowsStayAwakeGuard
    # Let's see if we should extract that too. But for safety,
    # let's just do the massive block first.
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(config_content)
        
    print(f"Extracted {len(extraction)} lines to train_config.py")
    
    # Patch train_agent.py
    patched_agent = lines[:start_idx] + ["from train_config import *\n\n"] + lines[end_idx:]
    
    with open(agent_path, 'w', encoding='utf-8') as f:
        f.writelines(patched_agent)
        
    print("Patched train_agent.py")

if __name__ == '__main__':
    main()
