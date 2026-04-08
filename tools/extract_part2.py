import re
import os

def main():
    agent_path = r'c:\dev\trading\train_agent.py'
    config_path = r'c:\dev\trading\train_config.py'
    
    with open(agent_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # We want to extract lines matching the various loose globals and recovery configs.
    # Let's read through and collect indices of what we want to move.
    
    # 1. Recovery config block: From _timed_recovery_step to the end of STAGE_A_RECOVERY_TARGETS
    start_recovery = -1
    end_recovery = -1
    
    for i, line in enumerate(lines):
        if line.startswith('def _timed_recovery_step'):
            start_recovery = i
        elif line.startswith('# ── Purged walk-forward config ──'):
            end_recovery = i
            break
            
    # 2. Walk-forward config block
    start_vf = end_recovery
    end_vf = -1
    for i in range(start_vf, len(lines)):
        if line.startswith('CURRENT_TRAINING_RUN_PATH'):
            # The next lines are `log = logging.getLogger ...` so stop here.
            end_vf = i + 1
            break
        line = lines[i]

    if start_recovery == -1 or end_recovery == -1 or end_vf == -1:
        print("Couldn't find target blocks.")
        return

    recovery_chunk = lines[start_recovery:end_vf]
    
    # Add to config
    with open(config_path, 'a', encoding='utf-8') as f:
        f.writelines(["\n\n"])
        f.writelines(recovery_chunk)
        
    print(f"Appended {len(recovery_chunk)} lines to config.")
    
    # Patch agent: remove the chunk from agent
    patched = lines[:start_recovery] + lines[end_vf:]
    with open(agent_path, 'w', encoding='utf-8') as f:
        f.writelines(patched)
        
    print("Patched agent.")

if __name__ == '__main__':
    main()
