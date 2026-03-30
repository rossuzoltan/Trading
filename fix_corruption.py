import os

def fix_train_agent():
    path = r'c:\dev\trading\train_agent.py'
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Target lines 3006-3015 (0-indexed 3005-3014)
    # We want to replace the whole block with the correct one.
    start = 3005
    end = 3015
    
    new_block = [
        '        diagnostics["holdout_gate_passed"] = bool(not holdout_gate_blockers)\n',
        '        diagnostics["deploy_ready"] = bool(diagnostics["data_sufficiency_passed"] and diagnostics["holdout_gate_passed"])\n',
        '\n',
        '        if sharpe > best_observed_sharpe:\n',
        '            best_observed_sharpe = float(sharpe)\n',
        '            best_observed_summary = _build_promoted_training_diagnostics(\n',
        '                diagnostics,\n',
        '                run_id=run_id,\n',
        '                artifact_candidate_selected=False,\n',
        '                artifact_candidate_reason=(\n',
        '                    "Best-evaluated fold from this run did not meet deployment artifact criteria."\n',
        '                    if not diagnostics["deploy_ready"]\n',
        '                    else "Best-evaluated fold from this run is deployment eligible but was not selected as the canonical candidate."\n',
        '                ),\n',
        '            )\n'
    ]
    
    # Basic verification that we are at the right spot
    if 'holdout_gate_passed' in lines[start] or 'holdout_gate_blockers' in lines[start]:
        lines[start:end+1] = new_block
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("Fixed train_agent.py")
    else:
        print(f"FAILED: 'holdout_gate_passed' not found at line {start+1}. Content: {lines[start]}")

def fix_runtime_gym_env():
    path = r'c:\dev\trading\runtime_gym_env.py'
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Target compute_participation_bonus logic
    # Find line 122 area
    start = 121 # 0-indexed for line 122
    end = 126   # up to 127
    
    new_logic = [
        '    mode = str(pcfg.get("mode", "entry")).lower()\n',
        '    if mode == "per_bar":\n',
        '        if int(new_position) != 0:\n',
        '            return float(pcfg.get("bonus_value", 0.0)) / 10.0\n',
        '        return 0.0\n',
        '\n',
        '    entry_happened = int(prev_position) == 0 and int(new_position) != 0\n',
        '    if bool(pcfg.get("only_from_flat", True)) and not entry_happened:\n',
        '        return 0.0\n',
        '    if not entry_happened:\n',
        '        return 0.0\n',
        '    return float(pcfg.get("bonus_value", 0.0))\n'
    ]
    
    if 'entry_happened' in lines[start]:
        lines[start:end+1] = new_logic
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("Fixed runtime_gym_env.py")
    else:
        # Search for entry_happened
        found = False
        for i, line in enumerate(lines):
            if 'entry_happened = int(prev_position) == 0' in line:
                lines[i:i+6] = new_logic
                with open(path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                print(f"Fixed runtime_gym_env.py at line {i+1}")
                found = True
                break
        if not found:
            print("FAILED: entry_happened not found in runtime_gym_env.py")

if __name__ == "__main__":
    fix_train_agent()
    fix_runtime_gym_env()
