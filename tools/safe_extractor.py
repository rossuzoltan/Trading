import ast
import os

EXTRACTIONS = {
    "train_curriculum.py": [
        "get_current_slippage_pips", "get_current_phase", "get_current_ent_coef",
        "get_current_participation_bonus", "get_final_slippage_pips", 
        "is_participation_bonus_active", "build_train_env_recovery_config",
        "LegacyCurriculumCallback", "CurriculumCallback"
    ],
    "train_callbacks.py": [
        "SaveVecNormalizeCallback", "AdaptiveKLLearningRateCallback", 
        "EnhancedLoggingCallback", "TrainingHeartbeatCallback"
    ],
    "train_diagnostics.py": [
        "aggregate_training_diagnostics", "TrainingDiagnosticsCallback"
    ],
    "train_failfast.py": [
        "FullPathEvalCallback", "_truth_run_fail_fast_eligible",
        "_truth_run_fail_fast_condition", "_overtrade_negative_edge_condition",
        "run_baseline_research_gate"
    ],
    "train_bootstrap.py": [
        "build_execution_cost_profile", "build_reward_profile", 
        "_install_maskable_fast_masking_patch"
    ]
}

HEADER = {
    "train_curriculum.py": "from __future__ import annotations\nimport os\nfrom typing import Any\nfrom stable_baselines3.common.callbacks import BaseCallback\nfrom train_config import *\n\n",
    "train_callbacks.py": "from __future__ import annotations\nimport os\nimport time\nimport json\nimport logging\nimport pandas as pd\nfrom pathlib import Path\nfrom collections import deque\nfrom typing import Any\nfrom stable_baselines3.common.callbacks import BaseCallback\nfrom train_config import *\n\n",
    "train_diagnostics.py": "from __future__ import annotations\nimport json\nfrom datetime import datetime, timezone\nimport numpy as np\nfrom stable_baselines3.common.callbacks import BaseCallback\nfrom collections import deque\nfrom train_config import *\n\n",
    "train_failfast.py": "from __future__ import annotations\nimport os\nimport math\nfrom typing import Any\nfrom stable_baselines3.common.callbacks import BaseCallback\nfrom train_config import *\n\n",
    "train_bootstrap.py": "from __future__ import annotations\nfrom typing import Any\nfrom train_config import *\nimport numpy as np\n\n"
}

def get_imports(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
    return "\n".join(imports) + "\n\n"

def main():
    agent_path = r'c:\dev\trading\train_agent.py'
    
    with open(agent_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
        lines = source_code.split('\n')
        
    tree = ast.parse(source_code)
    
    # Map node name to line ranges
    node_spans = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node_spans[node.name] = (node.lineno - 1, node.end_lineno)
            
    # Also collect all import lines so we can violently inject them into the child files 
    # to guarantee they compile cleanly without circular references to train_agent
    global_imports = get_imports(agent_path)

    # For each extraction, build the file and remove from agent
    removed_intervals = []
    
    for filename, objects in EXTRACTIONS.items():
        out_path = os.path.join(r'c:\dev\trading', filename)
        
        # Build file
        content = HEADER[filename]
        content += global_imports
        
        for obj in objects:
            if obj in node_spans:
                start, end = node_spans[obj]
                # Grab decorators too if any exist above the def. 
                # AST lineno points to the decorator!
                chunk = lines[start:end]
                content += "\n".join(chunk) + "\n\n"
                removed_intervals.append((start, end))
            else:
                print(f"Warning: {obj} not found!")

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created {filename} with {len(objects)} objects.")

    # Now filter train_agent.py
    new_lines = []
    for i, line in enumerate(lines):
        in_removed = any(start <= i < end for start, end in removed_intervals)
        if not in_removed:
            new_lines.append(line)
            
    # Inject imports back to the top of train_agent
    import_injections = [f"from {f[:-3]} import *" for f in EXTRACTIONS.keys()]
    
    # Find safe place to inject
    inject_idx = 0
    for i, line in enumerate(new_lines):
        if line.startswith("from train_config import *"):
            inject_idx = i + 1
            break
            
    new_lines = new_lines[:inject_idx] + import_injections + new_lines[inject_idx:]
    
    with open(agent_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(new_lines))
        
    print(f"Patched train_agent.py. Reduced to {len(new_lines)} lines.")

if __name__ == '__main__':
    main()
