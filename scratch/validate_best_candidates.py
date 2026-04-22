
import json
import subprocess
import os

def run_oos(symbol, config):
    cmd = [
        "python", "evaluate_oos.py",
        "--symbol", symbol,
        "--config", json.dumps(config)
    ]
    print(f"Running OOS for {symbol} with config: {config['params']}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running OOS for {symbol}: {result.stderr}")
        return None
    
    # The script prints the final result line with "OOS_SUMMARY:"
    for line in result.stdout.split('\n'):
        if "OOS_SUMMARY:" in line:
            return json.loads(line.replace("OOS_SUMMARY:", "").strip())
    return None

def main():
    best_results = []
    
    # Process GBPUSD Top 3
    with open('artifacts/optimization_report_GBPUSD_train.json', 'r') as f:
        gbp_data = json.load(f)
        gbp_top = [r for r in gbp_data.get('results', []) if r.get('status') == 'PASSED'][:3]
        
    for i, r in enumerate(gbp_top):
        oos_res = run_oos("GBPUSD", r['config'])
        if oos_res:
            oos_res['id'] = f"GBPUSD_TOP_{i+1}"
            best_results.append(oos_res)

    # Process EURUSD Top 3
    with open('artifacts/optimization_report_EURUSD_train.json', 'r') as f:
        eur_data = json.load(f)
        eur_top = [r for r in eur_data.get('results', []) if r.get('status') == 'PASSED'][:3]
        
    for i, r in enumerate(eur_top):
        oos_res = run_oos("EURUSD", r['config'])
        if oos_res:
            oos_res['id'] = f"EURUSD_TOP_{i+1}"
            best_results.append(oos_res)

    with open('artifacts/oos_validation_results.json', 'w') as f:
        json.dump(best_results, f, indent=2)
    
    print(f"\nSaved {len(best_results)} OOS results to artifacts/oos_validation_results.json")

if __name__ == "__main__":
    main()
