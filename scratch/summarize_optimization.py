import json
import os
import pandas as pd

def summarize_results(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    results = data.get("results", [])
    if not results:
        print("No results found in JSON.")
        return
        
    df = pd.DataFrame(results)
    
    # Filter for passed candidates
    passed = df[df['status'] == 'PASSED'].copy()
    
    print(f"\n--- Summary for {os.path.basename(json_path)} ---")
    print(f"Total variants: {len(df)}")
    print(f"Passed variants: {len(passed)}")
    
    if len(passed) > 0:
        # Check Long/Short distribution
        passed['long_trades'] = passed.get('long_trades', 0)
        passed['short_trades'] = passed.get('short_trades', 0)
        
        top = passed.sort_values(by='net_pnl', ascending=False).head(5)
        print("\nTop 5 Candidates by Net PnL:")
        for _, row in top.iterrows():
            params = row['config']['params']
            print(f"Rule: {row['config']['rule_family']} | PnL: ${row['net_pnl']:.2f} | PF: {row['pf']:.2f} | Trades: {row['trades']} (L:{row.get('long_trades',0)}/S:{row.get('short_trades',0)}) | Signals: {row.get('signal_longs',0)}/{row.get('signal_shorts',0)}")
    else:
        print("\nNo candidates passed constraints.")
        # Why rejected?
        rejections = df[df['status'] == 'REJECTED']['reject_reason'].value_counts()
        print("\nTop Rejection Reasons:")
        print(rejections.head(3))

if __name__ == "__main__":
    summarize_results("artifacts/optimization_report_EURUSD_train.json")
    summarize_results("artifacts/optimization_report_GBPUSD_train.json")
