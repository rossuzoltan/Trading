import json
from collections import Counter

with open("artifacts/optimization_report_GBPUSD_train.json", "r") as f:
    data = json.load(f)

results = data.get("results", [])

print("=== GBPUSD TRAIN DEEP LOGGER ANALYSIS ===")
print(f"Total candidates evaluated: {len(results)}")

# Group by rule family
families = set(r["config"]["rule_family"] for r in results)

for family in families:
    f_res = [r for r in results if r["config"]["rule_family"] == family]
    passed = [r for r in f_res if r.get("status") == "PASSED" or r.get("confidence_band") in ["exploratory", "stable"]]
    rejected = [r for r in f_res if r not in passed]
    
    print(f"\n--- {family.upper()} ---")
    print(f"Evaluated: {len(f_res)}, Passed (Exploratory/Stable): {len(passed)}, Rejected: {len(rejected)}")
    
    # Analysis of trade counts and direction
    total_longs = sum(r.get("long_trades", 0) for r in f_res)
    total_shorts = sum(r.get("short_trades", 0) for r in f_res)
    
    print(f"Total Long signals fired across all variants: {total_longs}")
    print(f"Total Short signals fired across all variants: {total_shorts}")
    
    # Max PnL among rejected (maybe they were profitable but failed constraints?)
    if rejected:
        try:
            max_pnl_rej = max(rejected, key=lambda x: x.get("net_pnl", -9999))
            print(f"Highest PnL among REJECTED: ${max_pnl_rej.get('net_pnl', 0):.2f}")
            print(f"  Reason: {max_pnl_rej.get('reject_reason')}")
            print(f"  Trades: {max_pnl_rej.get('trades')} (L: {max_pnl_rej.get('long_trades')}, S: {max_pnl_rej.get('short_trades')})")
        except:
            pass

    # Most common rejection reasons
    reasons = [r.get("reject_reason") for r in rejected]
    # Simplify reasons for counting
    simple_reasons = []
    for r in reasons:
        if not r: continue
        parts = r.split(" | ")
        simple_reasons.extend([p.split(" (")[0] for p in parts])
    
    top_reasons = Counter(simple_reasons).most_common(3)
    print(f"Top rejection reasons: {top_reasons}")
