import json, sys
sys.stdout.reconfigure(encoding='utf-8')

def analyze(path, symbol):
    with open(path) as f:
        data = json.load(f)
    results = data.get('results', [])
    print(f'\n=== {symbol} DEEP ANALYSIS ===')
    print(f'Total evaluated: {len(results)}')
    families = sorted(set(r["config"]["rule_family"] for r in results))
    grand_long = grand_short = 0
    passed_total = 0
    for family in families:
        f_res = [r for r in results if r["config"]["rule_family"] == family]
        exploratory = [r for r in f_res if r.get('confidence_band') == 'exploratory']
        stable = [r for r in f_res if r.get('confidence_band') == 'stable']
        rejected = [r for r in f_res if r.get('status') == 'REJECTED' and r.get('confidence_band') not in ['exploratory','stable']]
        long_sum = sum(r.get('long_trades', 0) for r in f_res)
        short_sum = sum(r.get('short_trades', 0) for r in f_res)
        grand_long += long_sum
        grand_short += short_sum
        best = max([r for r in f_res if r.get('net_pnl') is not None], key=lambda x: x.get('net_pnl',-9999), default=None)
        passed_total += len(stable) + len(exploratory)
        zero_long_tag = '  !! ZERO LONGS - structural' if long_sum == 0 else ''
        print(f'  [{family}] stable={len(stable)} exploratory={len(exploratory)} rejected={len(rejected)}{zero_long_tag}')
        print(f'    Longs={long_sum}  Shorts={short_sum}')
        if best:
            reason = best.get('reject_reason') or 'PASSED'
            print(f'    Best: PnL={best["net_pnl"]:.2f} PF={best["pf"]:.2f} trades={best["trades"]}(L:{best.get("long_trades",0)} S:{best.get("short_trades",0)}) band={best.get("confidence_band")} reason={reason}')
    print(f'\nTOTALS: Long signals={grand_long} | Short signals={grand_short} | Passed total={passed_total}')
    if grand_long + grand_short > 0:
        print(f'  Long%={grand_long/(grand_long+grand_short)*100:.1f}%  Short%={grand_short/(grand_long+grand_short)*100:.1f}%')
    else:
        print('  !! ZERO TOTAL LONG OR SHORT SIGNALS ACROSS ALL VARIANTS')
    pnls = sorted([r.get('net_pnl', 0) for r in results if r.get('net_pnl') is not None], reverse=True)
    print(f'  Top5 PnL:    {["{:.2f}".format(p) for p in pnls[:5]]}')
    print(f'  Bottom5 PnL: {["{:.2f}".format(p) for p in pnls[-5:]]}')
    positive = sum(1 for p in pnls if p > 0)
    print(f'  Positive variants: {positive}/{len(pnls)}, Negative: {len(pnls)-positive}/{len(pnls)}')

analyze('artifacts/optimization_report_GBPUSD_train.json', 'GBPUSD')
analyze('artifacts/optimization_report_EURUSD_train.json', 'EURUSD')
