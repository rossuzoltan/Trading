"""diagnose_losses.py
======================
Phase 3 loss diagnostic tool. Loads execution audit logs or an OOS replay report,
decomposes the loss drivers (cost drag, holding period distributions, long/short
pnl, forced closes, and churn ratio), and outputs a markdown summary table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def analyze_trade_log(trade_log: list[dict[str, Any]]) -> dict[str, Any]:
    if not trade_log:
        return {"error": "Trade log is empty."}
        
    trades = trade_log
    count = len(trades)
    
    gross_pnl_usd = sum(float(t.get("gross_pnl_usd", 0.0)) for t in trades)
    net_pnl_usd = sum(float(t.get("net_pnl_usd", 0.0)) for t in trades)
    total_cost_usd = sum(float(t.get("transaction_cost_usd", 0.0)) for t in trades)
    commissions = sum(float(t.get("commission_usd", 0.0)) for t in trades)
    spread_slippage = sum(float(t.get("spread_slippage_cost_usd", 0.0)) for t in trades)
    
    long_trades = [t for t in trades if int(t.get("direction", 1)) > 0]
    short_trades = [t for t in trades if int(t.get("direction", 1)) < 0]
    
    long_pnl = sum(float(t.get("net_pnl_usd", 0.0)) for t in long_trades)
    short_pnl = sum(float(t.get("net_pnl_usd", 0.0)) for t in short_trades)
    
    forced_closes = [t for t in trades if bool(t.get("forced_close", False))]
    forced_pnl = sum(float(t.get("net_pnl_usd", 0.0)) for t in forced_closes)
    
    holding_lt_5 = [t for t in trades if int(t.get("holding_bars", 0)) < 5]
    holding_5_20 = [t for t in trades if 5 <= int(t.get("holding_bars", 0)) <= 20]
    holding_gt_20 = [t for t in trades if int(t.get("holding_bars", 0)) > 20]
    
    pnl_lt_5 = sum(float(t.get("net_pnl_usd", 0.0)) for t in holding_lt_5)
    pnl_5_20 = sum(float(t.get("net_pnl_usd", 0.0)) for t in holding_5_20)
    pnl_gt_20 = sum(float(t.get("net_pnl_usd", 0.0)) for t in holding_gt_20)

    # Churn / reversals (approximate heuristic: holding_bars < 5 usually implies rapid reversal or jitter)
    rapid_reversal_count = len(holding_lt_5)
    rapid_reversal_cost = sum(float(t.get("transaction_cost_usd", 0.0)) for t in holding_lt_5)
    
    drivers = [
        {"name": "Total Cost Drag", "impact": total_cost_usd, "pct_of_gross": _safe_div(total_cost_usd, abs(gross_pnl_usd)) * 100},
        {"name": "Short-lived Trades (Churn)", "impact": abs(pnl_lt_5), "pct_of_gross": _safe_div(abs(pnl_lt_5), abs(gross_pnl_usd)) * 100},
        {"name": "Forced Closes", "impact": abs(forced_pnl), "pct_of_gross": _safe_div(abs(forced_pnl), abs(gross_pnl_usd)) * 100},
    ]
    
    drivers.sort(key=lambda d: d["impact"], reverse=True)
    top_driver = drivers[0]

    return {
        "count": count,
        "gross_pnl_usd": gross_pnl_usd,
        "net_pnl_usd": net_pnl_usd,
        "cost_drag_usd": total_cost_usd,
        "commissions_usd": commissions,
        "spread_slippage_usd": spread_slippage,
        "cost_per_trade_usd": _safe_div(total_cost_usd, count),
        
        "long_count": len(long_trades),
        "long_net_pnl_usd": long_pnl,
        "short_count": len(short_trades),
        "short_net_pnl_usd": short_pnl,
        
        "forced_count": len(forced_closes),
        "forced_pnl_usd": forced_pnl,
        
        "hold_lt_5_count": len(holding_lt_5),
        "hold_lt_5_pnl_usd": pnl_lt_5,
        "hold_5_20_count": len(holding_5_20),
        "hold_5_20_pnl_usd": pnl_5_20,
        "hold_gt_20_count": len(holding_gt_20),
        "hold_gt_20_pnl_usd": pnl_gt_20,
        
        "rapid_reversal_frac": _safe_div(rapid_reversal_count, count),
        "rapid_reversal_cost": rapid_reversal_cost,
        "top_drivers": drivers[:3],
    }


def format_markdown_report(symbol: str, analysis: dict[str, Any]) -> str:
    if "error" in analysis:
        return f"# Loss Diagnostics: {symbol}\n\n**Error:** {analysis['error']}"
        
    out = [
        f"# Loss Diagnostics: {symbol}",
        "",
        "## 1. High-Level Economics",
        f"- **Trade Count:** {analysis['count']}",
        f"- **Gross PnL:** ${analysis['gross_pnl_usd']:.2f}",
        f"- **Net PnL:** ${analysis['net_pnl_usd']:.2f}",
        f"- **Total Cost Drag:** ${analysis['cost_drag_usd']:.2f} ({analysis['commissions_usd']:.2f} comm / {analysis['spread_slippage_usd']:.2f} spr)",
        f"- **Cost Per Trade:** ${analysis['cost_per_trade_usd']:.2f}",
        "",
        "## 2. PnL by Direction",
        f"- **Long:** {analysis['long_count']} trades, ${analysis['long_net_pnl_usd']:.2f} net",
        f"- **Short:** {analysis['short_count']} trades, ${analysis['short_net_pnl_usd']:.2f} net",
        "",
        "## 3. PnL by Holding Duration",
        f"- **< 5 bars (Churn):** {analysis['hold_lt_5_count']} trades, ${analysis['hold_lt_5_pnl_usd']:.2f} net",
        f"- **5 - 20 bars:** {analysis['hold_5_20_count']} trades, ${analysis['hold_5_20_pnl_usd']:.2f} net",
        f"- **> 20 bars:** {analysis['hold_gt_20_count']} trades, ${analysis['hold_gt_20_pnl_usd']:.2f} net",
        "",
        "## 4. Forced Exits",
        f"- **Forced Closes:** {analysis['forced_count']} trades",
        f"- **Forced PnL:** ${analysis['forced_pnl_usd']:.2f} net",
        "",
        "## 5. Top Loss Drivers",
    ]
    
    for i, driver in enumerate(analysis["top_drivers"], 1):
        out.append(f"{i}. **{driver['name']}**: ${driver['impact']:.2f} (~{driver['pct_of_gross']:.1f}% of gross magnitude)")
        
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Decompose execution losses and generate markdown report.")
    parser.add_argument("--symbol", default="EURUSD")
    args = parser.parse_args()
    
    # We look for a replay report, or an execution audit jsonl.
    report_path = Path("models") / f"replay_report_{str(args.symbol).lower()}.json"
    audit_path = Path("models") / f"execution_audit_{str(args.symbol).lower()}.jsonl"
    
    trade_log = []
    
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        # we can't get the raw trade log from the replay report easily if not saved.
        print(f"Loaded replay report details, but full execution logs needed.")
        
    if audit_path.exists():
        lines = audit_path.read_text(encoding="utf-8").strip().split()
        for line in lines:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "position_closed" or event.get("type") == "position_closed":
                # transform to trade
                trade_log.append(event)
                
    if not trade_log:
        analysis = {"error": f"No valid closed trades found in {audit_path}"}
    else:
        analysis = analyze_trade_log(trade_log)
        
    md_content = format_markdown_report(args.symbol, analysis)
    
    out_path = Path("docs") / "phase3_loss_diagnostics.md"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text(md_content, encoding="utf-8")
    
    print(md_content)
    print(f"\nReport written to {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
