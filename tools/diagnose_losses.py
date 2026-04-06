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


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL in {path} line {line_number}: {exc}") from exc
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _trade_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    if "net_pnl_usd" in record and "holding_bars" in record:
        return dict(record)
    event_name = str(record.get("event") or record.get("type") or "").strip().lower()
    if event_name in {"position_closed", "trade_closed"} and "net_pnl_usd" in record:
        return dict(record)
    return None


def _metrics_summary_from_report(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("trade_metrics", "replay_metrics"):
        metrics = payload.get(key)
        if isinstance(metrics, dict) and metrics:
            return dict(metrics)
    return None


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


def load_loss_context(*, symbol: str, report_path: Path, audit_path: Path) -> dict[str, Any]:
    trade_log: list[dict[str, Any]] = []
    trade_metrics: dict[str, Any] | None = None

    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        raw_trade_log = payload.get("trade_log")
        if isinstance(raw_trade_log, list):
            trade_log.extend([dict(item) for item in raw_trade_log if isinstance(item, dict)])
        trade_metrics = _metrics_summary_from_report(payload)

    if audit_path.exists():
        for record in _load_jsonl_records(audit_path):
            trade = _trade_from_record(record)
            if trade is not None:
                trade_log.append(trade)

    if trade_log:
        return {
            "symbol": symbol,
            "source": "trade_log",
            "trade_log": trade_log,
            "trade_metrics": trade_metrics,
        }

    if trade_metrics:
        return {
            "symbol": symbol,
            "source": "summary_only",
            "trade_metrics": trade_metrics,
        }

    return {
        "symbol": symbol,
        "source": "empty",
        "error": f"No closed-trade records found in {audit_path}",
    }


def format_markdown_report(symbol: str, analysis: dict[str, Any]) -> str:
    if analysis.get("source") == "summary_only":
        metrics = dict(analysis.get("trade_metrics", {}) or {})
        return "\n".join(
            [
                f"# Loss Diagnostics: {symbol}",
                "",
                "## Summary-Only View",
                "Closed-trade details were not available, so this report falls back to replay summary metrics.",
                f"- **Trade Count:** {int(metrics.get('trade_count', 0))}",
                f"- **Net PnL:** ${float(metrics.get('net_pnl_usd', 0.0)):.2f}",
                f"- **Profit Factor:** {float(metrics.get('profit_factor', 0.0)):.2f}",
                f"- **Expectancy:** ${float(metrics.get('expectancy_usd', 0.0)):.2f}",
                f"- **Win Rate:** {float(metrics.get('win_rate', 0.0)):.1%}",
                f"- **Average Hold Bars:** {float(metrics.get('avg_holding_bars', 0.0)):.2f}",
                "",
                "Per-trade loss decomposition requires a trade log or closed-trade audit records.",
            ]
        )
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
    
    context = load_loss_context(symbol=str(args.symbol).upper(), report_path=report_path, audit_path=audit_path)
    if context.get("source") == "trade_log":
        analysis = analyze_trade_log(list(context.get("trade_log", [])))
    elif context.get("source") == "summary_only":
        analysis = context
    else:
        analysis = {"error": context["error"]}
        
    md_content = format_markdown_report(args.symbol, analysis)
    
    out_path = Path("docs") / "phase3_loss_diagnostics.md"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text(md_content, encoding="utf-8")
    
    print(md_content)
    print(f"\nReport written to {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
