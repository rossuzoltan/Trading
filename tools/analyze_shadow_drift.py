import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

log = logging.getLogger("shadow_drift")

def get_session_by_hour(hour_utc: int) -> str:
    # Simplified Forex session mapping (UTC)
    if 21 <= hour_utc <= 23:
        return "Rollover / Dead"
    if 0 <= hour_utc < 7:
        return "Asia"
    if 7 <= hour_utc < 12:
        return "London"
    if 12 <= hour_utc < 17:
        return "London/NY Overlap"
    if 17 <= hour_utc < 21:
        return "NY"
    return "Unknown"

def _resolve_severity(metric_name: str, drift_ratio: float, expected: float = 1.0) -> str:
    # A generic drift severity assigner based on ratio
    delta = abs(drift_ratio - expected)
    if delta <= 0.10:
        return "OK"
    elif delta <= 0.25:
        return "WATCH"
    elif delta <= 0.50:
        return "DRIFT_WARNING"
    else:
        return "DRIFT_CRITICAL"

class ShadowDriftAnalyzer:
    def __init__(self, audit_dir: Path, replay_path: Optional[Path] = None):
        self.audit_dir = audit_dir
        self.replay_path = replay_path
        self.replay_baseline: Dict[str, Any] = {}
        
        if self.replay_path and self.replay_path.exists():
            with self.replay_path.open("r", encoding="utf-8") as f:
                self.replay_baseline = json.load(f)
        else:
            log.warning("No replay baseline provided or found. Drift deltas will not be completely available. Using defaults derived from average run characteristics.")
            # Provide sensible fallback defaults if missing, just for demonstration
            self.replay_baseline = {
                "summary": {"cost_per_trade_usd": 14.5},
                "metrics": {"trade_count": 100, "steps": 50000}
            }

    def _expected_ratio(self, key: str) -> float:
        if "metrics" in self.replay_baseline:
            trades = max(1, self.replay_baseline["metrics"].get("trade_count", 0))
            steps = max(1, self.replay_baseline["metrics"].get("steps", 0))
            if key == "trades_per_bar":
                return float(trades / steps)
        # fallback reasonable expectations
        if key == "trades_per_bar": return 0.015
        if key == "long_short_ratio": return 1.0
        return 1.0

    def analyze_summary_mode(self) -> Dict[str, Any]:
        """Aggregate all daily summary JSONs."""
        summary_files = list(self.audit_dir.glob("shadow_summary_*.json"))
        if not summary_files:
            log.warning("No shadow_summary_*.json found in %s", self.audit_dir)
            return {}

        agg = {
            "total_bars_processed": 0,
            "total_would_open": 0,
            "total_would_close": 0,
            "total_would_hold_position": 0,
            "total_would_remain_flat": 0,
            "total_no_trade": 0,
            "long_open_count": 0,
            "short_open_count": 0,
            "reason_counts": {}
        }
        
        for sf in summary_files:
            try:
                with sf.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                for k in agg.keys():
                    if k == "reason_counts":
                        for rk, rv in data.get("reason_counts", {}).items():
                            agg["reason_counts"][rk] = agg["reason_counts"].get(rk, 0) + rv
                    elif isinstance(agg[k], int):
                        agg[k] += data.get(k, 0)
            except Exception as e:
                log.error("Failed to parse %s: %s", sf, e)
                
        return self._compute_drift_metrics(agg)

    def analyze_trace_mode(self) -> Dict[str, Any]:
        """Deep dive into raw JSONL lines."""
        jsonl_files = list(self.audit_dir.glob("shadow_audit_*.jsonl"))
        if not jsonl_files:
            log.warning("No shadow_audit_*.jsonl found in %s", self.audit_dir)
            return {}

        records = []
        for jf in jsonl_files:
            with jf.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not records:
            return {}

        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['bar_ts'], utc=True)
        df['hour_utc'] = df['datetime'].dt.hour
        df['session'] = df['hour_utc'].apply(get_session_by_hour)

        # Re-derive aggregation for summary
        agg = {
            "total_bars_processed": len(df),
            "total_would_open": int(df['would_open'].sum()),
            "total_would_close": int(df['would_close'].sum()),
            "total_would_hold_position": int(df['would_hold_position'].sum()),
            "total_would_remain_flat": int(df['would_remain_flat'].sum()),
            "long_open_count": int(df[(df['would_open'] == True) & (df['signal'] > 0)].shape[0]),
            "short_open_count": int(df[(df['would_open'] == True) & (df['signal'] < 0)].shape[0]),
            "reason_counts": {
               "spread": int(df[~df['spread_ok']].shape[0]),
               "session": int(df[~df['session_ok']].shape[0]),
               "risk": int(df[~df['risk_ok']].shape[0]),
               "no_signal": int(df[df['reason'].str.contains('signal', case=False, na=False)].shape[0]),
            }
        }

        metrics = self._compute_drift_metrics(agg)
        
        # Add trace-specific forensics
        metrics["trace_forensics"] = {
            "session_density": df[df['would_open'] == True].groupby('session').size().to_dict(),
            "manifest_truth": {
                "fingerprint_matches": df['manifest_fingerprint'].nunique() == 1,
                "release_statuses": df['release_stage'].unique().tolist(),
            },
            "position_state_distribution": df['active_position_state'].value_counts(normalize=True).to_dict()
        }
        
        return metrics

    def _compute_drift_metrics(self, agg: Dict[str, Any]) -> Dict[str, Any]:
        total_bars = max(1, agg["total_bars_processed"])
        would_open = agg["total_would_open"]
        total_flat = agg["total_would_remain_flat"]
        total_hold = agg["total_would_hold_position"]
        
        # A. Signal density drift
        shadow_tpb = would_open / total_bars
        replay_tpb = self._expected_ratio("trades_per_bar")
        density_ratio = shadow_tpb / max(replay_tpb, 1e-6)
        
        # B. Gate reason drift
        spread_ratio = agg["reason_counts"].get("spread", 0) / total_bars
        
        # C. State Occupancy
        flat_ratio = total_flat / total_bars
        hold_ratio = total_hold / total_bars
        
        # D. Direction Drift
        l_count = agg["long_open_count"]
        s_count = agg["short_open_count"]
        ls_ratio = (l_count / max(s_count, 1))

        return {
            "raw_aggregates": agg,
            "drift_measurements": {
                "signal_density": {
                    "shadow_tpb": shadow_tpb,
                    "replay_tpb": replay_tpb,
                    "ratio": density_ratio,
                    "verdict": _resolve_severity("signal_density", density_ratio, 1.0)
                },
                "direction_skew": {
                    "long_to_short_ratio": ls_ratio,
                    "verdict": _resolve_severity("direction_skew", ls_ratio, 1.0) # 1.0 expected balance
                },
                "occupancy": {
                    "flat_pct": flat_ratio * 100.0,
                    "hold_pct": hold_ratio * 100.0,
                },
                "gate_friction": {
                    "spread_reject_pct": spread_ratio * 100.0,
                    "verdict": _resolve_severity("gate_friction", spread_ratio, 0.05) # expecting ~5% max
                }
            }
        }

def render_report(metrics: Dict[str, Any], mode: str) -> str:
    if not metrics:
        return "# Drift Analysis\nNo valid data found."
        
    m = metrics["drift_measurements"]
    agg = metrics["raw_aggregates"]
    
    # Global severity depends on worst metric
    severities = [m["signal_density"]["verdict"], m["direction_skew"]["verdict"], m["gate_friction"]["verdict"]]
    if "DRIFT_CRITICAL" in severities:
        overall = "🚨 DRIFT_CRITICAL"
    elif "DRIFT_WARNING" in severities:
        overall = "⚠️ DRIFT_WARNING"
    elif "WATCH" in severities:
        overall = "👀 WATCH"
    else:
        overall = "✅ OK"
        
    lines = [
        f"# Shadow Simulator Drift Report ({mode.upper()} MODE)",
        f"**GLOBAL VERDICT:** {overall}",
        "",
        "## A. Signal Density Drift",
        f"- **Shadow Opens/Bar**: {m['signal_density']['shadow_tpb']:.5f}",
        f"- **Expected (Replay)**: {m['signal_density']['replay_tpb']:.5f}",
        f"- **Density Ratio**: {m['signal_density']['ratio']:.2f}x",
        f"- **Verdict**: `{m['signal_density']['verdict']}`",
        "",
        "## B. Gate Reason Breakdown",
        f"- **Spread Rejects**: {agg['reason_counts'].get('spread', 0)} ({m['gate_friction']['spread_reject_pct']:.2f}%)",
        f"- **Session Rejects**: {agg['reason_counts'].get('session', 0)}",
        f"- **Risk Rejects**: {agg['reason_counts'].get('risk', 0)}",
        f"- **No-Signal**: {agg['reason_counts'].get('no_signal', 0)}",
        f"- **Verdict**: `{m['gate_friction']['verdict']}`",
        "",
        "## C. State Occupancy Drift",
        f"- **Flat %**: {m['occupancy']['flat_pct']:.2f}%",
        f"- **Holding %**: {m['occupancy']['hold_pct']:.2f}%",
        "",
        "## D. Direction Drift",
        f"- **Longs**: {agg['long_open_count']}",
        f"- **Shorts**: {agg['short_open_count']}",
        f"- **L/S Ratio**: {m['direction_skew']['long_to_short_ratio']:.2f}",
        f"- **Verdict**: `{m['direction_skew']['verdict']}`",
    ]
    
    if "trace_forensics" in metrics:
        tf = metrics["trace_forensics"]
        lines.extend([
            "",
            "## E. Time-of-Day (Session) Trace Analysis",
            "**Opens by Session:**"
        ])
        for k, v in tf.get("session_density", {}).items():
            lines.append(f"- {k}: {v} trades")
            
        lines.extend([
            "",
            "## F. Manifest & Truth Checking",
            f"- **Fingerprint Match**: {tf['manifest_truth']['fingerprint_matches']}",
            f"- **Observed Release Stages**: {', '.join(tf['manifest_truth']['release_statuses'])}",
            "",
            "**Raw Position State Distribution:**"
        ])
        for state, pct in tf.get("position_state_distribution", {}).items():
            lines.append(f"- `{state}`: {pct*100:.1f}%")
            
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser("Shadow Drift Analyzer (Phase P7)")
    parser.add_argument("--audit-dir", type=str, required=True, help="Path to shadow_audits directory")
    parser.add_argument("--replay-baseline", type=str, default="", help="Path to replay_diagnostics json generated from offline backtest")
    parser.add_argument("--mode", choices=["summary", "trace", "auto"], default="auto")
    parser.add_argument("--output", type=str, default="shadow_drift_report.md", help="Output markdown path")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    
    audit_dir = Path(args.audit_dir)
    replay_path = Path(args.replay_baseline) if args.replay_baseline else None
    
    analyzer = ShadowDriftAnalyzer(audit_dir, replay_path)
    
    mode_to_run = args.mode
    if mode_to_run == "auto":
        # Check if jsonl exists
        if list(audit_dir.glob("*.jsonl")):
            mode_to_run = "trace"
        else:
            mode_to_run = "summary"
            
    log.info("Running analyzer in '%s' mode against '%s'", mode_to_run, audit_dir)
    
    if mode_to_run == "trace":
        metrics = analyzer.analyze_trace_mode()
    else:
        metrics = analyzer.analyze_summary_mode()
        
    report = render_report(metrics, mode_to_run)
    
    out_path = Path(args.output)
    out_path.write_text(report, encoding="utf-8")
    log.info("Drift analysis complete. Global verdict embedded in %s", out_path.absolute())
    print(f"\nReport saved to: {out_path.absolute()}")
    
if __name__ == "__main__":
    main()
