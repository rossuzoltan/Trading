from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERDICT_LINE = re.compile(r"\*\s*Verdict:\s*`([^`]+)`", re.IGNORECASE)
ANCHOR_LINE = re.compile(r"\*\s*Anchor status:\s*`([^`]+)`", re.IGNORECASE)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_gate_markdown(path: Path) -> dict[str, Any]:
    payload = {"present": False, "verdict": None, "anchor_status": None}
    if not path.exists():
        return payload
    text = path.read_text(encoding="utf-8", errors="ignore")
    verdict_match = VERDICT_LINE.search(text)
    anchor_match = ANCHOR_LINE.search(text)
    payload["present"] = True
    payload["verdict"] = verdict_match.group(1).strip() if verdict_match else None
    payload["anchor_status"] = anchor_match.group(1).strip() if anchor_match else None
    return payload


def collect_gate_entries(gates_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for json_path in sorted(gates_root.glob("*/*/paper_live_gate.json")):
        gate = _load_json(json_path)
        markdown_path = json_path.with_suffix(".md")
        md_payload = _parse_gate_markdown(markdown_path)
        entry = {
            "symbol": str(gate.get("symbol", "UNKNOWN")).upper(),
            "manifest_hash": str(gate.get("manifest_hash", "")),
            "anchor_status": str(gate.get("anchor_status", "unknown")),
            "final_verdict": str(gate.get("final_verdict", "unknown")),
            "verdict_reason": str(gate.get("verdict_reason", "")),
            "drift_verdict": str(dict(gate.get("drift_metrics", {}) or {}).get("verdict", "unknown")),
            "trading_days": int(dict(gate.get("shadow_summary_stats", {}) or {}).get("trading_days", 0) or 0),
            "actionable_events": int(
                dict(gate.get("shadow_summary_stats", {}) or {}).get("actionable_event_count", 0) or 0
            ),
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
            "markdown_present": bool(md_payload["present"]),
            "markdown_verdict": md_payload["verdict"],
            "markdown_anchor_status": md_payload["anchor_status"],
            "markdown_matches_json": bool(
                md_payload["present"]
                and md_payload["verdict"] == str(gate.get("final_verdict", "unknown"))
                and md_payload["anchor_status"] == str(gate.get("anchor_status", "unknown"))
            ),
        }
        entries.append(entry)
    return entries


def build_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_verdict = Counter(entry["final_verdict"] for entry in entries)
    by_symbol = Counter(entry["symbol"] for entry in entries)
    by_symbol_and_verdict: dict[str, dict[str, int]] = defaultdict(dict)
    for symbol in sorted({entry["symbol"] for entry in entries}):
        symbol_rows = [entry for entry in entries if entry["symbol"] == symbol]
        verdict_counts = Counter(row["final_verdict"] for row in symbol_rows)
        by_symbol_and_verdict[symbol] = dict(sorted(verdict_counts.items()))
    markdown_mismatch = [entry for entry in entries if not entry["markdown_matches_json"]]
    return {
        "gate_count": len(entries),
        "verdict_counts": dict(sorted(by_verdict.items())),
        "symbol_counts": dict(sorted(by_symbol.items())),
        "symbol_verdict_counts": dict(sorted(by_symbol_and_verdict.items())),
        "markdown_mismatch_count": len(markdown_mismatch),
        "markdown_mismatches": markdown_mismatch,
        "entries": entries,
    }


def write_markdown(summary: dict[str, Any], output_md: Path) -> None:
    lines = [
        "# Paper-Live Gate Summary",
        "",
        f"* Gate files: `{summary['gate_count']}`",
        f"* Markdown mismatch count: `{summary['markdown_mismatch_count']}`",
        "",
        "## Verdict Counts",
        "| Verdict | Count |",
        "| :--- | ---: |",
    ]
    for verdict, count in dict(summary.get("verdict_counts", {}) or {}).items():
        lines.append(f"| {verdict} | {count} |")
    lines.extend(
        [
            "",
            "## Entries",
            "| Symbol | Manifest | Verdict | Anchor | Drift | Days | Events | Markdown parity |",
            "| :--- | :--- | :--- | :--- | :--- | ---: | ---: | :--- |",
        ]
    )
    for entry in summary.get("entries", []):
        lines.append(
            f"| {entry['symbol']} | `{entry['manifest_hash'][:12]}` | {entry['final_verdict']} | "
            f"{entry['anchor_status']} | {entry['drift_verdict']} | {entry['trading_days']} | "
            f"{entry['actionable_events']} | {entry['markdown_matches_json']} |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_visualization(summary: dict[str, Any], output_png: Path) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return "matplotlib-not-available"

    verdicts = sorted(dict(summary.get("verdict_counts", {}) or {}).keys())
    symbols = sorted(dict(summary.get("symbol_counts", {}) or {}).keys())
    if not verdicts or not symbols:
        return "no-data"

    bottom = [0] * len(verdicts)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for symbol in symbols:
        counts = dict(summary.get("symbol_verdict_counts", {}).get(symbol, {}) or {})
        heights = [int(counts.get(verdict, 0)) for verdict in verdicts]
        ax.bar(verdicts, heights, bottom=bottom, label=symbol)
        bottom = [base + height for base, height in zip(bottom, heights)]

    ax.set_title("Paper-Live Gate Verdicts by Symbol")
    ax.set_xlabel("Verdict")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paper-live gate artifacts and generate a compact dashboard.")
    parser.add_argument("--gates-root", default=str(ROOT / "artifacts" / "gates"))
    parser.add_argument("--output-prefix", default=str(ROOT / "artifacts" / "gates" / "gate_report_summary"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gates_root = Path(args.gates_root)
    output_prefix = Path(args.output_prefix)
    output_json = output_prefix.with_suffix(".json")
    output_md = output_prefix.with_suffix(".md")
    output_png = output_prefix.with_suffix(".png")

    entries = collect_gate_entries(gates_root)
    summary = build_summary(entries)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, output_md)
    viz_status = write_visualization(summary, output_png)
    print(
        json.dumps(
            {
                "gate_count": summary["gate_count"],
                "json": str(output_json),
                "markdown": str(output_md),
                "chart": str(output_png),
                "chart_status": viz_status,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
