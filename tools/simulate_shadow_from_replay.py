from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interpreter_guard import ensure_project_venv

ensure_project_venv(project_root=Path(__file__).resolve().parent, script_path=__file__)

from paper_live_metrics import write_shadow_summary
from selector_manifest import load_selector_manifest, validate_paper_live_candidate_manifest


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def convert_replay_audit_to_shadow_events(
    *,
    manifest_path: str | Path,
    source_audit_jsonl: str | Path,
    output_events_jsonl: str | Path,
) -> tuple[Path, dict[str, Any]]:
    manifest = load_selector_manifest(
        manifest_path,
        verify_manifest_hash=True,
        strict_manifest_hash=True,
        require_component_hashes=True,
    )
    validate_paper_live_candidate_manifest(manifest)

    source_path = Path(source_audit_jsonl)
    rows = _load_jsonl(source_path)
    out_rows: list[dict[str, Any]] = []

    for row in rows:
        reason = str(row.get("reason", "") or "")
        # The paper-live shadow summarizer expects a consistent schema but will
        # accept either timestamp_utc or bar_ts. We emit both for clarity.
        bar_ts = row.get("bar_ts")
        out_row = dict(row)
        out_row.update(
            {
                "timestamp_utc": bar_ts,
                "symbol": str(manifest.strategy_symbol).upper(),
                "ticks_per_bar": int(manifest.ticks_per_bar or manifest.bar_construction_ticks_per_bar or 0),
                "manifest_hash": str(manifest.manifest_hash),
                "logic_hash": str(manifest.logic_hash),
                "evaluator_hash": str(manifest.evaluator_hash),
                "signal_direction": int(row.get("signal", 0) or 0),
                "action_state": "authorized"
                if bool(row.get("allow_execution", False)) and bool(row.get("would_open", False))
                else ("authorized_exit" if bool(row.get("allow_execution", False)) and bool(row.get("would_close", False)) else ""),
                "no_trade_reason": reason,
                "session_filter_pass": reason != "session blocked",
                "risk_filter_pass": True,
                "position_state": str(row.get("active_state", "flat") or "flat"),
            }
        )
        out_rows.append(out_row)

    output_path = Path(output_events_jsonl)
    _write_jsonl(output_path, out_rows)

    meta = {
        "manifest_path": str(Path(manifest_path)),
        "source_audit_jsonl": str(source_path),
        "output_events_jsonl": str(output_path),
        "manifest_hash": manifest.manifest_hash,
        "logic_hash": manifest.logic_hash,
        "evaluator_hash": manifest.evaluator_hash,
        "symbol": manifest.strategy_symbol,
        "ticks_per_bar": int(manifest.ticks_per_bar or manifest.bar_construction_ticks_per_bar or 0),
        "event_count": len(out_rows),
    }
    return output_path, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate simulated shadow evidence from MT5 replay audit JSONL.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--source-audit-jsonl", required=True)
    parser.add_argument("--output-dir", default=str(Path("artifacts") / "research" / "simulated_shadow"))
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    manifest = load_selector_manifest(
        manifest_path,
        verify_manifest_hash=True,
        strict_manifest_hash=True,
        require_component_hashes=True,
    )
    validate_paper_live_candidate_manifest(manifest)

    out_root = Path(args.output_dir) / str(manifest.strategy_symbol).upper() / str(manifest.manifest_hash)
    events_path = out_root / "events.shadow_compatible.jsonl"
    summary_json = out_root / "shadow_summary.json"
    summary_md = out_root / "shadow_summary.md"
    source_txt = out_root / "source.txt"

    events_path, meta = convert_replay_audit_to_shadow_events(
        manifest_path=manifest_path,
        source_audit_jsonl=args.source_audit_jsonl,
        output_events_jsonl=events_path,
    )
    source_txt.parent.mkdir(parents=True, exist_ok=True)
    source_txt.write_text(str(Path(args.source_audit_jsonl)) + "\nconverted_from=mt5_historical_replay_report.audit.jsonl\n", encoding="utf-8")

    write_shadow_summary(
        events_path=events_path,
        summary_json_path=summary_json,
        summary_markdown_path=summary_md,
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
