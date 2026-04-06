from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path to ensure infra modules can be imported from tools/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from trading_config import deployment_paths


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


def build_summary(symbol: str) -> dict[str, Any]:
    paths = deployment_paths(symbol)
    rows = _load_jsonl(paths.execution_audit_path)
    accepted = [row for row in rows if bool(row.get("accepted"))]
    rejected = [row for row in rows if not bool(row.get("accepted"))]
    deltas = np.asarray([float(row.get("fill_delta_pips", 0.0) or 0.0) for row in accepted], dtype=np.float64)
    retcodes = Counter(str(row.get("retcode")) for row in rows)
    summary = {
        "symbol": symbol.upper(),
        "sample_count": len(rows),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "retcode_counts": dict(retcodes),
        "mean_fill_delta_pips": float(np.mean(deltas)) if len(deltas) else None,
        "mean_abs_fill_delta_pips": float(np.mean(np.abs(deltas))) if len(deltas) else None,
        "p95_abs_fill_delta_pips": float(np.percentile(np.abs(deltas), 95)) if len(deltas) else None,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize live MT5 execution audit.")
    parser.add_argument("--symbol", default="EURUSD")
    args = parser.parse_args()
    summary = build_summary(args.symbol)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


