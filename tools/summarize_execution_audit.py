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
    # Prefer recomputing drift from raw prices when available.
    # Historical rows may have inflated/incorrect fill_delta_pips; sent_price vs fill_price is the cleanest basis.
    recomputed: list[float] = []
    recomputed_requested: list[float] = []
    stored: list[float] = []
    for row in accepted:
        try:
            fill_price = float(row.get("fill_price", 0.0) or 0.0)
            sent_price = float(row.get("sent_price", 0.0) or 0.0)
            requested_price = float(row.get("requested_price", 0.0) or 0.0)
            pip_size = float(row.get("pip_size", 0.0) or 0.0)
        except Exception:
            fill_price = 0.0
            sent_price = 0.0
            requested_price = 0.0
            pip_size = 0.0

        # pip_size is not currently logged; infer from symbol where possible.
        if pip_size <= 0.0:
            from symbol_utils import pip_size_for_symbol

            pip_size = float(pip_size_for_symbol(str(row.get("symbol", symbol)) or symbol) or 0.0)

        if fill_price and sent_price and pip_size:
            recomputed.append((fill_price - sent_price) / pip_size)
        elif row.get("fill_delta_pips") is not None:
            stored.append(float(row.get("fill_delta_pips", 0.0) or 0.0))

        if fill_price and requested_price and pip_size:
            recomputed_requested.append((fill_price - requested_price) / pip_size)

    deltas = np.asarray(recomputed or stored, dtype=np.float64)
    requested_deltas = np.asarray(recomputed_requested, dtype=np.float64)
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
        "drift_basis": "sent_price" if recomputed else "stored_fill_delta_pips",
        "recomputed_from_prices": bool(recomputed),
        "recomputed_count": int(len(recomputed)),
        "stored_count": int(len(stored)),
        "mean_abs_fill_delta_pips_requested": float(np.mean(np.abs(requested_deltas))) if len(requested_deltas) else None,
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


