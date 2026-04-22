from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_metrics import load_json_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize trading evidence chain artifacts.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--manifest-dir", default=None)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    base = Path(args.manifest_dir) if args.manifest_dir else ROOT / "models" / "rc1"
    packs = sorted(p for p in base.iterdir() if p.is_dir() and p.name.lower().startswith(symbol.lower())) if base.exists() else []
    summary = []
    for pack in packs:
        entry = {"pack": str(pack), "files": {}}
        for name in ["manifest.json", "pre_test_gate.json", "baseline_scoreboard_rc1.json", "mt5_historical_replay_report.json"]:
            path = pack / name
            entry["files"][name] = load_json_report(path) if path.exists() else None
        summary.append(entry)
    print(json.dumps({"symbol": symbol, "packs": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
