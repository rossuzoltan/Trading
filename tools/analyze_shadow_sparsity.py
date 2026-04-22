from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description='Analyze shadow signal sparsity.')
    parser.add_argument('--events-path', required=True)
    args = parser.parse_args()

    path = Path(args.events_path)
    rows = load_jsonl(path)
    if not rows:
        print(json.dumps({'events': 0, 'message': 'no events found'}, indent=2))
        return 0

    total = len(rows)
    signals = sum(1 for r in rows if int(r.get('signal', r.get('signal_direction', 0)) or 0) != 0)
    opens = sum(1 for r in rows if bool(r.get('would_open', False)))
    closes = sum(1 for r in rows if bool(r.get('would_close', False)))
    reasons = Counter(str(r.get('reason', r.get('no_trade_reason', '')) or '') for r in rows)
    session_blocked = sum(1 for r in rows if not bool(r.get('session_ok', True)))
    spread_blocked = sum(1 for r in rows if not bool(r.get('spread_ok', True)))
    risk_blocked = sum(1 for r in rows if not bool(r.get('risk_ok', True)))

    payload = {
        'events': total,
        'signal_count': signals,
        'signal_rate': signals / total if total else 0.0,
        'would_open_count': opens,
        'would_close_count': closes,
        'session_blocked_count': session_blocked,
        'spread_blocked_count': spread_blocked,
        'risk_blocked_count': risk_blocked,
        'reason_counts': dict(reasons.most_common()),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
