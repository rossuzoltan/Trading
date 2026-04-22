from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context.macro_calendar import _file_sha256, load_macro_calendar
from selector_manifest import load_selector_manifest, save_selector_manifest


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update an RC pack macro calendar and re-pin calendar_sha256 in the selector manifest."
    )
    parser.add_argument("--manifest-path", "--manifest", dest="manifest_path", required=True)
    parser.add_argument(
        "--calendar-path",
        dest="calendar_path",
        default=None,
        help="Optional override. Default: <manifest_dir>/macro_calendar.json",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path).resolve()
    manifest = load_selector_manifest(
        manifest_path,
        verify_manifest_hash=True,
        strict_manifest_hash=True,
        require_component_hashes=True,
    )
    runtime_constraints = dict(manifest.runtime_constraints or {})
    context_cfg = dict(runtime_constraints.get("context") or {})
    if not bool(context_cfg.get("enabled", False)):
        raise RuntimeError("Manifest context is not enabled. Refusing to pin calendar SHA.")

    calendar_path = Path(args.calendar_path).resolve() if args.calendar_path else (manifest_path.parent / "macro_calendar.json")
    if not calendar_path.exists():
        raise FileNotFoundError(f"Missing calendar file: {calendar_path}")

    # Validate schema and timestamps deterministically.
    load_result = load_macro_calendar(calendar_path, expected_sha256=None)
    if load_result.calendar is None or load_result.error:
        raise RuntimeError(f"Calendar invalid: {load_result.error}")

    sha256 = _file_sha256(calendar_path)
    context_cfg["calendar_path"] = str(calendar_path.relative_to(manifest_path.parent))
    context_cfg["calendar_sha256"] = str(sha256)
    runtime_constraints["context"] = context_cfg

    updated = manifest.__class__(**{**manifest.__dict__, "runtime_constraints": runtime_constraints, "manifest_hash": ""})
    save_selector_manifest(updated, manifest_path)

    # Print updated manifest hash + calendar sha for operator audit scripts.
    refreshed = load_selector_manifest(manifest_path, verify_manifest_hash=True, strict_manifest_hash=True)
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "manifest_hash": refreshed.manifest_hash,
                "calendar_path": str(calendar_path),
                "calendar_sha256": sha256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

