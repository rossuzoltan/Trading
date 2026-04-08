from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interpreter_guard import ensure_project_venv


ensure_project_venv(project_root=REPO_ROOT, script_path=__file__)

from research.schema import (
    CURRENT_TRAINING_RUN_PATH,
    ProposalValidationError,
    assert_no_active_training_run,
    ensure_research_layout,
    load_proposal,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"{_utc_stamp()} | {message}", flush=True)


def _python_exe() -> Path:
    return REPO_ROOT / ".venv" / "Scripts" / "python.exe"


def _proposal_key(proposal_path: Path) -> tuple[str, str, bool]:
    proposal = load_proposal(proposal_path)
    return (proposal.experiment_name, proposal.symbol.upper(), bool(proposal.fast_mode))


def _result_dir_matches(result_dir: Path, proposal_path: Path) -> bool:
    copied_proposal = result_dir / "proposal.json"
    if not copied_proposal.exists():
        return False
    try:
        copied = json.loads(copied_proposal.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    target = load_proposal(proposal_path)
    return (
        str(copied.get("experiment_name", "")).strip() == target.experiment_name
        and str(copied.get("symbol", "")).strip().upper() == target.symbol.upper()
        and bool(copied.get("fast_mode", False)) == bool(target.fast_mode)
    )


def _matching_result_dirs(proposal_path: Path) -> list[Path]:
    layout = ensure_research_layout(REPO_ROOT)
    matches = [
        result_dir
        for result_dir in layout.results_dir.iterdir()
        if result_dir.is_dir() and _result_dir_matches(result_dir, proposal_path)
    ]
    return sorted(matches, key=lambda item: item.stat().st_mtime, reverse=True)


def _latest_matching_result_dir(proposal_path: Path) -> Path | None:
    matches = _matching_result_dirs(proposal_path)
    return matches[0] if matches else None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _active_training_run_payload() -> dict[str, Any] | None:
    path = REPO_ROOT / CURRENT_TRAINING_RUN_PATH
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else None


def _pid_is_running(raw_pid: Any) -> bool:
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        process_handle = kernel32.OpenProcess(0x100000, False, pid)
        if not process_handle:
            return False
        try:
            return int(kernel32.WaitForSingleObject(process_handle, 0)) == 0x00000102
        finally:
            kernel32.CloseHandle(process_handle)
    try:
        import os

        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _training_active() -> bool:
    payload = _active_training_run_payload()
    if isinstance(payload, dict):
        state = str(payload.get("state", "")).strip().lower()
        if state not in {"completed", "collapsed", "stopped"} and not state.startswith("failed_"):
            if _pid_is_running(payload.get("pid")):
                return True
    try:
        assert_no_active_training_run(REPO_ROOT / CURRENT_TRAINING_RUN_PATH)
    except ProposalValidationError:
        return True
    return False


def _wait_for_result(proposal_path: Path, *, poll_seconds: int, label: str) -> dict[str, Any] | None:
    _log(f"Waiting for {label} to finish: {proposal_path}")
    grace_without_result = 0
    last_result_dir: Path | None = None
    while True:
        result_dir = _latest_matching_result_dir(proposal_path)
        if result_dir is not None and result_dir != last_result_dir:
            _log(f"Observed result directory for {label}: {result_dir}")
            last_result_dir = result_dir
        if result_dir is not None:
            result_json = result_dir / "result.json"
            payload = _read_json(result_json)
            if isinstance(payload, dict):
                _log(
                    f"{label} finished with status={payload.get('run_status')} "
                    f"decision={payload.get('decision')} score={payload.get('composite_score')}"
                )
                return payload
        if _training_active():
            grace_without_result = 0
            time.sleep(max(poll_seconds, 5))
            continue
        grace_without_result += 1
        if grace_without_result >= 3:
            _log(f"No result.json found for {label} after training became inactive.")
            return None
        time.sleep(10)


def _stream_subprocess(command: list[str], *, label: str) -> int:
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    for line in process.stdout:
        _log(f"{label} | {line.rstrip()}")
    return int(process.wait())


def _run_proposal(proposal_path: Path, *, label: str, dry_run: bool) -> dict[str, Any] | None:
    command = [str(_python_exe()), str(REPO_ROOT / "tools" / "research_runner.py"), "--proposal", str(proposal_path)]
    _log(f"Launching {label}: {proposal_path}")
    if dry_run:
        _log(f"Dry-run only, skipping subprocess: {' '.join(command)}")
        return None
    started_dirs = {str(path) for path in _matching_result_dirs(proposal_path)}
    exit_code = _stream_subprocess(command, label=label)
    _log(f"{label} exited with code {exit_code}")
    result_dir = None
    for candidate in _matching_result_dirs(proposal_path):
        if str(candidate) not in started_dirs:
            result_dir = candidate
            break
    if result_dir is None:
        result_dir = _latest_matching_result_dir(proposal_path)
    if result_dir is None:
        _log(f"{label} produced no detectable result directory.")
        return None
    payload = _read_json(result_dir / "result.json")
    if not isinstance(payload, dict):
        _log(f"{label} finished without a readable result.json at {result_dir}")
        return None
    return payload


def _completed_fast_candidates(results: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        if str(result.get("run_status", "")).strip().lower() != "completed":
            continue
        completed.append(result)
    return completed


def _choose_winner(
    gated_result: dict[str, Any] | None,
    nogate_result: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    completed = _completed_fast_candidates([gated_result, nogate_result])
    if not completed:
        return None, None
    winner = max(
        completed,
        key=lambda item: float(item.get("composite_score", float("-inf")) or float("-inf")),
    )
    proposal_name = str((winner.get("resolved_proposal", {}) or {}).get("experiment_name", "")).strip()
    mode = "nogate" if proposal_name.endswith("_nogate") else "gate"
    return mode, winner


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a sequential unattended research queue.")
    parser.add_argument(
        "--current-proposal",
        default=str(REPO_ROOT / "research" / "proposals" / "eurusd_fast_reward_strip_window8.json"),
        help="The already-running fast proposal to wait for first.",
    )
    parser.add_argument(
        "--control-proposal",
        default=str(REPO_ROOT / "research" / "proposals" / "eurusd_fast_reward_strip_window8_nogate.json"),
        help="The fast no-gate control proposal to run after the current proposal.",
    )
    parser.add_argument(
        "--medium-gate-proposal",
        default=str(REPO_ROOT / "research" / "proposals" / "eurusd_medium_reward_strip_window8_gate.json"),
        help="The medium follow-up proposal to use when the gate-enabled fast run wins.",
    )
    parser.add_argument(
        "--medium-nogate-proposal",
        default=str(REPO_ROOT / "research" / "proposals" / "eurusd_medium_reward_strip_window8_nogate.json"),
        help="The medium follow-up proposal to use when the no-gate fast run wins.",
    )
    parser.add_argument(
        "--long-gate-proposal",
        default=str(REPO_ROOT / "research" / "proposals" / "eurusd_long_reward_strip_window8_gate.json"),
        help="The long tail proposal to use when the gate-enabled branch wins.",
    )
    parser.add_argument(
        "--long-nogate-proposal",
        default=str(REPO_ROOT / "research" / "proposals" / "eurusd_long_reward_strip_window8_nogate.json"),
        help="The long tail proposal to use when the no-gate branch wins.",
    )
    parser.add_argument("--poll-seconds", type=int, default=60, help="Polling interval while waiting on the current run.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the queue plan without launching new runs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    current_proposal = Path(args.current_proposal).resolve()
    control_proposal = Path(args.control_proposal).resolve()
    medium_gate_proposal = Path(args.medium_gate_proposal).resolve()
    medium_nogate_proposal = Path(args.medium_nogate_proposal).resolve()
    long_gate_proposal = Path(args.long_gate_proposal).resolve()
    long_nogate_proposal = Path(args.long_nogate_proposal).resolve()

    for proposal_path in (
        current_proposal,
        control_proposal,
        medium_gate_proposal,
        medium_nogate_proposal,
        long_gate_proposal,
        long_nogate_proposal,
    ):
        _log(f"Validating proposal: {proposal_path}")
        load_proposal(proposal_path)

    if args.dry_run:
        current_result_dir = _latest_matching_result_dir(current_proposal)
        _log(f"Dry-run current proposal: {current_proposal}")
        _log(f"Dry-run latest current result dir: {current_result_dir}")
        _log(f"Dry-run control proposal: {control_proposal}")
        _log(f"Dry-run medium gate proposal: {medium_gate_proposal}")
        _log(f"Dry-run medium no-gate proposal: {medium_nogate_proposal}")
        _log(f"Dry-run long gate proposal: {long_gate_proposal}")
        _log(f"Dry-run long no-gate proposal: {long_nogate_proposal}")
        _log(f"Dry-run active training detected: {_training_active()}")
        return 0

    gated_result = _wait_for_result(
        current_proposal,
        poll_seconds=max(int(args.poll_seconds), 5),
        label="current_fast",
    )
    nogate_result = _run_proposal(control_proposal, label="fast_nogate_control", dry_run=bool(args.dry_run))

    mode, winner = _choose_winner(gated_result, nogate_result)
    if winner is None or mode is None:
        _log("No completed fast result available. Skipping medium follow-up.")
        return 1

    winner_name = str((winner.get("resolved_proposal", {}) or {}).get("experiment_name", "")).strip()
    winner_score = winner.get("composite_score")
    _log(f"Fast winner: {winner_name} mode={mode} score={winner_score}")

    medium_proposal = medium_nogate_proposal if mode == "nogate" else medium_gate_proposal
    medium_result = _run_proposal(medium_proposal, label="medium_followup", dry_run=False)

    if isinstance(medium_result, dict):
        _log(
            f"Medium follow-up finished with status={medium_result.get('run_status')} "
            f"decision={medium_result.get('decision')} score={medium_result.get('composite_score')}"
        )
    else:
        _log("Medium follow-up produced no readable result.json.")
        return 1

    if str(medium_result.get("run_status", "")).strip().lower() != "completed":
        return 1

    long_proposal = long_nogate_proposal if mode == "nogate" else long_gate_proposal
    long_result = _run_proposal(long_proposal, label="long_tail_followup", dry_run=False)
    if isinstance(long_result, dict):
        _log(
            f"Long tail follow-up finished with status={long_result.get('run_status')} "
            f"decision={long_result.get('decision')} score={long_result.get('composite_score')}"
        )
        return 0 if str(long_result.get("run_status", "")).strip().lower() == "completed" else 1

    _log("Long tail follow-up produced no readable result.json.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
