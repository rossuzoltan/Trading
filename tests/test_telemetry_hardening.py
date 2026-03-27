from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

from training_status import resolve_current_run_heartbeat, summarize_heartbeat_schema


ROOT = Path(__file__).resolve().parent.parent
POWERSHELL = "powershell"


def run_powershell_json(command: str) -> dict[str, object]:
    completed = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if completed.returncode != 0:
        raise AssertionError(f"PowerShell failed:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    output = completed.stdout.strip()
    if not output:
        return {}
    return json.loads(output)


class TelemetryHardeningTests(unittest.TestCase):
    def test_training_status_summarizes_v2_fresh_and_legacy_states(self):
        fresh = summarize_heartbeat_schema(
            {
                "schema_version": 2,
                "timestamp_utc": "2026-03-26T00:00:10+00:00",
                "num_timesteps": 200,
                "ppo_diagnostics": {
                    "diagnostic_sample_count": 4,
                    "last_distinct_update_seen": 20,
                    "metrics_fresh": True,
                },
            }
        )
        self.assertEqual("v2", fresh["schema_state"])
        self.assertEqual("fresh", fresh["freshness_state"])
        self.assertEqual("clean", fresh["contamination_state"])

        legacy = summarize_heartbeat_schema(
            {
                "timestamp_utc": "2026-03-26T00:00:10+00:00",
                "num_timesteps": 200,
                "ppo_diagnostics": {
                    "diagnostic_sample_count": 0,
                    "metrics_fresh": False,
                },
            }
        )
        self.assertEqual("missing", legacy["schema_state"])
        self.assertEqual("contaminated", legacy["contamination_state"])
        self.assertIn("schema_mismatch", legacy["contamination_reasons"])

    def test_monitor_throughput_prefers_rolling_and_falls_back_to_lifetime(self):
        monitor_ps1 = str(ROOT / "monitor_training.ps1")
        command = fr"""
        $env:TRAINING_TELEMETRY_TEST = '1'
        . '{monitor_ps1}' -NoLoop
        $current = [pscustomobject]@{{ timestamp_utc = '2026-03-26T00:00:10+00:00'; num_timesteps = 200 }}
        $previous = [pscustomobject]@{{ timestamp_utc = '2026-03-26T00:00:05+00:00'; num_timesteps = 100 }}
        $rolling = Get-HeartbeatThroughput -CurrentHeartbeat $current -PreviousHeartbeat $previous -NowUtc ([datetime]::Parse('2026-03-26T00:00:10Z').ToUniversalTime()) -ProcessStartTime ([datetime]::Parse('2026-03-26T00:00:00Z').ToUniversalTime()) -TotalTimesteps 1000
        $lifetime = Get-HeartbeatThroughput -CurrentHeartbeat $current -PreviousHeartbeat $null -NowUtc ([datetime]::Parse('2026-03-26T00:00:10Z').ToUniversalTime()) -ProcessStartTime ([datetime]::Parse('2026-03-26T00:00:00Z').ToUniversalTime()) -TotalTimesteps 1000
        [pscustomobject]@{{ rolling = $rolling; lifetime = $lifetime }} | ConvertTo-Json -Compress -Depth 5
        """
        payload = run_powershell_json(command)
        self.assertEqual("rolling", payload["rolling"]["mode"])
        self.assertTrue(payload["rolling"]["used_rolling"])
        self.assertEqual("lifetime", payload["lifetime"]["mode"])
        self.assertFalse(payload["lifetime"]["used_rolling"])
        self.assertIn("20", str(payload["rolling"]["speed_text"]))

    def test_watch_classifies_healthy_no_progress_and_stale(self):
        watch_ps1 = str(ROOT / "watch_training.ps1")
        command = fr"""
        $env:TRAINING_TELEMETRY_TEST = '1'
        . '{watch_ps1}' -NoLoop
        $fresh = [pscustomobject]@{{ timestamp_utc = '2026-03-26T00:00:10+00:00'; num_timesteps = 200 }}
        $previous = [pscustomobject]@{{ timestamp_utc = '2026-03-26T00:00:05+00:00'; num_timesteps = 100 }}
        $same_steps = [pscustomobject]@{{ timestamp_utc = '2026-03-26T00:00:10+00:00'; num_timesteps = 100 }}
        $stale = [pscustomobject]@{{ timestamp_utc = '2026-03-25T23:58:30+00:00'; num_timesteps = 200 }}
        $healthy = Get-HeartbeatStatus -Heartbeat $fresh -PreviousHeartbeat $previous -NowUtc ([datetime]::Parse('2026-03-26T00:00:10Z').ToUniversalTime()) -StaleAfterSeconds 30
        $noProgress = Get-HeartbeatStatus -Heartbeat $same_steps -PreviousHeartbeat $previous -NowUtc ([datetime]::Parse('2026-03-26T00:00:10Z').ToUniversalTime()) -StaleAfterSeconds 30
        $staleStatus = Get-HeartbeatStatus -Heartbeat $stale -PreviousHeartbeat $previous -NowUtc ([datetime]::Parse('2026-03-26T00:00:10Z').ToUniversalTime()) -StaleAfterSeconds 30
        [pscustomobject]@{{ healthy = $healthy; no_progress = $noProgress; stale = $staleStatus }} | ConvertTo-Json -Compress -Depth 5
        """
        payload = run_powershell_json(command)
        self.assertEqual("healthy", payload["healthy"]["status"])
        self.assertEqual("no_progress", payload["no_progress"]["status"])
        self.assertEqual("stale_heartbeat", payload["stale"]["status"])

    def test_training_status_prefers_current_run_checkpoint_root(self):
        tmpdir = ROOT / "tests" / "tmp" / "telemetry_current_run"
        if tmpdir.exists():
            for child in sorted(tmpdir.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            legacy_root = tmpdir / "fold_0"
            legacy_root.mkdir(parents=True, exist_ok=True)
            (legacy_root / "training_heartbeat.json").write_text(
                json.dumps({"schema_version": 2, "timestamp_utc": "2026-03-25T00:00:00+00:00", "num_timesteps": 999}),
                encoding="utf-8",
            )
            current_root = tmpdir / "run_abc" / "fold_0"
            current_root.mkdir(parents=True, exist_ok=True)
            (current_root / "training_heartbeat.json").write_text(
                json.dumps({"schema_version": 2, "timestamp_utc": "2026-03-26T00:00:00+00:00", "num_timesteps": 123}),
                encoding="utf-8",
            )
            (tmpdir / "current_training_run.json").write_text(
                json.dumps({"run_id": "abc", "checkpoints_root": str(tmpdir / "run_abc"), "symbol": "USDJPY"}),
                encoding="utf-8",
            )

            heartbeat_path, context = resolve_current_run_heartbeat(tmpdir)

            self.assertIsNotNone(context)
            self.assertEqual(current_root / "training_heartbeat.json", heartbeat_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
