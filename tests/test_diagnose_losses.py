from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from tools.diagnose_losses import (
    analyze_trade_log,
    format_markdown_report,
    load_loss_context,
)


TEST_TMP_ROOT = Path("tests/tmp")


def make_test_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class DiagnoseLossesTests(unittest.TestCase):
    def test_load_loss_context_reads_line_delimited_trade_json(self) -> None:
        tmpdir = make_test_dir("diagnose_losses_jsonl")
        audit_path = tmpdir / "execution_audit_eurusd.jsonl"
        report_path = tmpdir / "replay_report_eurusd.json"
        audit_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "event": "position_closed",
                            "direction": 1,
                            "gross_pnl_usd": 12.5,
                            "net_pnl_usd": 10.0,
                            "transaction_cost_usd": 2.5,
                            "commission_usd": 1.0,
                            "spread_slippage_cost_usd": 1.5,
                            "holding_bars": 4,
                            "forced_close": False,
                        }
                    ),
                    "",
                    json.dumps(
                        {
                            "event": "position_closed",
                            "direction": -1,
                            "gross_pnl_usd": -7.0,
                            "net_pnl_usd": -9.0,
                            "transaction_cost_usd": 2.0,
                            "commission_usd": 1.0,
                            "spread_slippage_cost_usd": 1.0,
                            "holding_bars": 12,
                            "forced_close": True,
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )
        try:
            context = load_loss_context(symbol="EURUSD", report_path=report_path, audit_path=audit_path)
            self.assertEqual("trade_log", context["source"])
            self.assertEqual(2, len(context["trade_log"]))
            analysis = analyze_trade_log(context["trade_log"])
            self.assertEqual(2, analysis["count"])
            self.assertAlmostEqual(1.0, analysis["net_pnl_usd"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_loss_context_falls_back_to_summary_only_report(self) -> None:
        tmpdir = make_test_dir("diagnose_losses_summary")
        audit_path = tmpdir / "execution_audit_eurusd.jsonl"
        report_path = tmpdir / "replay_report_eurusd.json"
        report_path.write_text(
            json.dumps(
                {
                    "replay_metrics": {
                        "trade_count": 7,
                        "net_pnl_usd": -42.5,
                        "profit_factor": 0.8,
                        "expectancy_usd": -6.07,
                        "win_rate": 0.43,
                        "avg_holding_bars": 5.5,
                    }
                }
            ),
            encoding="utf-8",
        )
        audit_path.write_text(json.dumps({"event": "order_executed", "side": "open"}), encoding="utf-8")
        try:
            context = load_loss_context(symbol="EURUSD", report_path=report_path, audit_path=audit_path)
            self.assertEqual("summary_only", context["source"])
            report = format_markdown_report("EURUSD", context)
            self.assertIn("Summary-Only View", report)
            self.assertIn("Profit Factor", report)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
