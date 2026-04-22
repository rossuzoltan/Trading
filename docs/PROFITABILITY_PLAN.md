# Profitability Plan

## Current Directive

Focus on EURUSD only.

Sequence:
1. Keep EURUSD shadow evidence collection running.
2. Build attestable ops evidence (MT5 preflight + restart drill `real_mt5` + execution audit drift within thresholds).
3. Refresh and re-test replay evidence on fresh data when operator time allows.
4. Use targeted ablation to explain recent replay drift before broad re-optimization.
5. Only then think about live money.

## Notes

- Do not expand to GBPUSD or other symbols until EURUSD is materially stronger and operational evidence is complete.
- Treat live-money discussion as blocked until the evidence chain is complete.
- Current grounded state on `2026-04-21`:
  - promoted-manifest EURUSD RC1 OOS is still positive (`+$39.47`, `PF 1.172`, `110` trades)
  - the same RC1 pack is still fragile under slippage stress
  - the latest 30-day MT5 replay is `DRIFT_CRITICAL` because the recent live window is short-heavy (`1` long / `13` short opens)
  - `pre_test_gate` is now ready, but `paper_live_gate` still demotes because shadow evidence and ops evidence remain incomplete
- Near-term research priorities:
  - confirm whether recent drift persists after a fresh tick-data rebuild
  - keep using `tools/ablate_recent_replay.py` to compare guard variants on both holdout OOS and the recent MT5 window
  - improve diagnostics first, then re-optimize
- Near-term engineering priorities:
  - preserve the richer replay/ablation logging paths under `artifacts/research/`
  - ensure restart drill evidence is `real_mt5` attestable for ops sign-off
  - add stronger baseline/reporting coverage for `runtime_price_mr_unguarded` so slope/spread guards can be judged directly
  - shadow logging: set `SHADOW_LOG_FULL_FEATURES=1` when running the shadow simulator to capture full feature snapshots per bar
