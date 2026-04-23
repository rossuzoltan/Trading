# Shadow Sweep Report - EURUSD

* Audit root: `C:\dev\trading\artifacts\shadow_sweep_evidence_ladder_v2_20260423_132114`
* Window: `2026-04-23T11:21:14.135000+00:00` -> `2026-04-23T18:41:01.720000+00:00`
* Profiles: `8`

## Ranking
* Ranking withheld: no profile met the minimum sample/realism gate (`20` days, `30` actionable, `80%` realized coverage, proven cost parity).

## Per-Profile
### p01_guarded_core
* Manifest hash: `b3356f616b4784716d25e46837b06cba15e6967ce701bab8da14fa17c1e8cfa5`
* Events: `7384` Trading days: `1` Actionable: `163`
* Opens: `82` Closes: `82` Holds: `492` Flat: `6729`
* Realism: coverage=`100.00%` cost_parity=`True` rankable=`False`
* Signals: `574` Rule-candidate: `574` Raw-price: `2443` Guard-blocked(raw): `1869`
* Guard failures (top): `spread_z_limit=1314, time_delta_z_limit=849`
* Rule blocks (top): `spread_z_limit=3911, price_z_threshold=1868, time_delta_z_limit=950`
* Net PnL (event snapshots): `469.10` pips over `82` trades
* Avg/trade: `5.72` pips (95% CI `4.40` .. `7.04`)
* Ranking blockers: `trading_days<20`

### p02_guarded_plus
* Manifest hash: `27d36313a4630a59c2d373bac7f7323a9f1fd74b5a9692ff1efbcf9e7e61f52c`
* Events: `7384` Trading days: `1` Actionable: `183`
* Opens: `93` Closes: `93` Holds: `715` Flat: `6486`
* Realism: coverage=`100.00%` cost_parity=`True` rankable=`False`
* Signals: `808` Rule-candidate: `808` Raw-price: `2443` Guard-blocked(raw): `1635`
* Guard failures (top): `spread_z_limit=1011, time_delta_z_limit=849`
* Rule blocks (top): `spread_z_limit=3282, price_z_threshold=2167, time_delta_z_limit=1036, ma50_slope_limit=1`
* Net PnL (event snapshots): `541.27` pips over `93` trades
* Avg/trade: `5.82` pips (95% CI `4.64` .. `7.00`)
* Ranking blockers: `trading_days<20`

### p03_centerline
* Manifest hash: `7c22706a8294a62d09322976461489e8470e0565127ecee32a1cdce989baca38`
* Events: `7384` Trading days: `1` Actionable: `192`
* Opens: `98` Closes: `98` Holds: `906` Flat: `6286`
* Realism: coverage=`100.00%` cost_parity=`True` rankable=`False`
* Signals: `1004` Rule-candidate: `1004` Raw-price: `2443` Guard-blocked(raw): `1439`
* Guard failures (top): `time_delta_z_limit=849, spread_z_limit=791`
* Rule blocks (top): `spread_z_limit=2834, price_z_threshold=2374, time_delta_z_limit=1060, ma50_slope_limit=14, ma20_slope_limit=4`
* Net PnL (event snapshots): `525.23` pips over `98` trades
* Avg/trade: `5.36` pips (95% CI `4.12` .. `6.60`)
* Ranking blockers: `trading_days<20`

### p04_spread_step
* Manifest hash: `5716e4b363b40be005bb5148d5f720346f6c3d7408c9b9edbf4e640dca35d4ba`
* Events: `7384` Trading days: `1` Actionable: `204`
* Opens: `104` Closes: `104` Holds: `950` Flat: `6230`
* Realism: coverage=`100.00%` cost_parity=`True` rankable=`False`
* Signals: `1054` Rule-candidate: `1054` Raw-price: `2443` Guard-blocked(raw): `1389`
* Guard failures (top): `time_delta_z_limit=849, spread_z_limit=736`
* Rule blocks (top): `spread_z_limit=2561, price_z_threshold=2468, time_delta_z_limit=1171, ma50_slope_limit=24, ma20_slope_limit=6`
* Net PnL (event snapshots): `584.38` pips over `104` trades
* Avg/trade: `5.62` pips (95% CI `4.50` .. `6.74`)
* Ranking blockers: `trading_days<20`

### p05_slope_step
* Manifest hash: `ffc4269b0cb5ec35db41cc9a3938c9090b430d8a4e0001f451cd1c130a342a3f`
* Events: `7384` Trading days: `1` Actionable: `161`
* Opens: `81` Closes: `81` Holds: `555` Flat: `6668`
* Realism: coverage=`100.00%` cost_parity=`True` rankable=`False`
* Signals: `636` Rule-candidate: `636` Raw-price: `2443` Guard-blocked(raw): `1807`
* Guard failures (top): `spread_z_limit=1223, time_delta_z_limit=849`
* Rule blocks (top): `spread_z_limit=3758, price_z_threshold=1932, time_delta_z_limit=978`
* Net PnL (event snapshots): `486.50` pips over `81` trades
* Avg/trade: `6.01` pips (95% CI `4.78` .. `7.23`)
* Ranking blockers: `trading_days<20`

### p06_balanced_relaxed
* Manifest hash: `668388af1efcd06f2cfb1b10028f89a62b222aac8fde0c2c94d3c8106dd7ea5f`
* Events: `7384` Trading days: `1` Actionable: `192`
* Opens: `98` Closes: `98` Holds: `906` Flat: `6286`
* Realism: coverage=`100.00%` cost_parity=`True` rankable=`False`
* Signals: `1004` Rule-candidate: `1004` Raw-price: `2443` Guard-blocked(raw): `1439`
* Guard failures (top): `time_delta_z_limit=849, spread_z_limit=791`
* Rule blocks (top): `spread_z_limit=2834, price_z_threshold=2378, time_delta_z_limit=1060, ma50_slope_limit=10, ma20_slope_limit=4`
* Net PnL (event snapshots): `525.23` pips over `98` trades
* Avg/trade: `5.36` pips (95% CI `4.12` .. `6.60`)
* Ranking blockers: `trading_days<20`

### p07_upper_guardrail
* Manifest hash: `ac80ea01a30d4e56047068a2a8803fa01a972462864dd22f6980209c728c311c`
* Events: `7384` Trading days: `1` Actionable: `212`
* Opens: `108` Closes: `108` Holds: `958` Flat: `6214`
* Realism: coverage=`100.00%` cost_parity=`True` rankable=`False`
* Signals: `1066` Rule-candidate: `1066` Raw-price: `2443` Guard-blocked(raw): `1377`
* Guard failures (top): `time_delta_z_limit=849, spread_z_limit=706`
* Rule blocks (top): `price_z_threshold=2570, spread_z_limit=2434, time_delta_z_limit=1192, ma50_slope_limit=13, ma20_slope_limit=5`
* Net PnL (event snapshots): `608.38` pips over `108` trades
* Avg/trade: `5.63` pips (95% CI `4.53` .. `6.73`)
* Ranking blockers: `trading_days<20`

### p08_exploratory_ceiling
* Manifest hash: `69be1e29e33c0838afd531d0874bdf70dab912a896ccf2da04ce471a7e571282`
* Events: `7384` Trading days: `1` Actionable: `226`
* Opens: `115` Closes: `115` Holds: `1079` Flat: `6079`
* Realism: coverage=`100.00%` cost_parity=`True` rankable=`False`
* Signals: `1194` Rule-candidate: `1194` Raw-price: `2443` Guard-blocked(raw): `1249`
* Guard failures (top): `time_delta_z_limit=849, spread_z_limit=578`
* Rule blocks (top): `price_z_threshold=2825, spread_z_limit=2054, time_delta_z_limit=1195, ma50_slope_limit=4, ma20_slope_limit=1`
* Net PnL (event snapshots): `587.48` pips over `115` trades
* Avg/trade: `5.11` pips (95% CI `3.95` .. `6.27`)
* Ranking blockers: `trading_days<20`

## Divergence (Last 10)
* `2026-04-23T18:09:26.351000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core
  - `sig=1 allow=1 act=hold reason=authorized` -> p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-23T18:11:16.062000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core
  - `sig=1 allow=1 act=hold reason=authorized` -> p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-23T18:13:44.458000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core
  - `sig=1 allow=1 act=hold reason=authorized` -> p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-23T18:16:26.517000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core
  - `sig=1 allow=1 act=hold reason=authorized` -> p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-23T18:20:03.592000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core
  - `sig=1 allow=1 act=hold reason=authorized` -> p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-23T18:23:20.346000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core
  - `sig=1 allow=1 act=hold reason=authorized` -> p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-23T18:26:32.693000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core
  - `sig=1 allow=1 act=hold reason=authorized` -> p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-23T18:30:30.078000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core
  - `sig=1 allow=1 act=hold reason=authorized` -> p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-23T18:31:21.763000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core, p05_slope_step
  - `sig=1 allow=1 act=open reason=authorized` -> p02_guarded_plus, p03_centerline, p04_spread_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-23T18:35:08.513000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core
  - `sig=0 allow=1 act=close reason=authorized_exit` -> p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
