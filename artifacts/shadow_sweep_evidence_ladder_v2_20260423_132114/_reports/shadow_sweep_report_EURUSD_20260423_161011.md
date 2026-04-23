# Shadow Sweep Report - EURUSD

* Audit root: `C:\dev\trading\artifacts\shadow_sweep_evidence_ladder_v2_20260423_132114`
* Window: `2026-04-23T11:21:14.135000+00:00` -> `2026-04-23T18:09:25.496000+00:00`
* Profiles: `8`

## Per-Profile
### p01_guarded_core
* Manifest hash: `b3356f616b4784716d25e46837b06cba15e6967ce701bab8da14fa17c1e8cfa5`
* Events: `6850` Trading days: `1` Actionable: `163`
* Opens: `82` Closes: `82` Holds: `492` Flat: `6195`
* Signals: `574` Rule-candidate: `574` Raw-price: `2159` Guard-blocked(raw): `1585`
* Guard failures (top): `spread_z_limit=1100, time_delta_z_limit=779`
* Rule blocks (top): `spread_z_limit=3615, price_z_threshold=1762, time_delta_z_limit=818`
* Est PnL (snapshot): `44.85` pips over `21` trades

### p02_guarded_plus
* Manifest hash: `27d36313a4630a59c2d373bac7f7323a9f1fd74b5a9692ff1efbcf9e7e61f52c`
* Events: `6850` Trading days: `1` Actionable: `181`
* Opens: `92` Closes: `92` Holds: `624` Flat: `6045`
* Signals: `716` Rule-candidate: `716` Raw-price: `2159` Guard-blocked(raw): `1443`
* Guard failures (top): `spread_z_limit=889, time_delta_z_limit=779`
* Rule blocks (top): `spread_z_limit=3096, price_z_threshold=2043, time_delta_z_limit=905, ma50_slope_limit=1`
* Est PnL (snapshot): `22.05` pips over `21` trades

### p03_centerline
* Manifest hash: `7c22706a8294a62d09322976461489e8470e0565127ecee32a1cdce989baca38`
* Events: `6850` Trading days: `1` Actionable: `190`
* Opens: `97` Closes: `97` Holds: `815` Flat: `5845`
* Signals: `912` Rule-candidate: `912` Raw-price: `2159` Guard-blocked(raw): `1247`
* Guard failures (top): `time_delta_z_limit=779, spread_z_limit=669`
* Rule blocks (top): `spread_z_limit=2648, price_z_threshold=2250, time_delta_z_limit=929, ma50_slope_limit=14, ma20_slope_limit=4`
* Est PnL (snapshot): `15.20` pips over `26` trades

### p04_spread_step
* Manifest hash: `5716e4b363b40be005bb5148d5f720346f6c3d7408c9b9edbf4e640dca35d4ba`
* Events: `6850` Trading days: `1` Actionable: `202`
* Opens: `103` Closes: `103` Holds: `859` Flat: `5789`
* Signals: `962` Rule-candidate: `962` Raw-price: `2159` Guard-blocked(raw): `1197`
* Guard failures (top): `time_delta_z_limit=779, spread_z_limit=614`
* Rule blocks (top): `spread_z_limit=2375, price_z_threshold=2344, time_delta_z_limit=1040, ma50_slope_limit=24, ma20_slope_limit=6`
* Est PnL (snapshot): `5.10` pips over `27` trades

### p05_slope_step
* Manifest hash: `ffc4269b0cb5ec35db41cc9a3938c9090b430d8a4e0001f451cd1c130a342a3f`
* Events: `6850` Trading days: `1` Actionable: `159`
* Opens: `80` Closes: `80` Holds: `520` Flat: `6171`
* Signals: `600` Rule-candidate: `600` Raw-price: `2159` Guard-blocked(raw): `1559`
* Guard failures (top): `spread_z_limit=1045, time_delta_z_limit=779`
* Rule blocks (top): `spread_z_limit=3516, price_z_threshold=1808, time_delta_z_limit=847`
* Est PnL (snapshot): `31.50` pips over `19` trades

### p06_balanced_relaxed
* Manifest hash: `668388af1efcd06f2cfb1b10028f89a62b222aac8fde0c2c94d3c8106dd7ea5f`
* Events: `6850` Trading days: `1` Actionable: `190`
* Opens: `97` Closes: `97` Holds: `815` Flat: `5845`
* Signals: `912` Rule-candidate: `912` Raw-price: `2159` Guard-blocked(raw): `1247`
* Guard failures (top): `time_delta_z_limit=779, spread_z_limit=669`
* Rule blocks (top): `spread_z_limit=2648, price_z_threshold=2254, time_delta_z_limit=929, ma50_slope_limit=10, ma20_slope_limit=4`
* Est PnL (snapshot): `15.20` pips over `26` trades

### p07_upper_guardrail
* Manifest hash: `ac80ea01a30d4e56047068a2a8803fa01a972462864dd22f6980209c728c311c`
* Events: `6850` Trading days: `1` Actionable: `210`
* Opens: `107` Closes: `107` Holds: `867` Flat: `5773`
* Signals: `974` Rule-candidate: `974` Raw-price: `2159` Guard-blocked(raw): `1185`
* Guard failures (top): `time_delta_z_limit=779, spread_z_limit=584`
* Rule blocks (top): `price_z_threshold=2446, spread_z_limit=2248, time_delta_z_limit=1061, ma50_slope_limit=13, ma20_slope_limit=5`
* Est PnL (snapshot): `11.15` pips over `27` trades

### p08_exploratory_ceiling
* Manifest hash: `69be1e29e33c0838afd531d0874bdf70dab912a896ccf2da04ce471a7e571282`
* Events: `6850` Trading days: `1` Actionable: `224`
* Opens: `114` Closes: `114` Holds: `988` Flat: `5638`
* Signals: `1102` Rule-candidate: `1102` Raw-price: `2159` Guard-blocked(raw): `1057`
* Guard failures (top): `time_delta_z_limit=779, spread_z_limit=456`
* Rule blocks (top): `price_z_threshold=2701, spread_z_limit=1868, time_delta_z_limit=1064, ma50_slope_limit=4, ma20_slope_limit=1`
* Est PnL (snapshot): `12.45` pips over `32` trades

## Divergence (Last 10)
* `2026-04-23T15:00:38.829000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core, p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail
  - `sig=-1 allow=1 act=open reason=authorized` -> p08_exploratory_ceiling
* `2026-04-23T15:01:43.674000+00:00` clusters=`3`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core, p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed
  - `sig=-1 allow=1 act=open reason=authorized` -> p07_upper_guardrail
  - `sig=-1 allow=1 act=hold reason=authorized` -> p08_exploratory_ceiling
* `2026-04-23T15:03:06.644000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core, p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail
  - `sig=0 allow=1 act=close reason=authorized_exit` -> p08_exploratory_ceiling
* `2026-04-23T15:03:50.653000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core, p02_guarded_plus, p05_slope_step
  - `sig=-1 allow=1 act=open reason=authorized` -> p03_centerline, p04_spread_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-23T15:06:15.098000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core, p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed
  - `sig=1 allow=1 act=open reason=authorized` -> p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-23T15:06:15.121000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core, p05_slope_step
  - `sig=-1 allow=1 act=open reason=authorized` -> p02_guarded_plus, p03_centerline, p04_spread_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-23T15:18:22.345000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core
  - `sig=0 allow=1 act=close reason=authorized_exit` -> p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-23T15:41:34.801000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core, p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail
  - `sig=-1 allow=1 act=open reason=authorized` -> p08_exploratory_ceiling
* `2026-04-23T15:50:41.488000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core, p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail
  - `sig=1 allow=1 act=open reason=authorized` -> p08_exploratory_ceiling
* `2026-04-23T16:46:38.196000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core
  - `sig=1 allow=1 act=open reason=authorized` -> p02_guarded_plus, p03_centerline, p04_spread_step, p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
