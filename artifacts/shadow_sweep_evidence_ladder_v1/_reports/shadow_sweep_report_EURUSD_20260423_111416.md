# Shadow Sweep Report - EURUSD

* Audit root: `C:\dev\trading\artifacts\shadow_sweep_evidence_ladder_v1`
* Window: `2026-04-22T12:04:54.776000+00:00` -> `2026-04-23T09:09:22.543000+00:00`
* Profiles: `8`

## Per-Profile
### p01_guarded_core
* Manifest hash: `b3356f616b4784716d25e46837b06cba15e6967ce701bab8da14fa17c1e8cfa5`
* Events: `18` Trading days: `2` Actionable: `2`
* Opens: `1` Closes: `1` Holds: `0` Flat: `16`
* Signals: `1` Rule-candidate: `1` Raw-price: `5` Guard-blocked(raw): `4`
* Guard failures (top): `ma50_slope_limit=3, spread_z_limit=2, ma20_slope_limit=1`
* Rule blocks (top): `spread_z_limit=9, ma20_slope_limit=6`
* Runtime gate blocks (top): `session=1`

### p02_guarded_plus
* Manifest hash: `27d36313a4630a59c2d373bac7f7323a9f1fd74b5a9692ff1efbcf9e7e61f52c`
* Events: `18` Trading days: `2` Actionable: `2`
* Opens: `1` Closes: `1` Holds: `0` Flat: `16`
* Signals: `1` Rule-candidate: `1` Raw-price: `5` Guard-blocked(raw): `4`
* Guard failures (top): `ma50_slope_limit=3, spread_z_limit=2, ma20_slope_limit=1`
* Rule blocks (top): `ma20_slope_limit=8, spread_z_limit=7`
* Runtime gate blocks (top): `session=1`

### p03_centerline
* Manifest hash: `7c22706a8294a62d09322976461489e8470e0565127ecee32a1cdce989baca38`
* Events: `18` Trading days: `2` Actionable: `4`
* Opens: `2` Closes: `2` Holds: `0` Flat: `14`
* Signals: `2` Rule-candidate: `2` Raw-price: `5` Guard-blocked(raw): `3`
* Guard failures (top): `ma20_slope_limit=1, ma50_slope_limit=1, spread_z_limit=1`
* Rule blocks (top): `ma20_slope_limit=10, spread_z_limit=3`
* Runtime gate blocks (top): `session=1`

### p04_spread_step
* Manifest hash: `5716e4b363b40be005bb5148d5f720346f6c3d7408c9b9edbf4e640dca35d4ba`
* Events: `18` Trading days: `2` Actionable: `4`
* Opens: `2` Closes: `2` Holds: `1` Flat: `13`
* Signals: `3` Rule-candidate: `3` Raw-price: `5` Guard-blocked(raw): `2`
* Guard failures (top): `ma20_slope_limit=1, ma50_slope_limit=1`
* Rule blocks (top): `ma20_slope_limit=10, price_z_threshold=1, spread_z_limit=1`
* Runtime gate blocks (top): `session=1`

### p05_slope_step
* Manifest hash: `ffc4269b0cb5ec35db41cc9a3938c9090b430d8a4e0001f451cd1c130a342a3f`
* Events: `18` Trading days: `2` Actionable: `4`
* Opens: `2` Closes: `2` Holds: `1` Flat: `13`
* Signals: `3` Rule-candidate: `3` Raw-price: `5` Guard-blocked(raw): `2`
* Guard failures (top): `spread_z_limit=2`
* Rule blocks (top): `spread_z_limit=7, ma20_slope_limit=5`
* Runtime gate blocks (top): `session=1`

### p06_balanced_relaxed
* Manifest hash: `668388af1efcd06f2cfb1b10028f89a62b222aac8fde0c2c94d3c8106dd7ea5f`
* Events: `18` Trading days: `2` Actionable: `2`
* Opens: `1` Closes: `1` Holds: `3` Flat: `13`
* Signals: `4` Rule-candidate: `4` Raw-price: `5` Guard-blocked(raw): `1`
* Guard failures (top): `spread_z_limit=1`
* Rule blocks (top): `ma20_slope_limit=9, spread_z_limit=2, price_z_threshold=1`
* Runtime gate blocks (top): `session=1`

### p07_upper_guardrail
* Manifest hash: `ac80ea01a30d4e56047068a2a8803fa01a972462864dd22f6980209c728c311c`
* Events: `18` Trading days: `2` Actionable: `2`
* Opens: `1` Closes: `1` Holds: `4` Flat: `12`
* Signals: `5` Rule-candidate: `5` Raw-price: `5` Guard-blocked(raw): `0`
* Rule blocks (top): `ma20_slope_limit=8, price_z_threshold=2, spread_z_limit=1`
* Runtime gate blocks (top): `session=1`

### p08_exploratory_ceiling
* Manifest hash: `69be1e29e33c0838afd531d0874bdf70dab912a896ccf2da04ce471a7e571282`
* Events: `18` Trading days: `2` Actionable: `2`
* Opens: `1` Closes: `1` Holds: `4` Flat: `12`
* Signals: `5` Rule-candidate: `5` Raw-price: `5` Guard-blocked(raw): `0`
* Rule blocks (top): `price_z_threshold=7, ma20_slope_limit=4`
* Runtime gate blocks (top): `session=1`

## Divergence (Last 10)
* `2026-04-22T12:04:54.776000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core, p02_guarded_plus, p03_centerline, p05_slope_step, p06_balanced_relaxed
  - `sig=-1 allow=1 act=open reason=authorized` -> p04_spread_step, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-22T12:50:33.348000+00:00` clusters=`2`
  - `sig=-1 allow=1 act=open reason=authorized` -> p01_guarded_core, p02_guarded_plus, p03_centerline, p05_slope_step, p06_balanced_relaxed
  - `sig=-1 allow=1 act=hold reason=authorized` -> p04_spread_step, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-22T13:31:26.210000+00:00` clusters=`2`
  - `sig=0 allow=1 act=close reason=authorized_exit` -> p01_guarded_core, p02_guarded_plus, p03_centerline, p04_spread_step
  - `sig=-1 allow=1 act=hold reason=authorized` -> p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-22T14:16:28.973000+00:00` clusters=`4`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core, p02_guarded_plus
  - `sig=-1 allow=1 act=open reason=authorized` -> p03_centerline, p04_spread_step
  - `sig=0 allow=1 act=close reason=authorized_exit` -> p05_slope_step
  - `sig=-1 allow=1 act=hold reason=authorized` -> p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-22T14:52:43.304000+00:00` clusters=`4`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core, p02_guarded_plus
  - `sig=0 allow=1 act=close reason=authorized_exit` -> p03_centerline, p04_spread_step
  - `sig=-1 allow=1 act=open reason=authorized` -> p05_slope_step
  - `sig=-1 allow=1 act=hold reason=authorized` -> p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
* `2026-04-22T15:35:10.782000+00:00` clusters=`2`
  - `sig=0 allow=0 act=flat reason=no signal` -> p01_guarded_core, p02_guarded_plus, p03_centerline, p04_spread_step
  - `sig=0 allow=1 act=close reason=authorized_exit` -> p05_slope_step, p06_balanced_relaxed, p07_upper_guardrail, p08_exploratory_ceiling
