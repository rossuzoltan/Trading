# Full Rehab Plan: Runtime Correctness + Baseline-First RL Recovery

## Summary
A megvalósítás két kapura épüljön, ebben a sorrendben:

1. **Engineering truthfulness gate**: a runtime, multiprocessing, heartbeat és monitor legyen auditálható és félreérthetetlen.
2. **Research validity gate**: az RL csak akkor maradhat elsődleges út, ha ugyanazon spliten egy egyszerű baseline modell bizonyítottan jobb a zajnál.

A terv **nem** vált `RecurrentPPO`-ra és **nem** erőlteti a GPU-t. A támogatott stack marad `MaskablePPO + RuntimeGymEnv + volume bars`, mert a jelenlegi fő hiba nem a CUDA hiánya, hanem a gyenge signal, a rossz reward és a félrevezető eval.

## Key Changes
### 1. Runtime / Telemetry Hardening
- Tartsd meg a mostani `interpreter_guard` irányt, és terjeszd ki minden közvetlenül futtatott, `.venv`-függő top-level entrypointra.
- `train_agent.py` induláskor explicit állítsa be a multiprocessing executable-t a repo `.venv` interpreterére, és egyszer logolja:
  - `sys.executable`
  - `sys.prefix`
  - configured multiprocessing executable
  - vec env type / env worker count
- A heartbeat schema legyen **v2**, új kötelező mezőkkel:
  - `schema_version`
  - `process_started_utc`
  - `num_timesteps`
  - `n_updates`
  - `diagnostic_sample_count`
  - `ppo_diagnostics`
- A `ppo_diagnostics` tartalmazza a mostani metrikákat, plusz:
  - `last_distinct_update_seen`
  - `metrics_fresh`
- A `monitor_training.ps1` rolling throughputot számoljon két egymást követő heartbeatből; lifetime average csak fallback legyen.
- A `watch_training.ps1` jelölje külön:
  - `healthy`
  - `stale_heartbeat`
  - `no_progress`
- A pre-fix heartbeat/log artefaktumokat kezeld “diagnostically contaminated” állapotként; ezt a `training_status.py` is írja ki, ha hiányzik a v2 schema.

### 2. Baseline-First Research Gate
- `train_agent.py` elején vezess be kötelező baseline kaput `TRAIN_REQUIRE_BASELINE_GATE=1` defaulttal.
- A baseline ugyanazt a train/val/holdout szeletelést használja, mint az RL.
- A baseline csomag legyen:
  - `Ridge` regresszor
  - egy sekély faalapú baseline (`HistGradientBoostingRegressor` vagy ennek megfelelő olcsó tree baseline)
- A közös target legyen **5-bar forward log return**.
- A baseline OOS report külön artefaktumba menjen: `models/baseline_diagnostics_{symbol}.json`.
- Az RL tréning csak akkor indulhat el, ha **legalább egy** baseline teljesíti a final holdouton mindhárom feltételt:
  - `R² > 0.0`
  - Pearson corr `>= 0.05`
  - sign accuracy `>= 0.52`
- Ha a gate fail, a training álljon le kemény hibával, és írja ki: “RL not justified: baseline gate failed.”

### 3. Evaluation Protocol Repair
- A jelenlegi `EvalCallback`-alapú, 5-ször ugyanarra a determinisztikus pályára lefuttatott eval helyett vezess be **egyedi custom evalot**.
- Training közbeni eval minden checkpointnál pontosan **1 determinisztikus full-path** validáció legyen. Ne jelenjen meg többé félrevezető `+/- 0.00`.
- Fold végi report tartalmazzon:
  - full-path val metrics
  - holdout metrics
  - splitelt metrics a validáció első / középső / utolsó harmadára
- A `training_diagnostics_{symbol}.json` bővüljön:
  - `baseline_gate_passed`
  - `eval_protocol_valid`
  - `full_path_eval_used`
  - `segment_metrics`
- A deploy gate ne fogadjon el olyan futást, ahol az eval report még a régi duplikált-epizód sémára épül.

### 4. Reward + Feature Redesign
- A `RuntimeEngine` rewardból vedd ki a `rolling std normalization + tanh` fő logikát.
- Az új default reward legyen:
  - `reward = 10000 * log(equity_t / equity_{t-1})`
  - mínusz lineáris drawdown penalty
  - mínusz turnover / transaction penalty
  - nincs `tanh` a fő rewardon
  - numerikai védelemként csak széles clip maradjon, pl. `[-5, 5]`
- A rewardot ne “kritikusan zajsűrítő” per-bar normalizálás vezesse, hanem közvetlen post-cost equity delta.
- A feature setet v1-ben szűkítsd le erre a 8 mezőre:
  - `log_return`
  - `body_size`
  - `candle_range`
  - `ma20_slope`
  - `ma50_slope`
  - `vol_norm_atr`
  - `spread_z`
  - `time_delta_z`
- Az alábbi feature-ök alapból kerüljenek ki az RL stackből, és csak későbbi ablation után jöhetnek vissza:
  - RSI
  - MACD / MACDH
  - Bollinger feature-ök
  - ADX
  - Hurst
  - frac diff
  - napi ciklikus feature-ök

### 5. PPO Retune Only After Gates Pass
- PPO retune csak baseline gate + új eval + új reward után.
- Az új default PPO config legyen:
  - `learning_rate = 3e-4`, lineáris decay `1e-4`-ig
  - `n_steps = 2048`
  - `batch_size = 1024`
  - `n_epochs = 10`
  - `ent_coef = 0.005`
  - `target_kl = 0.015`
  - `policy_kwargs.pi = [128, 128]`
  - `policy_kwargs.vf = [256, 256, 128]`
- Marad `MaskablePPO("MlpPolicy")`; nincs `RecurrentPPO`, nincs `MlpLstmPolicy`.
- `device` marad `auto`; GPU-t csak akkor érdemes újranyitni, ha az env throughput fixek után is compute-bound a futás.

## Public Interfaces / Artifacts
- Új heartbeat schema: `training_heartbeat.json` v2.
- Új baseline artifact: `models/baseline_diagnostics_{symbol}.json`.
- Bővített diagnostics artifact: `models/training_diagnostics_{symbol}.json` baseline/eval validity mezőkkel.
- A training startup log kötelezően kiírja a runtime interpreter és multiprocessing környezetet.

## Test Plan
- Unit test: interpreter guard helyesen ismeri fel a repo `.venv`-et és nem reexecel import-only esetben.
- Unit test: diagnostics callback csak distinct PPO update-et mintavételez.
- Unit test: heartbeat v2 serializáció kötelező mezőkkel.
- Unit test: baseline gate synthetic adaton pass/fail szerint működik.
- Unit test: custom eval nem generál duplikált `+/- 0.00` jellegű statisztikát ugyanarra a pathra.
- Unit test: reward monotonitás
  - pozitív post-cost equity delta -> pozitív reward
  - negatív equity delta -> negatív reward
  - magasabb turnover ugyanarra az equity deltára -> alacsonyabb reward
- Smoke test: system Pythonból indított `train_agent.py` reexecel a repo `.venv`-be.
- Smoke test: `TRAIN_NUM_ENVS=2` mellett a startup log kiírja a várt multiprocessing executable-t.
- Smoke test: két egymást követő heartbeatből a monitor rolling speedet és stale státuszt helyesen számol.

## Acceptance Criteria
- A monitor többé nem mutathat heartbeat-age alapú hamis speedet.
- A heartbeatből bizonyítható legyen, hogy a PPO metrika friss distinct update-ből származik.
- A baseline gate jelenlegi EURUSD adaton várhatóan failel; ez helyes kimenet, nem regresszió.
- Az eval report többé nem használhat ismételt azonos determinisztikus epizódokat stabilitás-bizonyítéknak.
- Az RL futás csak akkor engedhető tovább, ha a baseline gate pass és az új eval report valid.
- A PPO tuning csak ezután következhet; önmagában `cuda`, `higher ent_coef`, vagy `higher lr` nem számít elfogadott első lépésnek.

## Assumptions / Defaults
- A jelenlegi workspace-ben már meglévő `interpreter_guard` és diagnostics de-dup változtatások megmaradnak, ezekre építünk.
- A historical pre-fix logok nem használhatók stabilitási vagy throughput következtetésre.
- A projekt elsődleges célja most nem “bármi áron RL futtatás”, hanem a zajfit kizárása és a baseline-first döntési lánc lezárása.
