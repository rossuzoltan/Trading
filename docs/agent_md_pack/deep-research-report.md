# Python-alapú trading rendszerek élő sikerének kritikus, falszifikáció-központú kutatása

## Scope assumptions

**Mit jelent itt a “Python-alapú trading rendszer”?** A kutatás abból indul ki, hogy Python tipikusan a research, adatpipeline, feature/training, monitoring és orchestration rétegben dominál, míg **mikrostruktúra-érzékeny, HFT-jellegű** execution/market-data feldolgozásnál a kritikus útvonalat gyakran alacsonyabb szintű (C++/FPGA) komponensek viszik; a “top 0.1%” HFT-hez szükséges sebességversenyt és annak mechanikus arbitrázs-rentjeit a szakirodalom is külön “arms race”-ként tárgyalja. citeturn1search2turn1search6

**Nem mosunk össze piacokat és időhorizontokat.** A következtetések érvényességét végig jelezzük az alábbi dimenziók mentén:

- **Eszközosztály:** equities (töredezett market structure, NBBO/Reg NMS jellegű best execution környezet), futures (tőzsdei, központosított matching és “matching algorithm” szabályok), FX (OTC dominancia; a globális FX turnover szerkezetét a BIS triennial survey dokumentálja), crypto (töredezett, 24/7, eltérő venue-k és infrastruktúra; fragmentációt BIS is tárgyalja). citeturn10search0turn10search3turn6search18turn10search2  
- **Időhorizont:** HFT (ms–s), intraday (percek–órák), swing (napok), daily+ (napok–hetek).  
- **Execution-szenzitivitás:**  
  - *Mikrostruktúra-érzékeny*: tick/L2/queue/latency/fee-rezsim erősen számít (pl. market making, spread-capture). A spread/adverse selection és price impact mikrostruktúra-alapjai klasszikus modellekben is központi téma. citeturn13search0turn13search3turn13search2  
  - *Alacsonyabb frekvenciás*: a fill model részletei kevésbé dominálnak, de **költségek, market impact, corporate actions, PIT-adatok, validáció és rezsimváltás** továbbra is döntő.

**Szabályozási és “production engineering” baseline:** EU-s környezetben az algoritmikus kereskedés kontrolljait RTS 6, a timestamp/clock sync elvárásokat RTS 25 írja le; ezek nem csak compliance-artefaktumok: gyakorlati minimumként is értelmezhetők a “deployálhatóság” szempontjából. citeturn8search0turn9search0turn8search7  
**SRE/ops** oldalról a monitoring, alerting, incident response, postmortem és “simplicity” elvek jól dokumentált “production survival” minták. citeturn8search2turn8search6

---

## Executive summary

### A tíz legfontosabb következtetés

1) **A monetizálható edge definíciója nem “Sharpe a backtestben”, hanem:** *edge* − *spread/fees/slippage/impact* − *hibák* − *rezsimromlás* = **költségek utáni, implementálható PnL**. Az “implementation shortfall” fogalom pontosan ezt az elmélet–megvalósítás rést formalizálja. citeturn6search0turn6search12  
**Evidenciaszint:** High confidence. **Minősítés:** Must-have (definíciós).

2) **A leggyakoribb bukás okai strukturálisak, nem modell-szintűek:** data leakage, PIT-hibák, survivorship bias, corporate action/roll kezelés, irreális fill és cost model, valamint ops/monitoring hiányosságok. A survivorship bias “látszólagos prediktálhatóságot” is generálhat (klasszikus bizonyíték). citeturn2search4turn2search0  
**Evidenciaszint:** High confidence. **Minősítés:** Must-have (bias-kontroll).

3) **A “top 0.1%” rendszerek közös jellemzője:** *end-to-end fegyelem* a teljes láncban (adat → validáció → költség → execution → kockázat → monitoring → recovery), és nem a “legbonyolultabb modell”. A túlkomplex rendszerek productionben törékenyek; a “simplicity” és incident-kultúra ezért stratégiai, nem esztétikai kérdés. citeturn8search9turn8search6  
**Evidenciaszint:** Medium–High (SRE erős; finance-ben részben indirekt). **Minősítés:** Strongly recommended → HFT-ben Must-have.

4) **Backtest-overfitting és selection bias kvantifikálható és gyakran brutális:** Bailey et al. a backtest overfitting valószínűségét (PBO) és a CSCV-t direkt erre fejlesztették; ez explicit cáfolata annak a naiv hitnek, hogy “holdout elég”. citeturn0search4turn5search4turn0search8  
**Evidenciaszint:** High confidence. **Minősítés:** Must-have (robosztus szelekció).

5) **Data snooping/multiple testing:** White “Reality Check”, Hansen SPA, Romano–Wolf stepwise eljárások és Harvey–Liu–Zhu “factor zoo” eredményei ugyanarra figyelmeztetnek: sok próbálkozás mellett a “t≈2” típusú szignifikancia küszöb messze nem elég. citeturn16search0turn5search5turn16search1turn0search14  
**Evidenciaszint:** High confidence. **Minősítés:** Must-have (kutatási protokoll).

6) **Edge decay és crowding valós és mérhető:** publikáció után a prediktív hozamok gyakran csökkennek; crowding a faktorokban és stratégiákban crash-kockázatot és hozamromlást okozhat. citeturn0search11turn3search11turn3search7  
**Evidenciaszint:** Medium–High. **Minősítés:** Strongly recommended (capacity/decay menedzsment).

7) **Microstructure-alpha vs research-alpha:** mikrostruktúra-érzékeny stratégiáknál a queue priority, matching szabályok, fee-rezsim és latency teszi vagy töri a rendszert; matching és order entry specifikációk (pl. OUCH) konkrétan rögzítik a price-time prioritást; futuresnél a matching algoritmus (FIFO/alloc/pro-rata) venue-specifikus és edge-kritikus. citeturn6search18turn6search5turn6search1  
**Evidenciaszint:** High confidence. **Minősítés:** Must-have (HFT/intraday MM), Situational (daily+).

8) **Költség- és impact-model nélkül nincs deploy:** optimális execution és market impact elmélete (Almgren–Chriss; Gatheral no-dynamic-arbitrage; empirikus impact törvények) azt jelzi, hogy a végrehajtási költség nem “apróság”, hanem sok stratégiánál a PnL nagy része. citeturn1search0turn14search2turn14search0turn14search1  
**Evidenciaszint:** High confidence. **Minősítés:** Must-have.

9) **ML/AI akkor indokolt, ha kontrolláltan és bizonyítékosan javít a baseline-on:** nagy mintákon a ML módszerek hozhatnak előnyt (pl. cross-sectional return prediction), de ez *nem* automatikusan jelent költségek utáni, stabil stratégiát; a modellhaszon könnyen elolvad szelekciós torzításban és cost/execution driftben. citeturn4search0turn0search14turn12search1  
**Evidenciaszint:** Medium (előny bizonyított; monetizálás feltételes). **Minősítés:** Situational.

10) **A deployálhatóság kritériuma production-szerű kontrollrendszer:** kill switch, pre-trade limitek, monitoring, audit trail, tesztelési és változáskezelési fegyelem. EU RTS 6 explicit “kill functionality”-t és fejlesztés előtti tesztelési követelményt említ; US oldalon a SEC Market Access Rule is pre-trade kontrollokra épít. citeturn8search7turn8search0turn1search3turn1search18  
**Evidenciaszint:** High confidence (szabályozói + gyakorlati). **Minősítés:** Must-have.

### A tíz leggyakoribb hiba

1) **PIT hiánya** (delistings, corp actions, fundamentals időbélyeg, macro vintage) → hamis edge. citeturn2search4turn2search6turn11search8  
2) **Survivorship bias** (csak “élő” tickerek). citeturn2search4  
3) **Look-ahead/target leakage** (rossz join, rossz label-horizont). citeturn2search7turn16search6  
4) **Multiple testing** “szabad szemmel optimalizált” stratégia-választás mellett (Reality Check/SPA hiánya). citeturn16search0turn5search5  
5) **Irreális fill** (bar-alapú “töltés” és queue ignorálása mikrostruktúrában). citeturn6search18turn13search0  
6) **Hiányos költségmodell** (spread, fees, impact, borrow, funding). citeturn6search12turn14search0  
7) **Regime blindness** (nem-stacionaritás figyelmen kívül). citeturn3search0  
8) **Backtest-engine bug / mismatch** a live-hoz képest (időzónák, session-ek, corp action). citeturn9search0turn11search1  
9) **Risk control hiány** (limit-ek, drawdown stop, kill switch). citeturn8search7turn1search3  
10) **Ops/monitoring hiány** (nincs alerting, nincs postmortem, nincs rollback). citeturn8search6turn8search9  

### Tíz must-have elem

1) **Point-in-time adatmodell + as-of join szabályok** (explicit timestamp-semantics). citeturn2search7turn2search15turn9search0  
2) **Survivorship + delisting kezelés** equitiesnél (különösen long-short/alpha kutatásnál). citeturn2search4turn11search9  
3) **Corporate action és price adjustment policy** dokumentálva. citeturn11search3turn11search1  
4) **Macro data: vintage/revision kezelés** (real-time dataset szemlélet). citeturn2search6turn2search14  
5) **Szigorú validáció + szelekciós torzítás kontroll** (PBO/Reality Check/SPA + walk-forward). citeturn0search4turn16search0turn5search5turn16search6  
6) **Költség és market impact modell** (legalább implementációs shortfall szemlélet). citeturn6search12turn1search0turn14search2  
7) **Execution engine venue-specifikus szabályokkal** (order types, time-in-force, matching priority). citeturn6search18turn6search1  
8) **Pre-trade risk checks + kill switch + circuit breakers**. citeturn8search7turn1search3turn8search0  
9) **Monitoring/alerting + incident playbook + postmortem**. citeturn8search6turn8search2  
10) **Reconciliation & audit trail** (orders/fills/positions/cash állapothelyes visszaépítése). A szabályozói elvárások (kontrollok és dokumentáltság) ezt is erősítik. citeturn1search7turn8search1  

### Öt dolog, amit a legtöbb fejlesztő túlértékel

1) **“Sharpe a backtestben” mint végső bizonyíték** (SR becslési hiba, nemnormalitás, szelekció). citeturn12search8turn12search1  
2) **Túlkomplex ML/DL** kis mintán (overfit + drift + karbantarthatósági adó). citeturn0search4turn8search9  
3) **OHLCV-hez kötött intraday “precision”** (mikrostruktúrához kevés). citeturn13search0turn1search1  
4) **Paper trading ≈ live** (slippage, queue, venue state, partial fills, throttling). citeturn6search18turn6search1  
5) **“Általánosítható alpha”** (edge decay/crowding). citeturn0search11turn3search11  

### Öt dolog, amit alulértékelnek, pedig döntő

1) **Point-in-time korrekt join-ok és timestamp integrity**. citeturn2search7turn9search0  
2) **Selection bias és multiple testing kontroll** (Reality Check/SPA/HLZ). citeturn16search0turn5search5turn0search14  
3) **Execution realitás (matching/fees/priority)**, különösen HFT/intraday. citeturn6search18turn6search1turn6search7  
4) **Operational resilience** (retry/idempotency/state recovery) – a PnL túlélés feltétele. citeturn8search6turn8search9  
5) **Rezsim-diagnosztika + kill criteria** (mikor kell leállítani). citeturn3search0turn8search7  

---

## Top 0.1% blueprint

Az alábbi “blueprint” rétegenként írja le: **cél**, **miért kritikus**, **tipikus hibák**, **minimum elfogadható szint**, **top 0.1% jellegzetességek**, továbbá megadja a **minősítést** és az **evidenciaszintet**. (A “top 0.1%” itt nem marketing: azt a rendszerszintű fegyelmet jelenti, ami statisztikailag csökkenti a “szép backtest, rossz live” valószínűségét.)

### Architektúra réteg

**Cél:** determinisztikus, auditálható, tesztelhető end-to-end lánc; “research → production” átjárhatóság.  
**Miért kritikus:** a legtöbb live bukás “nem-model” hiba (idő, adat, state, execution). A production rendszerekben a monitoring/incident/postmortem és a “simplicity” a túlélés eszköze. citeturn8search9turn8search6  
**Tipikus hibák:** state “szétszórva” (nincs single source of truth), ad-hoc configok, nem reprodukálható build, túl sok implicit feltételezés.  
**Minimum:** verziózott konfiguráció, determinisztikus backtest-engine, külön research és execution modul, egységes adat-idő modell (UTC + explicit timezone). (Clock sync/timestamp elvárások compliance-ben is megjelennek.) citeturn9search0turn9search2  
**Top 0.1%:** event-sourced order/position ledger; idempotens order router; replay-képes market-data pipeline; “shadow mode” és canary release mintázatok (SRE launch-checklist logikával). citeturn8search2turn8search6  
**Minősítés:** Must-have. **Evidencia:** Medium–High.

### Data pipeline réteg

**Cél:** **point-in-time**, survivorship-mentes, corporate actions/roll korrekt, timestamp-szemantikailag tiszta adatréteg.  
**Miért kritikus:** survivorship bias és PIT hibák hamis prediktálhatóságot hoznak létre; macro adatoknál a revision/vintage problémát külön real-time dataset irodalom tárgyalja. citeturn2search4turn2search6turn2search14  
**Tipikus hibák:** delistings eldobása; “adjusted close” félreértése; macro “final” értékek használata; rossz as-of join; corporate action / split faktor duplázása.  
**Minimum:**  
- PIT join szabály: as-of join (legközelebbi *megelőző* esemény) explicit, engine-szinten enforced; erre adatbázis-szinten is van natív konstrukció. citeturn2search7turn2search15  
- Equities: delisting return/állapot kezelése (CRSP-szemlélet). citeturn11search9turn11search8  
- Macro: vintage tárolás vagy megbízható real-time feed; Croushore–Stark szerint a vintage számít és a revision-ek forecastot is befolyásolnak. citeturn2search6turn2search2  
**Top 0.1%:** adat-minőség SLA-k; automatikus anomália-ellenőrzés (split/dividend jump detector); ID-migráció (PERMNO/PERMCO jellegű stabil ID); teljes lineage és “as reported vs restated” jelölés (PIT vendorok is ezt emelik ki). citeturn11search8turn2search1  
**Minősítés:** Must-have. **Evidencia:** High confidence.

### Feature stack réteg

**Cél:** olyan feature-ek, amelyek **gazdaságilag** is indokolhatók, stabilak és költség/rezsimérzékenységük ismert.  
**Miért kritikus:** a feature engineering és a modellkomplexitás cseréje gyakran hamis trade: sokszor a “jó feature + egyszerű modell” stabilabb, mint a “gyenge feature + komplex modell” (különösen drift mellett). (A túlkomplex rendszerek megbízhatósági adóját SRE “simplicity” szemlélet formalizálja.) citeturn8search9turn3search0  
**Tipikus hibák:** információtartalom nélküli “indikátor-zoo”; nem PIT-korrekt fundamentals; “feature leakage” label-horizontból.  
**Minimum:** feature-ekhez **data availability timestamp** és **lookback window** explicit; “no future data” unit tesztek; cross-sectional feature-eknél universe definíció PIT-korrekt (különben cross-sectional leakage). citeturn2search7turn2search4  
**Top 0.1%:** feature store PIT-szemantikával; drift-detektorok feature-eloszlásokra; “feature importance stability” rezsimenként.  
**Minősítés:** Strongly recommended (HFT-ben Must-have). **Evidencia:** Medium.

### Training pipeline réteg

**Cél:** reprodukálható tréning, kontrolált modellválasztás, explicit cost/turnover tudatosság.  
**Miért kritikus:** ML előnyök irodalma többnyire nagy adat- és modellezési fegyelemmel dolgozik; az ipari kudarcok gyakran szelekció/overfit és drift miatt történnek. citeturn4search0turn0search4turn0search14  
**Tipikus hibák:** “tuning on test”; túl sok hiperparaméter; nem pénzügyi célfüggvény (pl. accuracy) optimalizálása; turnover figyelmen kívül hagyása.  
**Minimum:** modellkomplexitás-korlát; baseline-ok (linear/tree/rules) kötelező; minden modellhez becsült **turnover** és költségérzékenység; kísérletnapló (seed, adatverzió, feature verzió).  
**Top 0.1%:** nested jellegű model selection folyamat; model registry; automatikus retrain csak akkor, ha “evidence threshold” teljesül; költség-tudatos loss/utility (különösen RL-nél). citeturn12search1turn4search2  
**Minősítés:** Strongly recommended. **Evidencia:** Medium.

### Validation pipeline réteg

**Cél:** a backtest **előszűrő** legyen, ne bizonyíték; a cél a hamis pozitívok agresszív kiszűrése.  
**Miért kritikus:**  
- PBO/CSCV azt mutatja, hogy klasszikus holdout gyakran nem védi ki a backtest-overfittinget. citeturn0search4turn5search4  
- Reality Check / SPA / stepwise multiple testing közvetlenül a “data snooping” ellen ad kontrollt. citeturn16search0turn5search5turn16search1  
- A “factor zoo” jelenség miatt a küszöböknek szigorodniuk kell. citeturn0search14  
**Tipikus hibák:** overlapping labels; leakage; walk-forward csak 1 útvonalon (nagy variancia); tuning a teljes idősoron; “benchmark” hiánya.  
**Minimum:**  
- walk-forward / rolling-origin validáció (idősor CV alapelvek). citeturn16search6turn16search2  
- data-snooping kontroll (legalább Reality Check vagy SPA-analóg). citeturn16search0turn5search5  
- Sharpe bizonytalanságot és szelekciót figyelembe vevő metrikák (PSR/DSR irány). citeturn12search8turn12search1  
**Top 0.1%:** PBO/CSCV + Reality Check/SPA kombó; stresszteszt rezsim-váltásokra; szimulált költségszcenáriók; “deflated skill” (DSR) jellegű korrekció; publikáció utáni decay-analóg ellenőrzés (ha ismert faktorokra épít). citeturn0search11turn12search1turn0search4  
**Minősítés:** Must-have. **Evidencia:** High confidence.

### Execution pipeline réteg

**Cél:** venue-specifikus, költség- és mikrostruktúra-tudatos order routing/placement; reális fill model; trade lifecycle kezelése.  
**Miért kritikus:** a spread/adverse selection és a price impact strukturális; a matching priority és order entry protokollok konkrétan rögzítik, hogyan alakul a queue és a fill. citeturn13search0turn6search18turn6search1  
**Tipikus hibák:** market/limit order “naiv” használata; time-in-force félreértése; throttling/partial fill figyelmen kívül; futures matching algoritmus nem ismerése. citeturn6search1turn6search5  
**Minimum:**  
- Order types & TIF támogatás; venue-spec matching és fee modell; reális slippage/impact becslés (minimum: implementation shortfall szemlélet). citeturn6search12turn6search18  
- Futuresnél: contract specs, tick/multiplier, session, és roll-kezelés dokumentált. citeturn17search3turn17search18  
**Top 0.1%:**  
- microstructure model-alapú placement (spread, order flow információtartalom; Hasbrouck-féle price impact szemlélet). citeturn13search2turn1search1  
- impact-aware slicing (Almgren–Chriss; Gatheral constraints). citeturn1search0turn14search2  
- latency mérése és determinisztikus event-time vs receive-time kezelés; timestamp integritás (RTS 25 logika). citeturn9search0turn9search2  
**Minősítés:** Must-have (HFT/intraday), Strongly recommended (swing/daily+). **Evidencia:** High confidence.

### Risk pipeline réteg

**Cél:** túlélés: drawdown kontroll, exposure limit, stressz, “stop trading” szabályok, position sizing.  
**Miért kritikus:** monetizálható edge csak akkor realizálható, ha a rendszer nem “blow up”; a kontrollok hiánya tipikus bukási mód. Szabályozói kontrollkövetelmények (pre-trade limitek) is ezt erősítik. citeturn1search3turn1search18turn8search0  
**Tipikus hibák:** túl nagy leverage, pro-ciklikus sizing, korrelációs “vakfolt”, tail risk ignorálása, limit nélküli order spam.  
**Minimum:**  
- hard risk limitek (max notional, max order size, max daily loss, max open risk);  
- drawdown-based circuit breaker;  
- liquiditás-alapú capacity korlát (impact nő az order mérettel; empirikus impact törvények). citeturn14search0turn14search1  
**Top 0.1%:** dinamikus risk budget rezsimenként; crowding-mérők figyelése; kill switch integrálva (RTS 6 Q&A szerint “azonnali outstanding order cancel” képesség). citeturn8search7turn3search11  
**Minősítés:** Must-have. **Evidencia:** High confidence.

### Monitoring pipeline réteg

**Cél:** élő drift, adat- és execution-anomáliák, risk breach, infrastruktúra hibák gyors detektálása + automatizált “safe state”.  
**Miért kritikus:** incident response és monitoring a production rendszerek túlélési eszközei; Google SRE külön fejezetekben tárgyalja monitoring/alerting/incident/postmortem kultúrát. citeturn8search2turn8search6  
**Tipikus hibák:** nincs SLO/SLA; alert fatigue; nincs “single pane of glass”; nincs post-trade recon.  
**Minimum:** késleltetés, fill ratio, slippage vs expected, cost drift, adat késés/hiány, pozíciók és cash recon, pre-trade limit violation log.  
**Top 0.1%:** SLO-k (adatfrissesség, order roundtrip); runbook-ok; automatikus downgrade (pl. csak risk-reducing orders); “blameless postmortem”. citeturn8search6turn8search9  
**Minősítés:** Must-have. **Evidencia:** Medium–High.

### Operations / recovery / failover pipeline

**Cél:** idempotens order kezelés, state recovery, reconciliation, deployment rollback, staged rollout (shadow/paper/canary).  
**Miért kritikus:** a live kereskedésben a rendszerhibák nem “csak bugok”, hanem **PnL események**; a dokumentált incident menedzsment és a launch/release engineering ezért közvetlen pénzügyi kontroll. citeturn8search6turn8search9  
**Tipikus hibák:** “retry ≠ idempotens”; reconnect után duplázott order; nem egyező broker vs internal position; rollback hiánya.  
**Minimum:** crash után determinisztikus újraindítás; trade/position ledger; broker/exchange recovery playbook. EU RTS 6 a rendszerkapacitás, reziliencia és tesztelés követelményét explicit említi. citeturn8search0turn8search1  
**Top 0.1%:** kétfázisú “order intent → order placement” napló; teljes replay; canary deployment; shadow mode-ban heteken át mért live/backtest drift. citeturn8search2turn0search4  
**Minősítés:** Must-have. **Evidencia:** Medium–High.

---

## Anti-pattern list

Az alábbi anti-patternök tipikusan **hamis backtest-edge**-et generálnak vagy a live-ban kinyírják a rendszert. Mindegyiknél jelzem: *miért bukás*, *milyen hamis jel csábít*, *felismerés*, *mitigáció*, *deploy gate*.

### Klasszikus backtest-csapdák és “szép Sharpe, rossz live” mintázatok

**Anti-pattern: “Sok paramétert próbáltam, a legjobb kell”**  
- **Miért bukás:** specification search → selection bias; PBO/CSCV közvetlenül kvantifikálja a backtest-overfitting valószínűségét optimalizáció/stratégia-szelekció után. citeturn0search4turn0search8  
- **Hamis jel:** extrém Sharpe/profit factor rövid mintán.  
- **Felismerés:** teljesítmény “összeomlik” új időszakokban; instabil paraméterek.  
- **Mitigáció:** CSCV/PBO; nested szelekció; White Reality Check / Hansen SPA; stepwise multiple testing. citeturn16search0turn5search5turn16search1  
- **Deploy gate:** ha nincs explicit multiple-testing kontroll → **Do not deploy**.  
**Veszély:** Critical. **Evidencia:** High.

**Anti-pattern: “Survivorship-mentes? minek, csak nagy részvényeket trade-elek”**  
- **Miért bukás:** survivorship bias már önmagában “prediktálhatóság illúziót” adhat; Brown et al. klasszikus bizonyíték. citeturn2search4turn2search0  
- **Hamis jel:** stabil outperformance “a múltban”, de universe valójában időben változó.  
- **Mitigáció:** survivor-bias-free univerzum; delistings és delisting returns kezelése (CRSP-szemlélet). citeturn11search8turn11search9  
- **Deploy gate:** equities long-short / cross-sectional stratégiáknál PIT+survivorship nélkül → **Do not deploy**.  
**Veszély:** High. **Evidencia:** High.

**Anti-pattern: “Adjusted close mindenre jó”**  
- **Miért bukás:** corporate action kezelés félreértése; total return vs tradable price; duplázott/dividend leakage jellegű hibák. Corporate action policy-ket index-metodológiai dokumentumok is részletesen tárgyalják, mert a számítási kontinuítás nem egyenlő a tradability-vel. citeturn11search3turn11search7  
- **Mitigáció:** explicit “as-traded” vs “total return adjusted” sorozatok; corporate action eventek PIT idősorban.  
- **Deploy gate:** corp action policy dokumentálatlan → **Do not deploy**.  
**Veszély:** High. **Evidencia:** Medium–High.

**Anti-pattern: “Macro adat = FRED final, kész”**  
- **Miért bukás:** macro adat felülírás/revision; a vintage számít. Croushore–Stark real-time dataset ezt empirikusan vizsgálja. citeturn2search6turn2search2  
- **Mitigáció:** vintage tárolás; “as released” idősor.  
- **Deploy gate:** macro-driven signal “final” adaton validálva → **Do not deploy** (ha a signal pont a release környékén kereskedne).  
**Veszély:** High. **Evidencia:** High.

**Anti-pattern: “Bar-on-close fill” (különösen intraday)**  
- **Miért bukás:** fill realitás hiánya; mikrostruktúrában a spread és adverse selection strukturális, a matching price-time priority és queue számít. citeturn13search0turn6search18turn6search1  
- **Mitigáció:** trade/tick alapú szimuláció; venue matching modell; conservative fill assumptions; implementation shortfall mérés. citeturn6search12turn6search0  
- **Deploy gate:** mikrostruktúra-érzékeny stratégiában bar-fill → **Do not deploy**.  
**Veszély:** Critical. **Evidencia:** High.

### Mely metrikák a leggyakrabban félrevezetők?

**Sharpe ratio “nyersen”:** SR becslési bizonytalansága és feltételei (IID, normalitás, autocorrelation) sérülhetnek; Lo részletesen tárgyalja a Sharpe statisztikáját, Bailey–Lopez de Prado pedig PSR/DSR jellegű korrekciókat ad (nem-normalitás + szelekció). citeturn12search8turn12search1  
**Evidencia:** High. **Minősítés:** Usually overrated (mint egyetlen döntési metrika).

**Profit factor / win rate:** könnyen manipulálható trade slicinggel; nem bünteti jól az impactot és tail risket. (Evidencia inkább gyakorlati; akadémiai közvetlen forrás limitált.)  
**Evidencia:** Evidence limited. **Minősítés:** Usually overrated.

**Backtest t-statok “faktor-zoo” környezetben:** HLZ szerint a sok hipotézis/faktor miatt a szignifikancia-küszöbnek emelkednie kell. citeturn0search14turn0search2  
**Evidencia:** High. **Minősítés:** Dangerous, ha nincs korrekció.

---

## AI decision memo

Ez a rész **külön fejezetként** értékeli a rule-based, supervised ML, deep learning és RL rendszereket. A cél: mikor adnak *valódi* előnyt, és mikor csak komplexitást/instabilitást.

### Rule-based rendszerek

**Hol adhat valódi előnyt:**  
- Alacsonyabb frekvencián (daily+/swing) a jól specifikált, költség-tudatos szabályrendszer gyakran robusztusabb és auditálhatóbb, mint egy feketedoboz; rezsimváltásnál könnyebb “kill criteria”-t definiálni. (Evidencia: inkább engineering és falszifikációs logika; SRE “simplicity” elv támogatja az egyszerűbb rendszerek megbízhatóságát.) citeturn8search9  
**Hol bukik:** túl sok kézi paraméter → data snooping; validáció nélkül “indicator-zoo”. citeturn16search0turn0search4  
**Adatigény:** közepes. **Interpretálhatóság/karbantarthatóság:** magas.  
**Költség- és execution-érzékenység:** a stratégia típusától függ; spread-capture szabályoknál magas. citeturn13search0turn6search18  
**Tipikus failure mode:** “ragasztott” szabályok rezsimváltáskor; túloptimalizált küszöbök. citeturn3search0turn0search4  
**Verdict:** **Baseline-nak kötelező**.  
**Minősítés:** Must-have (baseline). **Evidencia:** Medium.

### Supervised ML modellek

**Hol adhat valódi előnyt:**  
- Nagy dimenziójú prediktorhalmazoknál, nemlineáris interakciók esetén: Gu–Kelly–Xiu a cross-sectional asset pricing problémában széles ML összehasonlítást ad, és több módszernek prediktív előnyt tulajdonít. citeturn4search0turn4search7turn4search3  
**Hol csak komplexitást ad:**  
- Ha a label-horizont rövid és a mikrostruktúra dominál, a supervised ML edge könnyen elolvad a fill/cost driftben.  
- Ha a minta kicsi (kevés rezsim) → overfit; PBO és multiple testing irodalom szerint a szelekciós torzítás könnyen extrém. citeturn0search4turn16search0turn0search14  
**Adatigény:** magas (különösen, ha rezsimállóságot várunk).  
**Interpretálhatóság:** közepes (tree/linear jobb; black-box rosszabb).  
**Karbantarthatóság:** közepes–alacsony (drift, retrain, feature változások).  
**Költség- és execution-érzékenység:** gyakran alulbecsült; a monetizálhatóság kulcsa a turnover és impact modell. citeturn6search12turn14search0  
**Tipikus failure mode:** leakage (target/feature/cross-sectional), túl sok tuning, instabil out-of-sample. citeturn2search7turn0search4turn16search6  
**Minimális bizonyíték küszöb (javasolt):**  
- rolling-origin / walk-forward OOS + data snooping kontroll (Reality Check/SPA) + cost-aware OOS PnL;  
- DSR/PSR-szerű “deflated” skill vagy legalább Sharpe statisztikai bizonytalanság (Lo 2002) figyelembevétele. citeturn16search0turn5search5turn12search8turn12search1  
**Verdict:** **Situational** – sokszor indokolt, de csak szigorú bizonyítékkal.

### Deep learning modellek

**Hol adhat valódi előnyt:**  
- Nagy adatmennyiségnél és összetett nemlinearitásnál; különösen, ha strukturált + unstrukturált adatot kombinál (pl. szöveg/sentiment). A sentiment irodalomban van bizonyíték média tartalom és piac kapcsolatára (Tetlock), de a robust monetizálás erősen feltételes. citeturn15search4turn15search0  
**Hol hátrány:**  
- Minta/rezsim kevés → instabil; interpretálhatóság alacsony; drift esetén “silent failure” kockázat.  
**Adatigény:** nagyon magas.  
**Rezsimérzékenység:** magas.  
**Költség-érzékenység:** gyakran indirekt (DL “jel” előállít, de execution megeszi).  
**Failure mode:** “representation leakage”, tuning, nonstationarity. citeturn3search0turn0search4  
**Verdict:** **Usually overrated** retail/korlátozott adat környezetben; **Situational** intézményi adat-méretnél.

### Reinforcement learning rendszerek

**Hol adhat valódi előnyt:** elsősorban **szekvenciális döntési problémákban**, ahol a reward közvetlenül “implementation shortfall / execution cost” jellegű: pl. execution optimalizáció. Friss szakirodalom RL-alapú execution benchmarkingról számol be és összehasonlítási keretet ad. citeturn4search10turn6search12  
**Hol különösen veszélyes:** “alpha hunting” RL-lel realistátlan szimulátorban → **simulation-to-live gap**; a DRL trading survey-k gyakori problémákként tárgyalják a modellezési feltételezések és validáció korlátait. citeturn4search2turn4search9  
**Reward function design (kötelező kritika):**  
- **Sparse vs dense reward:** PnL-alapú reward zajos és nagy varianciájú; reward shaping segíthet, de torzíthat (a reward választás kritikus, rossz design suboptimális viselkedéshez vezethet). citeturn4search15turn4search2  
- **Transaction cost-aware training:** RL-nél nem opcionális; korai RL trading munkák is hangsúlyozzák a transaction cost szerepét. citeturn4search5turn6search12  
- **Off-policy vs on-policy:** off-policy módszereknél a distribution shift és “behavior policy” mismatch fókusz-kockázat; finance-ben a piac nem-stacionárius (rezsimváltás). citeturn3search0turn4search2  
**Adatigény:** extrém magas, különösen ha sok akció-lehetőség van és cost-aware reward kell.  
**Interpretálhatóság:** alacsony.  
**Karbantarthatóság:** alacsony–közepes (simulátor karbantartása is része).  
**Tipikus failure mode:** reward hacking; túlilleszkedés a szimulátor mikrostruktúrájára; élőben eltérő fill/latency/fee. citeturn4search2turn6search18  
**Minimális bizonyíték küszöb (javasolt):**  
- high-fidelity szimulátor + out-of-simulator stressztesztek;  
- cost-aware reward;  
- shadow mode hosszú ideig, külön drift-méréssel. citeturn4search2turn8search6turn0search4  
**Verdict:** **Dangerous / high false-positive risk** alpha stratégiákhoz; **Situational** execution optimalizációhoz.

### Mikor legyen a végső döntés: “AI not justified”

**AI not justified**, ha bármelyik igaz:  
- A baseline (egyszerű rule/linear/tree) már hoz hasonló OOS és költség utáni eredményt, miközben az AI modell instabil. (SRE “simplicity” elv + drift kockázat.) citeturn8search9turn3search0  
- Nincs data-snooping kontroll és PIT bizonyíték, vagy a modell csak “tuninggal” él. citeturn16search0turn0search4  
- A javulás csak *pre-cost backtestben* látszik; cost/impact után eltűnik. citeturn6search12turn14search0  
- Nem tudod auditálni, miért romlott el (interpretálhatóság/monitoring hiánya). citeturn8search2turn8search6  

---

## Live-trading readiness checklist

**Fontos:** mivel nem auditáltuk a konkrét rendszeredet, a PASS/FAIL/UNKNOWN mezőt **alapértelmezetten UNKNOWN-ra** állítom. A checklist célja: ha bármelyik **Critical** tétel FAIL, az a szakmai minimum szerint **“do not deploy”**.

1) **Adatforrások licenc és “data entitlement” tisztázott** – jogi/ops kockázat. **Status:** UNKNOWN. **FAIL fix:** csak engedélyezett feed. **Súlyosság:** High.

2) **Minden időbélyeg UTC-ben, explicit timezone kezelés** – időeltolás = look-ahead. **Status:** UNKNOWN. **Fix:** UTC standard + tesztek. **Súlyosság:** High. citeturn9search0turn9search2  

3) **Clock sync / timestamp pontosság mérve és dokumentált** – sorrendiség, késleltetés, audit. **Status:** UNKNOWN. **Fix:** NTP/PTP fegyelem + mérés. **Súlyosság:** High. citeturn9search0turn9search2  

4) **Point-in-time adatmodell (PIT) minden nem-ár adatnál** – fundamental/macro/sentiment. **Status:** UNKNOWN. **Fix:** PIT store + as-of join. **Súlyosság:** Critical. citeturn2search7turn2search6turn2search4  

5) **As-of join szabály engine-szinten enforced** (legközelebbi megelőző érték). **Status:** UNKNOWN. **Fix:** DB ASOF JOIN vagy egyenértékű logika. **Súlyosság:** Critical. citeturn2search7turn2search15  

6) **Survivorship-mentes universe (equities)**, delistings kezelése. **Status:** UNKNOWN. **Fix:** CRSP-szerű vagy ekvivalens adat. **Súlyosság:** Critical. citeturn2search4turn11search8  

7) **Delisting returns / delisting value kezelés** – hozam-mérés helyessége. **Status:** UNKNOWN. **Fix:** delisting return beépítése. **Súlyosság:** High. citeturn11search9  

8) **Corporate action policy dokumentált és konzisztens** (split/dividend/rights). **Status:** UNKNOWN. **Fix:** policy + automatikus ellenőrzések. **Súlyosság:** High. citeturn11search3turn11search7  

9) **Macro adatoknál vintage/revision kezelés** (as released). **Status:** UNKNOWN. **Fix:** real-time dataset / vintage store. **Súlyosság:** High. citeturn2search6turn2search14  

10) **Futuresnél roll metodológia explicit** (tradeable vs back-adjust). **Status:** UNKNOWN. **Fix:** roll szabály + teszt. **Súlyosság:** High. citeturn17search18turn17search20  

11) **Futures contract specs (tick, multiplier, session, limits) beégetve**. **Status:** UNKNOWN. **Fix:** venue-spec adat. **Súlyosság:** High. citeturn17search3  

12) **FX universe és venue/LP modell explicit** (OTC sajátosságok). **Status:** UNKNOWN. **Fix:** venue-spec execution modell. **Súlyosság:** Medium. citeturn10search0turn10search9  

13) **Crypto venue-spec szabályok + fragmentáció kezelése** (24/7, külön fee, liquidity). **Status:** UNKNOWN. **Fix:** venue-by-venue risk + data checks. **Súlyosság:** Medium. citeturn10search2  

14) **Backtest engine determinisztikus (seed, ordering, fill)**. **Status:** UNKNOWN. **Fix:** determinisztikus scheduler. **Súlyosság:** High.

15) **Live/backtest parity tesztek** (ugyanaz a jel ugyanarra a döntésre jut). **Status:** UNKNOWN. **Fix:** replay + golden tests. **Súlyosság:** Critical. citeturn0search4  

16) **Unrealistic fill tiltás mikrostruktúra-érzékeny stratégiáknál**. **Status:** UNKNOWN. **Fix:** conservative fill + L1/L2/tick sim. **Súlyosság:** Critical. citeturn6search18turn13search0  

17) **Transaction cost modell: spread+fees+rebates**. **Status:** UNKNOWN. **Fix:** venue fee schedule integráció. **Súlyosság:** High. citeturn6search7  

18) **Market impact modell legalább szcenárió-szinten**. **Status:** UNKNOWN. **Fix:** impact back-of-envelope + stressz. **Súlyosság:** High. citeturn1search0turn14search0  

19) **Implementation shortfall mérés élőben és backtestben**. **Status:** UNKNOWN. **Fix:** IS KPI dashboard. **Súlyosság:** High. citeturn6search12turn6search0  

20) **Latency mérés (market data → decision → order ack → fill)**. **Status:** UNKNOWN. **Fix:** end-to-end tracing. **Súlyosság:** High. citeturn8search2turn9search0  

21) **Order types & TIF komplett és tesztelt**. **Status:** UNKNOWN. **Fix:** venue-spec compliance teszt. **Súlyosság:** High. citeturn6search18turn6search1  

22) **Matching priority ismerete és modellezése** (FIFO/pro-rata). **Status:** UNKNOWN. **Fix:** venue matching doc alapján. **Súlyosság:** High. citeturn6search1turn6search5  

23) **Throttling/rate limit kezelése** – order spam = venue/broker tiltás. **Status:** UNKNOWN. **Fix:** rate limit + backoff. **Súlyosság:** Medium.

24) **Self-trade prevention / wash trade kontroll**. **Status:** UNKNOWN. **Fix:** STP beállítás/logic. **Súlyosság:** High.

25) **Pre-trade risk checks (max size, max notional, fat finger)**. **Status:** UNKNOWN. **Fix:** hard gate a routerben. **Súlyosság:** Critical. citeturn1search18turn1search3  

26) **Kill switch: azonnali order cancel capability**. **Status:** UNKNOWN. **Fix:** “one action cancels all” implementáció. **Súlyosság:** Critical. citeturn8search7turn8search0  

27) **Circuit breaker: daily loss limit → trading stop**. **Status:** UNKNOWN. **Fix:** risk engine rule. **Súlyosság:** Critical.

28) **Position limit/exposure limit per instrument/asset class**. **Status:** UNKNOWN. **Fix:** limit table + enforcement. **Súlyosság:** High.

29) **Drawdown control + stop criteria** (mikor áll le a stratégia). **Status:** UNKNOWN. **Fix:** policy + automatikus. **Súlyosság:** High.

30) **Portfolio-szintű kockázat: korreláció/stressz**. **Status:** UNKNOWN. **Fix:** stress test rezsimenként. **Súlyosság:** High. citeturn3search0  

31) **Capacity estimate (impact-alapú)** és max AUM/size szabály. **Status:** UNKNOWN. **Fix:** impact szcenáriók. **Súlyosság:** High. citeturn14search0turn3search11  

32) **Edge decay/crowding monitor** (ha factor/flow jellegű). **Status:** UNKNOWN. **Fix:** crowding proxy + limit. **Súlyosság:** Medium–High. citeturn3search11turn3search7  

33) **Validáció: rolling-origin / walk-forward dokumentált**. **Status:** UNKNOWN. **Fix:** pipeline beépítés. **Súlyosság:** High. citeturn16search6turn16search2  

34) **Validáció: data snooping kontroll (Reality Check/SPA vagy ekvivalens)**. **Status:** UNKNOWN. **Fix:** RC/SPA tesztek. **Súlyosság:** Critical. citeturn16search0turn5search5  

35) **Validáció: PBO/CSCV vagy ekvivalens szelekciós kockázatbecslés**. **Status:** UNKNOWN. **Fix:** CSCV implementáció. **Súlyosság:** High. citeturn0search4turn5search4  

36) **Metri kák: Sharpe bizonytalanság és szelekció korrekció** (PSR/DSR irány). **Status:** UNKNOWN. **Fix:** PSR/DSR számítás. **Súlyosság:** Medium–High. citeturn12search8turn12search1  

37) **Train/test contamination audit** (adat pipeline szinten). **Status:** UNKNOWN. **Fix:** lineage + unit teszt. **Súlyosság:** High.

38) **Experiment tracking: adatverzió + feature verzió + seed + model artifact**. **Status:** UNKNOWN. **Fix:** MLflow/W&B-szerű rendszer (vagy saját). **Súlyosság:** Medium.

39) **Reproducibility: egy gombnyomásra reprodukálható eredmények**. **Status:** UNKNOWN. **Fix:** containerization + pinned deps. **Súlyosság:** High.

40) **Logging: order lifecycle teljes (intent→send→ack→fill→cancel)**. **Status:** UNKNOWN. **Fix:** structured logs. **Súlyosság:** High.

41) **Reconciliation: broker vs internal positions/cash napi és intraday**. **Status:** UNKNOWN. **Fix:** recon job + alert. **Súlyosság:** Critical.

42) **Monitoring: adat késés/hiány detektálás**. **Status:** UNKNOWN. **Fix:** freshness SLO. **Súlyosság:** High. citeturn8search2  

43) **Monitoring: slippage drift (expected vs realized)**. **Status:** UNKNOWN. **Fix:** IS dashboard. **Súlyosság:** High. citeturn6search12  

44) **Monitoring: risk limit breach alert + auto action**. **Status:** UNKNOWN. **Fix:** auto disable strategy. **Súlyosság:** Critical.

45) **Incident response playbook** (ki mit csinál, kommunikáció, rollback). **Status:** UNKNOWN. **Fix:** incident guide/runbook. **Súlyosság:** High. citeturn8search6turn8search2  

46) **Postmortem kultúra és kötelező root-cause elemzés**. **Status:** UNKNOWN. **Fix:** blameless postmortem template. **Súlyosság:** Medium. citeturn8search6turn8search9  

47) **Shadow mode / paper trading / staged rollout** (canary). **Status:** UNKNOWN. **Fix:** fokozatos rollout. **Súlyosság:** High. citeturn8search2turn8search6  

48) **Broker/exchange failure handling** (disconnect, reject, partial outage). **Status:** UNKNOWN. **Fix:** retry+backoff+safe state. **Súlyosság:** High.

49) **Deployment rollback és config rollback** (verziózott). **Status:** UNKNOWN. **Fix:** release engineering. **Súlyosság:** High. citeturn8search9  

50) **Compliance-alap kontrollok dokumentálása** (különösen EU: RTS 6; US: market access). **Status:** UNKNOWN. **Fix:** kontrollmátrix + audit. **Súlyosság:** Medium–High. citeturn8search0turn1search18  

---

## Final verdict

### Döntés: Research more

**Rövid indoklás:** A kérdés szerint egy általános “Python-alapú trading rendszer” deploy-követelményeit vizsgáltuk, nem egy konkrét implementációt. A fenti evidencia alapján a “top 0.1%” és a “szép backtest, rossz live” közti különbség döntően **data/validáció/költség/execution/ops fegyelem**, amelyet csak konkrét pipeline-audit és live shadow mérések tudnak PASS-ra hozni. A backtest-overfitting és data snooping kockázatok dokumentáltan nagyok, ezért bizonyíték nélkül a “Build now” döntés statisztikailag felelőtlen. citeturn0search4turn16search0turn12search1turn6search12  

**Legnagyobb kockázati tényező:** **hamis monetizálható edge** (PIT/leakage/snooping + költség/impact alulbecslés). citeturn2search7turn0search14turn14search0turn6search12  

**Legfontosabb tipikus hiányosság:** **live/backtest parity + recon + kill switch** (production kontroll). EU RTS 6 “kill functionality” és pre-deployment tesztelés hangsúlyozza, hogy ez nem opcionális. citeturn8search7turn8search0  

**Legvalószínűbb failure mode:**  
- Intraday/HFT: spread/adverse selection + queue/latency miatt a backtest fill “nem létezik”, a valós slippage megeszi az edge-et. citeturn13search0turn6search18turn1search2  
- Daily+: rezsimváltás + publikáció/crowding miatti decay; OOS romlás. citeturn3search0turn0search11turn3search11  

**Mi kellene ahhoz, hogy a verdict egy szinttel jobb legyen (“Build now”):**  
- A checklist **Critical** pontjai mind PASS (különösen PIT, as-of join, cost/impact, execution realitás, kill switch, recon, parity);  
- több, egymástól független OOS útvonalon (rolling-origin) bizonyított, data snooping-korrigált teljesítmény;  
- shadow mode-ban mért slippage/IS drift kontrollált tartományban. citeturn16search6turn16search0turn6search12turn8search6  

---

## Ranked importance list

Az alábbi rangsor **általános** (multi-asset, de nem HFT-specifikus). Zárójelben jelzem, hol változik jelentősen frekvenciával/piaccal.

1) **Adatminőség (PIT correctness)** – ha rossz az adat-idő modell, minden más hamis. (Equities/fundamental/macro különösen.) citeturn2search4turn2search6turn2search7  
2) **Validáció (snooping/overfit kontroll)** – a backtest csak előszűrő; PBO/Reality Check/SPA nélkül a hamis pozitív dominál. citeturn0search4turn16search0turn5search5turn0search14  
3) **Execution (költség + fill realitás)** – intraday/HFT-ben gyakran #1; daily+ esetben is “PnL adó”. citeturn6search12turn1search0turn6search18turn13search0  
4) **Risk management** – túlélés nélkül nincs hosszú távú edge; kill switch és limitek minimumok. citeturn1search18turn8search7turn8search0  
5) **Monitoring** – drift és hiba gyors detektálása; productionben stratégiai. citeturn8search2turn8search6  
6) **Infrastruktúra** – (HFT-ben feljebb kerülhet #2-3 környékére a latency miatt; daily+ esetben lejjebb.) citeturn1search2turn9search0  
7) **Feature engineering** – jó feature > bonyolult modell; de csak PIT-korrekt módon. citeturn2search7turn8search9  
8) **Capital allocation** – capacity/impact miatt nagy pénznél feljebb kerül; kicsinél kevésbé. citeturn14search0turn14search1  
9) **Model complexity** – sokszor negatív korrelációban van a robusztussággal; csak indokolt esetben. citeturn0search4turn8search9  
10) **AI használat** – **nem cél**, hanem eszköz; gyakran overrated, kivéve jól bizonyított, adatgazdag problémáknál (pl. bizonyos cross-sectional predikciók) vagy execution RL. citeturn4search0turn4search10turn4search2  

**Frekvencia/piac szerinti erős eltérés:**  
- **HFT/intraday (equities/futures/crypto CLOB):** execution+microstructure+infra feljebb kerül (matching priority, queue, fee-rezsim). citeturn6search18turn6search1turn1search2  
- **FX (OTC):** venue/LP és adat (nincs “egyetlen tape”) hangsúlyosabb; market structure különbözik. citeturn10search0turn10search9  
- **Daily+/fundamental/macro:** PIT correctness és validáció dominál; a mikrostruktúra részletek kevésbé, de nem nulla. citeturn2search6turn2search4turn16search6