Phase 3 Extended Plan — további optimalizációk

Ez a kör már nem csak a “klasszikus” num_envs / batch_size / parquet / numba vonalat viszi tovább, hanem a teljes training pipeline-t próbálja gyorsítani: policy update, worker memóriahasználat, logging, eval, ETL és laptopos stabilitás oldalról is.

1. Cél

A cél most már nem csak az, hogy picit nőjön az SPS, hanem hogy:

nőjön a sustained training throughput
csökkenjen a RAM- és worker-overhead
stabilabb legyen a hosszú futás
kevesebb idő menjen el nem-train jellegű blokkra
jobban kihasználjátok a gépet anélkül, hogy túlmelegedés vagy fölösleges process churn lenne

A hardver továbbra is egy 8C/16T Ryzen 7 260 alapú laptop platform, szóval a tuningot ehhez kell igazítani, nem desktop/workstation logikával.

2. High ROI bővítések
[NEW] AMP / mixed precision a policy update körül

Érintett: train_agent.py, policy/model init, optimizer step

Mit:

opcionális torch.cuda.amp.autocast
GradScaler FP16 esetén
BF16 próba, ha stabil

Miért:

a rollout nem ettől lesz gyorsabb
viszont a PPO update oldal tud gyorsulni
GPU memóriaterhelés is javulhat

Bekapcsolás:

TRAIN_USE_AMP=1
TRAIN_AMP_DTYPE=fp16|bf16

Mérés:

train step idő
SPS
reward/kl/entropy stabilitás
NaN/instability figyelés
[NEW] torch.compile próba a policy hálón

Érintett: model/policy konstrukció

Mit:

opcionálisan compile-olni a policy/value modellt
csak train módban benchmarkolva

Miért:

PyTorch 2.x esetén néha meglepően jó nyereség
olcsó kipróbálni
de nem garantált, ezért feature flag kell

Bekapcsolás:

TRAIN_TORCH_COMPILE=1
TRAIN_TORCH_COMPILE_MODE=default|reduce-overhead|max-autotune

Mérés:

cold start overhead
rollout idő
update idő
teljes falióra / X timesteps
[NEW] Shared dataset / memmap worker oldalon

Érintett: dataset loader, env init, worker startup

Mit:

a workerek ne tartsanak külön teljes in-memory példányokat
read-only közös háttér:
numpy.memmap
Arrow/Parquet lazy read
shared read-only arrays
fold/symbol szerinti részbetöltés

Miért:

ez lehet az egyik legnagyobb gyakorlati nyereség
kevesebb RAM
gyorsabb worker startup
kevesebb process-fork/spawn költség

Elv:

env csak a szükséges szelethez férjen hozzá
ne duplikálja a teljes feature-táblát workerenként
[NEW] Python object churn csökkentése

Érintett: env stepping, feature access, observation build

Mit:

listák és objektumok helyett:
np.ndarray
float32
kontiguus tömbök
oszlopos / index-alapú elérés
minimalizálni:
property access
dataclass/object wrapping
stepenkénti új objektumképzést

Miért:

RL env steppingnél a Python overhead nagyon sokat számít
főleg több worker mellett
[NEW] Logging overhead visszavágása

Érintett: train loop, audit, debug, callbacks

Mit:

külön választani:
debug log
train summary log
audit/event log
ne legyen stepenkénti nehéz logolás
JSON serializáció csak ritkán
print minimalizálása hot pathban

Miért:

gyakran meglepően sok idő megy el rá
főleg Windows + több processz + fájl I/O esetén
3. Medium ROI bővítések
[NEW] Async eval / checkpoint stratégia

Érintett: eval loop, checkpoint manager

Mit:

ritkább checkpoint
ritkább vagy elkülönített eval
ne blokkolja feleslegesen a fő train loopot

Miért:

a teljes wall-clock training időt csökkenti
nem csak az SPS-t javítja, hanem a “hasznos munkát”

Javaslat:

checkpoint csak fontos mérföldköveknél
eval külön periodikával
full artifact mentés ritkábban
[NEW] CPU affinity / worker pinning

Érintett: worker launcher, multiprocessing startup

Mit:

opcionális affinity a worker processzekre
6–8 worker preferált pinning
1–2 logikai szál maradjon szabad az OS-nek és fő processznek

Miért:

laptopon stabilabb lehet a latency
kevesebb scheduler-zaj
nem mindig hoz sok SPS-t, de kiszámíthatóbbá teheti a futást

Megjegyzés:

ezt csak mérés után érdemes bent hagyni
[NEW] ETL oldalon Polars / PyArrow

Érintett: offline dataset prep, feature precompute, conversion pipeline

Mit:

ahol csak ETL/scan/filter/groupby fut:
polars
pyarrow
pandas maradjon ott, ahol tényleg kell

Miért:

gyorsabb preprocessing
jobb columnar workflow
kisebb memória-overhead sok esetben

Fontos:

ez inkább az előkészítési időn segít, nem a step loopon
[NEW] Feature precompute agresszívebb szétválasztása

Érintett: feature pipeline

Offline menjen:

lassú rolling statok
determinisztikus időalapú feature-ök
minden, ami nem epizódállapot-függő

Online maradjon:

valóban statefüggő elemek
rövid, gyors, olcsó update-ek

Miért:

minél üresebb a hot path, annál jobb a training throughput
4. Measure-first kísérletek
[TRY] Kisebb / hatékonyabb policy háló

Érintett: policy config

Mit:

kisebb hidden size
kevesebb réteg
egyszerűbb input head

Miért:

sokszor jobb a kisebb modell + gyorsabb pipeline
nem biztos, hogy a nagyobb modell többet hoz edge-ben

Csak akkor tartsátok meg, ha:

reward / OOS quality nem romlik érdemben
[TRY] Numba kiterjesztés újabb hotspotokra

Érintett: feature / stat függvények

Mit:

csak profiler után
csak tiszta, numerikus, stabil függvényekre

Miért:

még lehet tartalék
de vakon ne menjen tovább a numba-zás
[TRY] Nagyobb batch_size, több env

Mit:

a korábbi 6 / 8 / 10 / 12 env mérés után
csak akkor tovább, ha:
RAM rendben
worker startup rendben
SPS tényleg nő
nem romlik a stability

Miért:

ezen a ponton már könnyű túlcsúszni a sweet spoton
5. Új konfigurációs kapcsolók

Javasolt új env varok:

TRAIN_USE_AMP=1
TRAIN_AMP_DTYPE=bf16
TRAIN_TORCH_COMPILE=0
TRAIN_TORCH_COMPILE_MODE=default

TRAIN_SHARED_DATASET=1
TRAIN_DATA_BACKEND=memmap   # memmap|parquet|arrow
TRAIN_REDUCE_LOGGING=1
TRAIN_ASYNC_EVAL=0
TRAIN_CPU_AFFINITY=0

TRAIN_NUM_ENVS=8
PPO_BATCH_SIZE=4096
6. Prioritási sorrend
Első körben megcsinálnám
AMP
torch.compile benchmark
shared dataset / memmap
logging overhead csökkentés
async eval/checkpoint ritkítás
Python object churn csökkentés
Második körben
CPU affinity
Polars/PyArrow ETL
agresszívebb offline feature precompute
kisebb policy háló teszt
Csak mérés után
további Numba-bővítés
10–12 env fölé menés
agresszív batch növelés
7. Verification Plan

Minden változtatásnál ugyanazt a rövid benchmark csomagot futtatnám:

Mini benchmark
50k–100k timestep
azonos seed
azonos fold/symbol
azonos reward config
Mérendő
SPS
rollout time
update time
CPU kihasználtság
RAM
VRAM
worker startup idő
reset/fold-switch overhead
training stability
NaN / divergens loss jel
Kimenet

egy rövid táblázat:

baseline
AMP
compile
shared dataset
logging reduced
async eval
kombinált best config
8. Rövid végső ajánlás

Ha csak a legjobb esélyű következő csomagot kell kiválasztani, én ezt vinném tovább:

AMP + shared dataset/memmap + logging visszavágás + async eval/checkpoint + Python object churn csökkentés

Erre jönne rá külön benchmarkként:

torch.compile

Ez szerintem reálisabb és nagyobb eséllyel hasznos, mint az, hogy vakon tovább emeljük az env-számot vagy még több Numba-t dobáljunk mindenre.