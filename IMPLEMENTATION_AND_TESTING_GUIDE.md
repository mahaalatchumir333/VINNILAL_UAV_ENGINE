# Step-by-Step Implementation & Testing Guide
## UAV Aero-Engine Digital Twin — MVP (Problem Statement 26054)

This guide walks through every file in `uav_digital_twin.zip` in the order
you should build/verify it, what goes IN, what comes OUT, why it matters
to the overall project, and a concrete test you can run at each stage to
prove that piece works before moving to the next.

The build order matters: **each layer depends on the one before it.**
Building bottom-up (simulator → twin → analytics → backend) means you can
test every module in isolation with plain Python, before any web server
or dashboard is involved.

```
Layer 1: simulator/        → generates labelled telemetry
Layer 2: digital_twin/     → predicts expected values, computes residuals
Layer 3: analytics/        → features → anomaly → fault → health → RUL
Layer 4: backend/          → wires 1-3 together, exposes API/WebSocket
```

---

## Step 0 — Prerequisites

**Needed:**
- Python 3.10+
- pip

**Install:**
```bash
unzip uav_digital_twin.zip
cd uav_digital_twin
pip install -r requirements.txt
```
`requirements.txt` installs: `fastapi`, `uvicorn`, `pydantic>=2`, `numpy`,
`scikit-learn`. Nothing else is required for the MVP.

**Verify:**
```bash
python3 -c "import fastapi, numpy, sklearn; print('deps OK')"
```
Expected output: `deps OK`

---

## Step 1 — `simulator/fault_injection.py`

**What it does:** Maintains hidden degradation variables (0 = healthy,
1 = fully failed) that ramp up over a scheduled time window. This is what
makes the synthetic data *labelled* and *scenario-based* instead of random.

**Input:** Nothing at construction. You call `.schedule(FaultConfig(...))`
to arm a fault, then `.advance(dt)` every simulation tick.

**Output:** Internal state (`injector_degradation`, `cooling_efficiency_loss`,
etc., each 0.0–1.0) plus two lookup methods:
- `active_fault_label()` → string, the ground-truth fault name
- `true_health_index()` → float 0–100, ground-truth health

**Why it matters:** Every fault-detection metric (precision, recall,
detection lead time) in the whole project depends on this ground truth
being trustworthy. If this file is wrong, your evaluation numbers are
meaningless even if the AI looks like it's working.

**Test it:**
```bash
cd simulator
python3 -c "
from fault_injection import injector_degradation_scenario
fs = injector_degradation_scenario(start_s=10, ramp_s=20)
for t in [0, 10, 15, 20, 30, 40]:
    fs.t = t - 1  # advance() increments t, so pre-set then call advance(1)
    fs.advance(1)
    print(f't={t:>3}s  injector_degradation={fs.injector_degradation:.2f}  label={fs.active_fault_label()}')
"
```
**Expected output pattern:**
```
t=  0s  injector_degradation=0.00  label=none
t= 10s  injector_degradation=0.00  label=none
t= 15s  injector_degradation=0.20  label=injector_degradation
t= 20s  injector_degradation=0.40  label=injector_degradation
t= 30s  injector_degradation=0.80  label=injector_degradation
t= 40s  injector_degradation=0.80  label=injector_degradation
```
(Values cap at `max_severity=0.8` for this factory's default — that's expected, not a bug.)
If degradation stays at 0 the whole time, the ramp window or `start_s`
is misconfigured — fix before moving on.

---

## Step 2 — `simulator/mission_profiles.py`

**What it does:** Defines mission phases (start, takeoff, climb, cruise,
loiter, throttle transition, descent, shutdown) as time-ordered segments,
each with a throttle level and altitude. This is the "operating scenario"
context the physics model needs.

**Input:** None to construct (factory functions like
`make_endurance_mission()` build a ready profile). At runtime you call
`.state_at(t)` and `.altitude_at(t)` with elapsed seconds.

**Output:** `(MissionPhase, throttle)` tuple and altitude in feet for any
timestamp `t`.

**Why it matters:** Without a mission profile, the simulator would only
ever produce one flat operating point. Mission phases are what let you
later show "the same engine health looks different under Mission A vs
Mission B" — one of the project's stated differentiators.

**Test it:**
```bash
python3 -c "
from mission_profiles import make_endurance_mission
m = make_endurance_mission()
for t in [0, 60, 200, 500, 20000, 28590]:
    phase, throttle = m.state_at(t)
    alt = m.altitude_at(t)
    print(f't={t:>6}s  phase={phase.value:<20} throttle={throttle:.2f}  alt={alt}ft')
print('mission duration (s):', m.duration())
"
```
**Expected output pattern:** phase should progress start → takeoff →
climb → cruise → ... → shutdown as `t` increases, and duration should be
~28620 seconds (the endurance mission's total length).

---

## Step 3 — `simulator/engine_simulator.py`

**What it does:** Combines the mission profile + fault state into one
telemetry packet per tick, adding realistic sensor noise. This is the
**single most important upstream input** for everything downstream.

**Input:** A `MissionProfile` object, an optional `FaultState`, an
optional random `seed` (for reproducibility), and `dt` (seconds per step)
passed to `.step()`.

**Output:** A dict telemetry packet — `timestamp`, `engine_id`,
`mission_id`, `rpm`, `throttle`, `cht`, `egt`, `oil_pressure`,
`oil_temperature`, `fuel_flow`, `vibration`, `battery_voltage`,
`alternator_current`, `injection_timing`, `altitude`,
`ambient_temperature`, plus `_ground_truth_fault` and
`_ground_truth_health` (evaluation-only, never sent to the AI layer).

**Why it matters:** This is the closest thing the prototype has to a
"physical engine." Every other module either consumes its output directly
or consumes something derived from it.

**Test it (normal engine, no fault):**
```bash
python3 -c "
from engine_simulator import EngineSimulator
from mission_profiles import make_normal_cruise_mission

sim = EngineSimulator(seed=1)
mission = make_normal_cruise_mission()
for i in range(5):
    pkt = sim.step(mission)
    print(pkt['rpm'], pkt['egt'], pkt['cht'], pkt['oil_pressure'], pkt['_ground_truth_fault'])
"
```
**Expected output pattern:** RPM near ~3800–3820 (idle/start phase for the
first 5 ticks), EGT/CHT holding near baseline, `_ground_truth_fault` =
`none` for all 5 rows.

**Test it (with injected fault):**
```bash
python3 -c "
from engine_simulator import EngineSimulator
from mission_profiles import make_endurance_mission
from fault_injection import injector_degradation_scenario

sim = EngineSimulator(seed=1, fault_state=injector_degradation_scenario(start_s=30, ramp_s=60))
mission = make_endurance_mission()
for i in range(100):
    pkt = sim.step(mission)
    if i % 20 == 0:
        print(i, pkt['egt'], pkt['fuel_flow'], pkt['_ground_truth_fault'], round(pkt['_ground_truth_health'],1))
"
```
**Expected output pattern:** `_ground_truth_fault` switches from `none`
to `injector_degradation` between i=20 and i=40, EGT jumps from ~570 to
~760+ and fuel_flow rises from ~14 to ~23+, `_ground_truth_health` falls
from 100 toward 30-60 by i=80.

---

## Step 4 — `digital_twin/physics_model.py`

**What it does:** Given operating conditions ONLY (throttle, altitude,
ambient temperature — no fault state), predicts what a **healthy** engine
should read. This is the "twin" — it never sees the fault, by design.

**Input:** `throttle` (0–1), `altitude_ft`, `ambient_temp_c`.

**Output:** A dict of expected values for the same 7 sensors the
simulator produces (`rpm`, `fuel_flow`, `egt`, `cht`, `oil_temperature`,
`oil_pressure`, `vibration`).

**Why it matters:** This is the reference the whole project is built
around. Without a twin prediction, you only have raw sensor values —
which is exactly the "ordinary IoT dashboard" the problem statement
criticizes. The twin is what turns a dashboard into a Digital Twin.

**Test it:**
```bash
cd ../digital_twin
python3 -c "
from physics_model import DigitalTwin
twin = DigitalTwin()
print('idle:  ', twin.predict(throttle=0.0, altitude_ft=0, ambient_temp_c=25))
print('cruise:', twin.predict(throttle=0.65, altitude_ft=6000, ambient_temp_c=25))
print('hot:   ', twin.predict(throttle=0.65, altitude_ft=6000, ambient_temp_c=40))
"
```
**Expected output pattern:** EGT/CHT should be higher at cruise throttle
than idle, and higher again at 40°C ambient than 25°C — if these don't
move in that direction, a coefficient sign is wrong.

---

## Step 5 — `digital_twin/residual_engine.py`

**What it does:** Subtracts the twin's prediction from the simulator's
measured value, for every sensor, then normalizes by a typical
normal-operation noise scale (σ) so residuals across different sensors
become comparable "how many sigmas off" numbers.

**Input:** One telemetry packet (from Step 3) — must contain `throttle`,
`altitude`, `ambient_temperature`, plus the 7 measured sensor values.

**Output:** A `ResidualResult` with `.raw` (physical units, e.g. °C),
`.normalized` (σ units), `.predicted`, and `.measured` dicts.

**Why it matters:** The normalized residual is the single feature that
feeds almost everything downstream — anomaly scoring, fault evidence, and
health index all key off it.

**Test it:**
```bash
python3 -c "
from residual_engine import ResidualEngine
from physics_model import DigitalTwin
import sys; sys.path.append('../simulator')
from engine_simulator import EngineSimulator
from mission_profiles import make_normal_cruise_mission

sim = EngineSimulator(seed=2)
mission = make_normal_cruise_mission()
re = ResidualEngine(twin=DigitalTwin())
for i in range(3):
    pkt = sim.step(mission)
    res = re.compute(pkt)
    print('normalized residuals:', {k: round(v,2) for k,v in res.normalized.items()})
"
```
**Expected output pattern:** All normalized residuals should sit roughly
within ±2 (small noise around 0) since there is no injected fault yet.

---

## Step 6 — `analytics/feature_engineering.py`

**What it does:** Buffers the last ~30 samples (a `FeatureWindow`) and
converts that rolling window into named features: slopes (EGT/CHT/oil
temp trend per second), RMS vibration, RPM fluctuation, fuel-flow/RPM
ratio, oil pressure margin, and the latest normalized residuals.

**Input:** A sequence of `(t, telemetry_dict, normalized_residuals_dict)`
pushed one at a time via `.push()`.

**Output:** `extract_features(window)` → a flat dict of 14 named
features (see `FEATURE_ORDER`), ready for the anomaly detector and fault
classifier.

**Why it matters:** Raw instantaneous sensor values are noisy and don't
show *trends*. A single high EGT reading means nothing; an EGT that has
been climbing for 40 seconds does. This module is what gives the
downstream models trend-awareness instead of point-in-time noise.

**Test it:**
```bash
cd ../analytics
python3 -c "
import sys; sys.path.append('../simulator'); sys.path.append('../digital_twin')
from engine_simulator import EngineSimulator
from mission_profiles import make_endurance_mission
from fault_injection import overheating_scenario
from residual_engine import ResidualEngine
from physics_model import DigitalTwin
from feature_engineering import FeatureWindow, extract_features

sim = EngineSimulator(seed=3, fault_state=overheating_scenario(start_s=10, ramp_s=60))
mission = make_endurance_mission()
re = ResidualEngine(twin=DigitalTwin())
w = FeatureWindow(maxlen=30)
for i in range(80):
    pkt = sim.step(mission)
    res = re.compute(pkt)
    w.push(i, pkt, res.normalized)
    if w.is_ready() and i % 20 == 0:
        f = extract_features(w)
        print(i, 'cht_slope=', round(f['cht_slope'],3), 'egt_residual=', round(f['egt_residual'],2))
"
```
**Expected output pattern:** `cht_slope` turns strongly positive while
the fault is actively ramping (~i=20-40), then eases as the degradation
approaches its maximum severity and the engine settles at a new,
higher-CHT steady state — the trend flattens once the fault stops
worsening, it doesn't need to keep climbing forever.

---

## Step 7 — `analytics/anomaly_detector.py`

**What it does:** Combines the physics-residual score (how far off the
twin prediction is) with an Isolation Forest score (how unusual the
feature vector looks statistically) into one 0–1 anomaly score, then
applies a persistence state machine (Normal→Suspect→Warning→Critical)
so a single noisy sample can't trigger a false alarm.

**Input:** `.fit(list_of_normal_feature_dicts)` once, using ONLY
normal-operation data. Then `.score(features_dict)` per tick.

**Output:** An `AnomalyResult` with `physics_score`, `ml_score`,
`combined_score`, `state` (instant), and `confirmed_state` (persisted).

**Why it matters:** This is the module that decides *when* to raise an
alert at all, and is deliberately built to require 5 consecutive
confirming windows before escalating — directly addressing the "false
alarm" risk called out in the project's own risk register.

**Test it (must train on normal data first):**
```bash
python3 -c "
import sys; sys.path.append('../simulator'); sys.path.append('../digital_twin')
from engine_simulator import EngineSimulator
from mission_profiles import make_normal_cruise_mission
from residual_engine import ResidualEngine
from physics_model import DigitalTwin
from feature_engineering import FeatureWindow, extract_features
from anomaly_detector import AnomalyDetector

# 1. build normal training data
sim = EngineSimulator(seed=9)
mission = make_normal_cruise_mission()
re = ResidualEngine(twin=DigitalTwin())
w = FeatureWindow(maxlen=30)
normal_rows = []
for i in range(300):
    pkt = sim.step(mission)
    res = re.compute(pkt)
    w.push(i, pkt, res.normalized)
    if w.is_ready():
        normal_rows.append(extract_features(w))

det = AnomalyDetector()
det.fit(normal_rows)
print('trained on', len(normal_rows), 'normal windows')

# 2. score a few more normal samples -> should stay 'normal'
for i in range(5):
    pkt = sim.step(mission)
    res = re.compute(pkt)
    w.push(300+i, pkt, res.normalized)
    f = extract_features(w)
    r = det.score(f)
    print('normal check:', r.confirmed_state, round(r.combined_score,3))
"
```
**Expected output:** `trained on ~296 normal windows`, and all "normal
check" rows should print `AnomalyState.NORMAL` with a low combined score
(< 0.4). If normal data is scoring as anomalous, the training set or
thresholds need recalibration.

---

## Step 8 — `analytics/fault_classifier.py`

**What it does:** A transparent, rule-assisted classifier that maps the
feature dict to a likely fault name + confidence + evidence strings
(e.g. "EGT residual elevated", "Fuel-flow/RPM mismatch detected").

**Input:** The same feature dict from Step 6.

**Output:** `FaultResult(likely_fault, confidence, evidence: list[str])`.

**Why it matters:** This is what turns "anomaly score: 0.82" into
something a maintenance engineer can act on. It's the explainability
layer the problem statement explicitly asks for.

**Test it:**
```bash
python3 -c "
from fault_classifier import classify_fault

# hand-craft a feature vector representative of injector degradation
features = {
    'egt_slope': 0.05, 'cht_slope': 0.01, 'oil_pressure_slope': 0.0,
    'oil_temperature_slope': 0.0, 'vibration_rms': 0.1, 'rpm_fluctuation': 20.0,
    'fuel_flow_per_rpm': 0.006, 'oil_pressure_margin': 4.3,
    'egt_residual': 2.4, 'cht_residual': 0.3, 'oil_pressure_residual': 0.1,
    'oil_temperature_residual': 0.2, 'vibration_residual': 0.1, 'fuel_flow_residual': 1.8,
}
r = classify_fault(features)
print(r.likely_fault, r.confidence, r.evidence)
"
```
**Expected output:** `injector_degradation` with confidence around
0.6–0.7 and evidence mentioning EGT residual + fuel-flow/RPM mismatch.

---

## Step 9 — `analytics/health_index.py`

**What it does:** Converts degradation-relevant residuals/features into
five subsystem scores (thermal, lubrication, combustion, vibration,
sensor) and a single weighted overall score (0–100).

**Input:** The feature dict, plus optional `telemetry_quality`
(`"good"`/`"degraded"`/`"poor"`).

**Output:** `HealthResult` with 6 fields, all 0–100.

**Test it:**
```bash
python3 -c "
from health_index import compute_health_index
healthy = {'egt_residual':0.1,'cht_residual':0.1,'oil_pressure_residual':0.0,
           'oil_temperature_residual':0.0,'fuel_flow_residual':0.1,'vibration_rms':0.05}
degraded = {'egt_residual':2.8,'cht_residual':2.0,'oil_pressure_residual':-2.5,
            'oil_temperature_residual':1.5,'fuel_flow_residual':2.0,'vibration_rms':0.6}
print('healthy: ', compute_health_index(healthy))
print('degraded:', compute_health_index(degraded))
"
```
**Expected output:** `healthy` overall should be close to 100; `degraded`
should drop noticeably (well under 60).

---

## Step 10 — `analytics/rul_estimator.py`

**What it does:** Keeps a rolling window of overall health values,
fits a straight-line trend, and projects when health will cross a
failure threshold (default 30). Always tags output with a confidence
level and an explicit assumption disclaimer.

**Input:** `.update(t_seconds, health_index)` called every tick.

**Output:** `RULResult(estimated_minutes, confidence, assumption,
degradation_rate_per_min)`.

**Test it:**
```bash
python3 -c "
from rul_estimator import RULEstimator
est = RULEstimator()
# simulate a steadily declining health index
health = 90.0
for i in range(15):
    health -= 3.0  # 3 points per tick
    r = est.update(i*60, health)  # i*60 = minutes-as-seconds spacing
    print(i, round(health,1), r.estimated_minutes, r.confidence)
"
```
**Expected output:** For the first ~9 rows, `estimated_minutes` is
`None` (`insufficient_data`); after 10 rows it should report a finite
number of minutes with `confidence='medium'` or better, decreasing as
health approaches the 30-point threshold.

---

## Step 11 — `backend/schemas.py` and `backend/main.py`

**What it does:** `schemas.py` is the frozen team contract (Pydantic
models for telemetry in, analytics out). `main.py` wires every module
above into one pipeline (`run_analytics_pipeline`), trains the anomaly
detector on startup, stores results in SQLite, and exposes REST +
WebSocket endpoints.

**Input:** Raw telemetry dict (validated against `TelemetryPacket`)
arriving over WebSocket, or a `MissionSimulationRequest` over REST.

**Output:** `AnalyticsResponse`-shaped JSON, stored + streamed.

**Test it — start the server:**
```bash
cd ../backend
uvicorn main:app --reload --port 8000
```
Expected startup log: no traceback, ending in
`Application startup complete.`

**Test it — REST:**
```bash
curl -s http://127.0.0.1:8000/health
```
Expected: `{"status":"ok","time":"..."}`

```bash
curl -s -X POST http://127.0.0.1:8000/api/mission/simulate \
  -H "Content-Type: application/json" \
  -d '{"mission_type":"endurance","duration_minutes":20,"altitude_ft":8000,
       "ambient_temperature_c":36,"starting_health_index":90,
       "fault_scenario":"injector_degradation"}'
```
Expected: a JSON object with `predicted_max_cht`, `predicted_max_egt`,
`predicted_fuel_consumption_l`, `predicted_health_index_end`,
`mission_risk`, `recommendation`.

**Test it — WebSocket (Python client):**
```bash
python3 -c "
import asyncio, websockets, json
async def main():
    async with websockets.connect('ws://127.0.0.1:8000/ws/telemetry') as ws:
        for _ in range(5):
            msg = json.loads(await ws.recv())
            a = msg['analytics']
            print(a['status'], a['health_index'], a['likely_fault'])
asyncio.run(main())
"
```
(`pip install websockets` first if not already installed.)
Expected: five printed rows, status starting at `normal` with health
index near 100.

---

## Full End-to-End Verification Test

This is the single test that proves the *whole loop* works, matching the
project's "Definition of Done":

```bash
cd backend
python3 -c "
import main
from engine_simulator import EngineSimulator
from mission_profiles import make_endurance_mission
from fault_injection import injector_degradation_scenario

sim = EngineSimulator(seed=5, fault_state=injector_degradation_scenario(start_s=50, ramp_s=200))
mission = make_endurance_mission()

first_alert = None
for i in range(320):
    pkt = sim.step(mission, dt=1.0)
    a = main.run_analytics_pipeline(pkt)
    if a['status'] != 'normal' and first_alert is None:
        first_alert = i
        print(f'First alert at t={i}s')
        print('  status:', a['status'])
        print('  health_index:', a['health_index'])
        print('  likely_fault:', a['likely_fault'])
        print('  fault_confidence:', a['fault_confidence'])
        print('  contributing_factors:', a['contributing_factors'])
        print('  ground truth was:', pkt['_ground_truth_fault'])
"
```

**Pass criteria (all must be true):**
1. `first_alert` is not `None` — an alert fires at all.
2. `likely_fault` == `'injector_degradation'` — matches ground truth.
3. `contributing_factors` is non-empty and names EGT/fuel-flow evidence.
4. `health_index` at the alert is measurably below 100 (degradation is
   visible before the alert, not a sudden jump).
5. The alert fires **before** `_ground_truth_health` would have crossed
   the RUL failure threshold (30) — this is your "early warning lead
   time" metric, and you should record the actual number rather than
   assume one.

If all five hold, you have functioning proof of the core claim in the
project brief: *the system detects developing degradation from trends
and residuals, not just threshold crossings.*
