# VINNILAL — Engine Digital Twin Intelligence
### Prototype for DRDO Problem Statement 26054 (MALE-UAV aero-piston engine health monitoring)

This document states exactly what each role built, what data it runs on, and
what has been tested versus what remains a design assumption — so nothing
here is overclaimed to evaluators.

---

## 1. Digital Twin / Simulation Engineer

**Files:** `simulator/engine_simulator.py`, `simulator/mission_profiles.py`,
`simulator/fault_injection.py`, `simulator/generate_datasets.py`,
`digital_twin/physics_model.py`, `digital_twin/residual_engine.py`

**What it does:** A reduced-order, physics-informed engine model predicts
expected sensor behaviour (RPM, EGT, CHT, oil pressure/temperature, fuel
flow, vibration) from operating conditions (throttle, altitude, ambient
temperature) alone — it never sees fault state, by design. A separate
simulator generates measured telemetry with realistic noise and six
independently injectable fault modes (injector degradation, cooling
degradation, lubrication degradation, misfire, vibration increase, EGT
sensor drift), each with a labelled ground-truth severity trajectory.
Faults can now be triggered **live**, mid-stream, via the backend's
`/api/simulator/inject` endpoint — not just pre-scheduled at start.

**Status:** Complete and tested. Misfire's RPM-instability physics was a
real bug found and fixed mid-project (misfire originally never touched RPM
at all).

---

## 2. AI/ML Engineer (Analytics Layer)

**Files:** `analytics/feature_engineering.py`, `analytics/anomaly_detector.py`,
`analytics/fault_classifier.py`, `analytics/ml_fault_classifier.py` +
`train_ml_classifier.py` + `ml_fault_classifier.joblib`, `analytics/health_index.py`,
`analytics/rul_estimator.py`, `analytics/evaluate_metrics.py`,
`analytics/evaluate_ml_classifier_generalization.py`

**What it does:**
- Rolling-window feature engineering (slopes, RMS, cross-sensor ratios) from measured-vs-twin residuals
- Hybrid anomaly detector: physics-residual score + Isolation Forest, with a persistence state machine (Normal→Suspect→Warning→Critical) to suppress single-sample noise
- Two interchangeable fault classifiers: a transparent rule-based one (default, for explainability) and a Random Forest trained on all 12 scenario datasets (toggle via `USE_ML_CLASSIFIER` in `main.py`)
- Subsystem + overall health index (0–100)
- Prototype RUL with a 90% confidence interval, explicitly labelled as a scenario-based estimate, not certified aerospace RUL

**Evaluation status:** the latest completed run of `analytics/evaluate_metrics.py`
before the cooling-rule change produced no alerts on the labeled datasets:
precision/recall/F1 were `0.00 / 0.00 / 0.00`, false alarms were `0/6`, and
detection lead time was not available. The zero false-alarm count is not
meaningful without any detections. This does not reproduce the previously
reported `1.00 / 1.00 / 1.00` full-pipeline metrics; those figures should not
be treated as verified until the evaluator is corrected and rerun. A rerun
after the current changes was stopped after several minutes of CPU work without
producing results. The focused simulator regression and unit tests below do
pass, but do not substitute for held-out scenario metrics.

**Previously reported results (not reproduced in the latest full-pipeline run):**
| Metric | Value |
|---|---|
| Precision / Recall / F1 (rule-based, full pipeline) | 1.00 / 1.00 / 1.00 (not reproduced) |
| False-alarm rate (6 normal/missing-data/network-loss datasets) | 0% (not reproduced; latest run had no detections) |
| Average detection lead time | +165.8s (not reproduced) |
| ML classifier accuracy on **unseen** simulation runs | 98.25% (prior report; not rerun here) |
| ML classifier's weakest class | EGT sensor drift, 88.36% (prior report; not rerun here) |

These historical figures are retained for traceability only and must not be
presented as freshly validated until the end-to-end evaluator reproduces them.

---

## 3. Backend Engineer

**Files:** `backend/main.py`, `backend/schemas.py`

**What it currently does:** FastAPI service that:
- Runs the simulator internally and streams validated telemetry + analytics over `/ws/telemetry` at 1 Hz.
- Exposes `/health`, `/status`, `/faults`, `/api/simulator/faults`, `/api/simulator/controls`, `/api/simulator/inject`, `/api/simulator/clear`, `/clear`, and `/api/simulator/status`.
- Rejects missing, non-finite, and out-of-range telemetry from analytics and reports degraded quality instead of filling in invented sensor values.
- Serves `frontend/vinnilal_dashboard.html` directly at `GET /`.

**Implemented:** `/api/engine/{id}/latest` and `/api/engine/{id}/history` persist and retrieve validated WebSocket telemetry in SQLite. The mission risk API compares the current health estimate against short cruise, hot-weather endurance, and high-altitude reconnaissance scenarios. The `Dockerfile` and `docker-compose.yml` provide a FastAPI + Nginx deployment with persistent SQLite storage.

**External surrogate inputs:** C-MAPSS FD001-FD004, a user-provided OBD-II log, and all three test sets of the user-provided IMS Bearings archive are converted into separate derived files under `data/external/`. C-MAPSS remains cycle-based; OBD data is aggregated to relative one-second bins; IMS vibration waveforms are summarized by record and channel with within-test relative time. Absolute timestamps and source filenames are omitted from the summaries. No fault/RUL labels are invented for IMS. The importer preserves domain boundaries and records source/license caveats. These data are not relabeled as UAV telemetry or evidence of UAV model validation.

**Still outstanding:** Verify source/release terms for the user-supplied surrogate archives before redistribution. None of these surrogate datasets is validated as UAV-engine telemetry.

---

## 4. Frontend Engineer

**Files:** `frontend/vinnilal_dashboard.html` (served at `/`) and `frontend/dashboard.html` (legacy, not served by a route)

**What it currently does:** Live engine readouts, flight phase and throttle/altitude/temperature controls, live fault injection, connection and health status, mission stress estimates, saved telemetry history with a scrub slider, and an illustrative 3D attitude display. It also shows measured-vs-twin CHT/EGT traces, prototype subsystem health bars, an alert/evidence console from backend diagnostics, local CSV scenario replay with a row scrubber, and browser downloads for a telemetry JSON report or maintenance-ticket JSON. The replayed rows are not sent to backend analytics. The 3D pitch/roll are phase-based illustrative values, not measured flight-control telemetry.

**Still not implemented in the served page:** server-side maintenance-ticket workflow, full CSV analytics reprocessing, and JSON/PDF report generation on the backend. The current ticket/report buttons download client-side JSON snapshots and fetch up to 5,000 saved telemetry rows for inclusion.

**Runtime validation caveat:** Docker Compose configuration was parsed, but Docker CLI/daemon was unavailable in the development environment, so image build, container health, Nginx HTTP routing, and proxied WebSocket behavior remain unverified end-to-end.

**Telemetry quality gate behavior:** invalid, missing, non-finite, or out-of-range packets are rejected from analytics and marked degraded. The gate does not clamp RPM or fill missing temperatures with baseline estimates.

**Cooling classifier calibration:** the classifier previously omitted the direct CHT residual and depended on a noisy short-window slope plus EGT residual. Added an explainable 3σ CHT residual rule; it is above the observed healthy-scenario maximum (2.19σ) and below the reproducible severity-0.8 cooling injection residual (about 13.3σ). A seeded 12-tick simulator regression test now confirms the specific `overheating_cooling_degradation` diagnosis on every tick. This is simulator evidence, not held-out operational validation. The dashboard only displays classifier diagnoses/evidence when backend status is non-normal.

**Explainable analytics:** the live WebSocket response includes measured-versus-twin sensor residuals, anomaly-score calculations and decision thresholds, all rule-based fault candidate votes and rule outcomes, weighted subsystem-health calculations, and the RUL trend assumption/uncertainty. The dashboard exposes this in an expandable explanation panel, and the latest sample's detail is included in JSON exports. Per-tick explanation payloads are excluded from SQLite history to keep storage and historical exports bounded; history retains its compact analytics summary.

**Explainability limitations:** this is a transparent rule/threshold trace, not SHAP or causal attribution. Rule confidence is not calibrated probability. No Isolation Forest is loaded in the live backend, so its anomaly score now uses a documented physics-only fallback at 100% effective weight; the XAI payload distinguishes effective weights from configured hybrid weights. All scales and thresholds are prototype assumptions that need independent validation on representative engine data.

---

## Datasets — exactly what this prototype trains and evaluates on

**There is no real MALE-UAV engine telemetry in this project.** Real
aero-piston engine data for this application is restricted/unavailable, as
the problem statement itself anticipates ("simulated or real engine
datasets" are both acceptable). Every number above comes from:

**12 physics-informed, scenario-labelled synthetic datasets**
(`simulator/generate_datasets.py` → `data/*.csv`), each a full-mission
time series (600–900 samples at 1 Hz) with ground-truth fault labels and
health trajectories. Separate derived surrogate summaries are stored in
`data/external/` and must not be interpreted as UAV engine records:

| # | Dataset | Purpose |
|---|---|---|
| 01 | normal_cruise | Baseline / false-alarm testing |
| 02 | high_altitude | Environmental variation |
| 03 | hot_weather_endurance | Environmental variation |
| 04 | rapid_throttle_transition | Transient response |
| 05 | injector_degradation | Fault detection |
| 06 | misfire | Fault detection |
| 07 | lubrication_degradation | Fault detection |
| 08 | cooling_degradation | Fault detection |
| 09 | vibration_increase | Fault detection |
| 10 | egt_sensor_drift | Sensor-fault vs. real-degradation discrimination |
| 11 | missing_telemetry | Resilience testing (~8% random packet loss) |
| 12 | network_interruption | Resilience testing (60s contiguous outage) |

The ML classifier's headline 98.25% accuracy is measured on **freshly
generated runs with new random seeds and randomized fault-onset timing**
(`evaluate_ml_classifier_generalization.py`) — not the same rows used for
training, and not a naive random row-split of one continuous run (which
gave a misleading 100% due to time-series leakage between adjacent
seconds).

**What this means for evaluators:** every accuracy, lead-time, and
false-alarm number in this project is a claim about how the system
behaves on physics-informed simulated data with injected, labelled faults
— not a claim about real engine performance. This is stated explicitly
because it is exactly the distinction DRDO's own problem statement asks
projects not to blur.
