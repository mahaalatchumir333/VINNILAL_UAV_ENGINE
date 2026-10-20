# UAV Aero-Engine Digital Twin — MVP (Problem Statement 26054)

A physics-informed, explainable, mission-aware Digital Twin prototype for
MALE-UAV aero-piston engine health monitoring. This is a **hackathon
decision-support prototype**, not a certified flight-safety system.

## Run it

```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

- `GET  /health` — service liveness check
- `GET  /status` — simulator state and live flight controls
- `GET  /faults` or `/api/simulator/faults` — supported fault types
- `POST /api/simulator/controls` — set phase, throttle, altitude, and ambient temperature
- `POST /api/simulator/inject` — inject a supported live fault
- `POST /clear` or `/api/simulator/clear` — clear injected faults
- `GET  /api/simulator/status` — simulator time and fault state
- `GET  /api/mission/profiles` — available mission scenarios
- `POST /api/mission/simulate` — scenario stress estimate against current health
- `GET  /api/engine/{engine_id}/latest` — latest persisted sample
- `GET  /api/engine/{engine_id}/history?limit=500` — persisted history for replay
- `WS   /ws/telemetry` — validated live telemetry and analytics at 1 Hz

Telemetry outside the configured physical bounds or missing required fields
is marked degraded and excluded from analytics rather than imputed. The
mission scores are prototype comparisons, not flight clearance. Valid
telemetry and analytics are persisted to `telemetry.db`; set
`VINNILAL_DB_PATH` to move the database.

The dashboard compares live measured CHT/EGT with twin-predicted values,
shows prototype subsystem health indices and backend alert evidence, and
supports local CSV scenario replay with a row scrubber. CSV replay stays in
the browser and is not passed through backend analytics. The report and
maintenance-ticket controls download JSON snapshots with up to 5,000 saved
telemetry rows; they do not create server-side maintenance records. The
dashboard footer labels the synthetic and surrogate-data limitations.

## Explainable analytics

Expand **Explain this AI assessment** in the live dashboard for a per-tick
breakdown of:

- Each measured sensor, healthy-twin prediction, raw difference, residual
  scale, and normalized residual in sigma units.
- The physics residual score, ensemble weights, thresholds, persistence
  state, and hard physical limits that were passed or breached.
- Every rule-based fault candidate, candidate vote total, each triggered and
  untriggered rule, its observed feature, threshold, and vote contribution.
- Each subsystem health score, its input drivers, fixed weight, and points
  contributed to overall health.
- The prototype RUL trend, slope uncertainty, confidence qualifier, and
  assumptions.

These explanations describe the code's decision path; they are not causal
proof or a calibrated probability. The live backend uses rule-based fault
diagnosis. When no Isolation Forest model is loaded, anomaly scoring explicitly
falls back to the physics score at 100% weight (rather than scaling that score
down to 60%); the explanation reports both effective and configured weights.
Cooling diagnosis includes a direct CHT residual rule at 3σ, in addition to
the existing slope and EGT rules. Detailed per-tick explanations are sent over
the WebSocket and shown for the current sample; they are deliberately excluded
from every database history row to avoid multiplying storage and JSON export
size. JSON reports include the detailed explanation for the latest live sample
plus compact saved telemetry history.

## Container launch

```bash
docker compose up --build
```

The dashboard is served by Nginx at `http://localhost/`, with API and
WebSocket requests proxied to FastAPI. SQLite data is stored in the
`telemetry-data` named volume.

## Public deployment: Vercel + Render

The dashboard is a static frontend on Vercel; the persistent FastAPI process
and WebSocket stream run on Render. Vercel cannot host this app's long-lived
simulation/WebSocket loop as a static deployment.

1. Push this project to a GitHub repository. `.gitignore` excludes the local
   SQLite database, downloaded report/ticket JSON, Python caches, and local
   environment files; review `git status` before publishing.
2. In Render, create a Blueprint from the repository and select `render.yaml`.
   It creates the `vinnilal-backend` Python web service. Wait for its deploy
   to finish, then copy the service URL (for example,
   `https://vinnilal-backend.onrender.com`). Confirm
   `https://<render-service>/health` returns `{"status":"ok"}`.
3. In Vercel, import the same repository and set **Root Directory** to
   `frontend`. The included `frontend/vercel.json` maps `/` to the dashboard.
   Deploy the project and copy its public Vercel URL.
4. In Render's service settings, set `CORS_ORIGINS` to the exact Vercel origin
   (for example, `https://vinnilal.vercel.app`, without a path), then redeploy.
   For Vercel preview deployments, add their origins as a comma-separated
   list, or use `*` for a public prototype. CORS is not authentication.
5. Open the Vercel site, paste the Render service URL into **BACKEND API URL**,
   and select **CONNECT**. The URL is saved in that browser. HTTP requests use
   HTTPS and telemetry connects using `wss://`; the dashboard retries a lost
   WebSocket connection.

The Render blueprint uses the free plan for a demo. Free services can sleep
when idle, so the first connection may take time to wake and in-memory
simulation state restarts after service restarts. The blueprint stores SQLite
at `/tmp/telemetry.db`; this is ephemeral and history is not durable across
restarts or redeploys. Use a paid persistent disk (and configure
`VINNILAL_DB_PATH` to its mount) or migrate persistence to managed PostgreSQL
before relying on retained history. The public demo has no authentication:
anyone who can reach it can change the shared simulation controls and inject
faults. Do not connect real aircraft or sensitive telemetry.

## External surrogate datasets

The `data/external/` directory contains derived summaries from locally
provided C-MAPSS FD001-FD004 cycle files and an OBD-II CSV. Raw source archives
remain outside the repository. Recreate summaries with:

```powershell
py -m analytics.import_surrogate_benchmarks `
  --cmapss-dir "C:\path\to\C-MAPSS" `
  --obd-csv "C:\path\to\reverse_engineering_obd2.csv" `
  --ims-dir "C:\path\to\IMS-Bearing-Dataset"
```

C-MAPSS stays indexed in engine cycles; it is not relabeled as a 1 Hz
aero-piston record. OBD-II values are aggregated to relative one-second
segments and absolute timestamps/source segment identifiers are omitted. The
summary manifest records provenance and license caveats. These data are
separate surrogate domains: do not map their measurements to UAV engine
labels as if they were equivalent. IMS waveforms are reduced to per-record,
per-channel vibration features; timestamp offsets are retained within each
test while absolute timestamps and source filenames are omitted. The importer
does not create fault/RUL labels. Check the manifest's source/license caveat
and do not claim any surrogate validates UAV engine behavior.

## Project map

```
simulator/       Person 1 — scenario-based telemetry generator + fault injection
digital_twin/     Person 1 — reduced-order physics model + residual engine
analytics/        Person 2 — feature engineering, anomaly detection, fault
                  classification, health index, prototype RUL (AI/ML engineer's
                  core deliverables)
backend/          Person 3 — FastAPI orchestration, WebSocket streaming,
                  SQLite storage, replay + mission-simulation endpoints,
                  shared schemas.py contract
```

## Quick simulator check (no server needed)

```bash
python3 -c "
from simulator.engine_simulator import EngineSimulator
from simulator.mission_profiles import make_endurance_mission
from simulator.fault_injection import injector_degradation_scenario

sim = EngineSimulator(seed=5, fault_state=injector_degradation_scenario(start_s=50, ramp_s=200))
mission = make_endurance_mission()
for i in range(320):
    pkt = sim.step(mission, dt=1.0)
    print(i, pkt['rpm'], pkt['egt'], pkt['cht'])
"
```

## Important labelling discipline

- All RUL output is a **scenario-based prototype estimate**, never presented
  as validated aerospace RUL.
- The physics model is a **reduced-order, physics-informed** demonstrator,
  not a certified thermodynamic model.
- Synthetic telemetry is **scenario-driven and physics-informed**, not
  random noise.
- The system produces **decision support**, never autonomous flight-control
  commands.
