# API reference

Design-review finding (category 26 "Documentation"): "no API documentation beyond
FastAPI's auto-generated OpenAPI." The auto-generated docs (`GET /docs`, Swagger UI; `GET
/redoc`) remain the authoritative, always-up-to-date reference for exact request/response
schemas (every endpoint listed below now has a typed Pydantic `response_model` — see
`backend/models/schemas.py`) — this document adds the context that OpenAPI can't express:
which routes require authentication, which ones mutate live system state, and why.

Base URL (local dev): `http://127.0.0.1:8000`. All routes below are relative to that.

## Authentication

State-mutating routes require an `X-API-Key` header **when** `RESCUEWAY_API_KEY` is set on
the server (unset by default — see `backend/routers/auth.py` and `DEPLOYMENT.md`'s
"Security posture" section). Every route below is marked 🔒 if it's gated this way, and 🔓 if
it's always open (read-only routes, kept open so the live dashboard doesn't need the secret
embedded in every viewer).

## Live state

| Route | Method | Auth | Description |
|---|---|---|---|
| `/ws/live` | WebSocket | 🔓 | Broadcasts a full state snapshot on every signal-cycle tick and on every emergency event. This is what the dashboard actually renders from — every other GET route below is a point-in-time snapshot for polling/debugging, not the primary data path. |
| `/` | GET | 🔓 | Root info (`RootResponse`). |
| `/health` | GET | 🔓 | Liveness check — used by both Dockerfiles' `HEALTHCHECK`. |
| `/arduino/status` | GET | 🔓 | Whether the serial connection to the Arduino is live or the system is in simulation-fallback mode. |
| `/config` | GET | 🔓 | Current runtime configuration snapshot (`config.get_config_summary()`). |

## Emergency / zone control

| Route | Method | Auth | Description |
|---|---|---|---|
| `/zone/select` | POST | 🔒 | Triggers an emergency: `{"zone": 1-4, "traffic": "low"\|"medium"\|"high"\|"rush_hour"}`. This is the actual "an ambulance was detected in zone N" event — everything downstream (EmergencyManager preemption/queueing, the DQN's shielded decision, the Arduino signal change) starts here. |
| `/zone/clear` | POST | 🔒 | Manually clears the active emergency. |
| `/zone/status` | GET | 🔓 | Current zone/signal/queue state (`ZoneStatusResponse`). |
| `/zone/emergency-queue` | GET | 🔓 | The full ETA-priority queue (active + waiting ambulances). |
| `/zone/emergency-stats` | GET | 🔓 | Aggregate emergency-handling statistics (response times, preemption/merge counts). |

## Simulation control

| Route | Method | Auth | Description |
|---|---|---|---|
| `/simulation/status` | GET | 🔓 | Current simulator state (traffic pattern, arrival rates, queue lengths, timings). |
| `/simulation/pattern` | POST | 🔒 | Change traffic pattern at runtime: `{"pattern": "low"\|"medium"\|"high"\|"rush_hour"}`. |
| `/simulation/arrival-rates` | POST | 🔒 | Set custom per-zone vehicle arrival rates. |
| `/simulation/timings` | POST | 🔒 | Update green/yellow/emergency signal durations (server-side clamped to safe ranges — see `simulation.py`'s `*_RANGE` constants). |
| `/simulation/reset` | POST | 🔒 | Reset the simulator and metrics collector to initial state. |
| `/simulation/presets` | GET | 🔓 | Available traffic-pattern presets and valid timing ranges. |
| `/simulation/gps` | GET | 🔓 | Current ambulance GPS/zone-model status. |

## Benchmarking

| Route | Method | Auth | Description |
|---|---|---|---|
| `/compare/run` | POST | 🔒 | Runs the flagship 7-controller benchmark (`evaluation/comparator.py`) — CPU-bound, offloaded to a worker thread so it doesn't block `/ws/live`. Query params: `episodes`, `emergency_probability`, `pattern`. |
| `/compare/results` | GET | 🔓 | Most recent benchmark results (`CompareResultsResponse`). |
| `/compare/history` | GET | 🔓 | Historical benchmark runs from the database. |
| `/eval/live` | GET | 🔓 | Real-time metrics from the running `TrafficController`. |
| `/eval/trends` | GET | 🔓 | Recent metric trends for dashboard charts. |
| `/eval/summary` | GET | 🔓 | Summary evaluation statistics (`EvalSummaryResponse`). |
| `/eval/benchmark` | POST | 🔒 | **Deprecated** — redirects to the same pipeline `/compare/run` uses; kept for backward compatibility. |
| `/eval/stress-test` | POST | 🔒 | Runs the RL controller across all traffic patterns for comparison. |

## Logs and data

| Route | Method | Auth | Description |
|---|---|---|---|
| `/logs/recent` | GET | 🔓 | Recent log entries, filterable by level/category. |
| `/logs/summary` | GET | 🔓 | Logging system summary. |
| `/logs/counts` | GET | 🔓 | Event counts by category. |
| `/logs/export` | POST | 🔒 | Export all logs to a JSON file on disk. |
| `/logs/download` | GET | 🔓 | Download the main log file (`404` if it doesn't exist yet — not a `200` with an error body, see `backend/ARCHITECTURE.md` Part 8.12). |
| `/logs/emergencies` | GET | 🔓 | Emergency-specific log entries. |
| `/logs/errors` | GET | 🔓 | Error-level log entries. |
| `/data/stats` | GET | 🔓 | Database-level statistics (`DataStatsResponse`). |
| `/data/emergencies` | GET | 🔓 | Raw emergency records from the database. |
| `/data/decisions` | GET | 🔓 | Raw decision records from the database. |
| `/data/benchmarks` | GET | 🔓 | Raw benchmark-run records from the database. |

## Example: triggering an emergency with auth enabled

```bash
curl -X POST http://127.0.0.1:8000/zone/select \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RESCUEWAY_API_KEY" \
  -d '{"zone": 2, "traffic": "medium"}'
```

## Interactive docs

`GET /docs` (Swagger UI) and `GET /redoc` are always available and reflect the exact,
current Pydantic schemas — use those for the authoritative request/response shape of any
route; this document is for the *why*, not a schema mirror that can drift out of sync.
