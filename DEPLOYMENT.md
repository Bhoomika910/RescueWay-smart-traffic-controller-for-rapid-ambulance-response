# Deployment guide

Design-review finding (category 26 "Documentation" / category 28 "Deployment readiness"):
no Docker support, no deployment documentation. This covers the Docker/Compose setup added
to close that gap — see `backend/ARCHITECTURE.md` for the research/engineering history, and
`CONTRIBUTING.md` for local (non-Docker) development.

## What gets deployed

Two services: the FastAPI backend (RL inference + live signal control) and the React
dashboard (static build served by nginx). SUMO-based training/evaluation scripts
(`backend/sumo/*.py`) are **not** part of this deployment — they're research tooling run on
demand against a separate SUMO install, not an always-on service. See
[`docs/REPRODUCE.md`](backend/docs/REPRODUCE.md) for running those.

## Quick start (Docker Compose)

```bash
cp backend/.env.example backend/.env   # edit RESCUEWAY_API_KEY etc. as needed
docker compose up --build
# backend:  http://localhost:8000  (docs at /docs)
# frontend: http://localhost:5173
```

`docker-compose.yml` builds both images from their respective `Dockerfile`s, wires the
frontend's build-time `VITE_API_BASE_URL` to the backend service, and persists the trained
RL checkpoints / SQLite DB / logs in named volumes so they survive container recreation
(see `backend/.gitignore`'s comment on why checkpoints aren't tracked in git — they're
regenerated deterministically from a seed, not deployment artifacts).

## Configuration

All configuration is environment-variable driven — see `backend/.env.example` and
`frontend/.env.example` for the full list. The two that matter most for anything beyond
local development:

| Variable | Purpose | Default |
|---|---|---|
| `RESCUEWAY_API_KEY` | Enables auth on state-mutating routes (`/zone/select`, `/zone/clear`, `/compare/run`, `/eval/*`, `/logs/export`, `/simulation/*`). **Unset by default** — the server starts with a loud startup-log warning if this is empty; set it before exposing the server beyond localhost. | *(empty — auth disabled)* |
| `RESCUEWAY_FRONTEND` | Comma-separated CORS allow-list. | `http://localhost:5173,http://localhost:5174` |

Generate an API key: `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Every
protected request from a legitimate client needs the `X-API-Key: <value>` header — the
dashboard reads this from its own env if you wire it in (not done by default, since the
default deployment target is a trusted local network, not a public endpoint — see the
"Security posture" section below).

## Security posture — read before exposing this beyond localhost

This is a **research/prototype system**, not hardened for public-internet exposure. What's
actually in place:

- API-key auth on state-mutating routes (opt-in via `RESCUEWAY_API_KEY`) — see
  `backend/routers/auth.py` for the full rationale (a shared secret, not per-user auth,
  since the real deployment has exactly one trust boundary: the local network vs. the
  internet, not multiple distinct human operators).
- CORS restricted to an explicit origin allow-list (`RESCUEWAY_FRONTEND`), not `*`.
- Parameterized SQL throughout (`data/database.py`) — no string-interpolated queries.

What's **not** in place and should not be assumed: rate limiting, TLS termination (put this
behind a reverse proxy — nginx, Caddy, a cloud load balancer — for anything beyond a local
network), request size limits beyond FastAPI/Pydantic's defaults, or any audit logging
beyond the existing operational logs (`backend/logging_system/`). Do not expose the
container's ports directly to the public internet without addressing these.

## Health checks

Both Dockerfiles define `HEALTHCHECK` instructions (`GET /health` for the backend,
`GET /` for the frontend) that `docker compose` and most orchestrators (Kubernetes
readiness/liveness probes, via the same commands) can use directly.

## Hardware (Arduino)

The backend falls back to simulation mode automatically if it can't open the configured
serial port (`RESCUEWAY_ARDUINO_PORT`, default `COM20`) — this is the expected/default state
in a containerized deployment, since a Docker container generally can't see a host serial
device without an explicit device mapping (see `docker-compose.yml`'s commented-out
`devices:` block for the Linux/macOS syntax; Windows containers cannot pass through host COM
ports the same way — run the backend natively on Windows if real hardware control from a
Windows host is required).

## What is NOT verified in this environment

- The Docker images have **not** been build-tested in this session (no Docker daemon
  available in the development environment that produced them) — `docker build` has not
  actually been run against `backend/Dockerfile` or `frontend/Dockerfile`. Both were
  reviewed carefully for correctness (base images, layer ordering, `HEALTHCHECK` syntax,
  build-arg wiring) but this is a real, disclosed gap, not a claim of "tested and working."
  Before relying on this in production, run `docker compose build` and `docker compose up`
  once and confirm both health checks go green.
- The Arduino firmware (`backend/arduino/traffic_controller.ino`) has similarly not been
  compiled or flashed to real hardware in this environment — see
  `backend/ARCHITECTURE.md`'s Part 8.19/Part 9 discussion for the verification that was
  possible (structural static analysis) versus what wasn't (an actual compile).
