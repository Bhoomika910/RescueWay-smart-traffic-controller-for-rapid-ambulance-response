# RescueWay

**![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.13-blue.svg) ![Node](https://img.shields.io/badge/node-18%2B-blue.svg) ![Status](https://img.shields.io/badge/status-research%20prototype-yellow.svg)**

An RL-controlled emergency traffic signal system that clears a 4-way intersection **before**
an approaching ambulance arrives, rather than reacting once it's already waiting. Two RL
agents (tabular Q-Learning and Double DQN, with optional Dueling+PER), a real-time FastAPI
backend, a React dashboard, a SUMO microsimulation environment for realistic training, and
optional Arduino hardware for a physical 4-zone signal rig.

> This is a research/prototype system, not a production traffic-control product — see
> [`LICENSE`](LICENSE)'s scope note and [`DEPLOYMENT.md`](DEPLOYMENT.md)'s security posture
> section before using it as anything more than that.

## Why this exists

Traditional emergency preemption is reactive: a signal only changes after an ambulance is
already detected at (or near) the intersection, so the vehicle still has to slow or stop
while the transition happens. RescueWay's RL agents are trained to *anticipate* — to have
already cleared the ambulance's zone by the time it arrives. `backend/ARCHITECTURE.md` is
the full record of that redesign: what was reactive before, what changed, what was
deliberately left alone, and the actual (not aspirational) experimental results at every
stage.

## Results at a glance

(Full breakdown, methodology, and multi-seed variance figures: `backend/README.md` and
`backend/ARCHITECTURE.md`. Numbers below are the current headline results as of 2026-07-21 —
treat this section as a pointer, not the source of truth, since it will drift if not
re-synced after future retraining.)

| | Emergency accuracy | Avg ambulance delay |
|---|---|---|
| Tabular Q-Learning (25k episodes, seed-stable across 5 seeds) | 56.8% | 0.09s |
| Double DQN (SUMO-native, near-converged) | 84.6% | 0.4s |
| Double DQN + Safety Shield (hard override, guaranteed) | 100.0% | 0.0s |
| Rule-Based oracle reference (perfect information, not a fair baseline) | 100.0% | 0.1s |

The Safety Shield (`rl/safety_shield.py`) is what actually runs in production between the
learned policy and the signal hardware — it guarantees the correct zone is served every tick
during a declared emergency regardless of what the RL policy alone would have chosen, and is
why the deployed system's accuracy is 100% rather than the raw DQN's 84.6%.

## Quick start

**Docker (fastest path to a running system):**
```bash
cp backend/.env.example backend/.env
docker compose up --build
# backend:  http://localhost:8000   frontend: http://localhost:5173
```
See [`DEPLOYMENT.md`](DEPLOYMENT.md) for configuration, security posture, and what has and
hasn't been verified about the container images.

**Local (no Docker):**
```bash
cd backend && pip install -r requirements-dev.txt && uvicorn main:app --reload --port 8000
cd frontend && npm install && npm run dev
```
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full local setup, linting, and test commands.

**Hardware (optional):** the Arduino firmware is at `backend/arduino/traffic_controller.ino`;
the backend automatically falls back to simulation mode if no serial device is present. Pin
map and wiring are in `backend/README.md`.

## Project layout

```
backend/       FastAPI app, RL agents (rl/), simulators (traffic/, sumo/), evaluation
                pipeline (evaluation/), tests, and the full research/engineering log
                (ARCHITECTURE.md)
frontend/      React + Vite dashboard (live signal state, benchmark charts, logs)
```

See the architecture diagram and module map at the top of
[`backend/ARCHITECTURE.md`](backend/ARCHITECTURE.md) for how the pieces connect.

## Documentation

| Doc | What's in it |
|---|---|
| [`backend/ARCHITECTURE.md`](backend/ARCHITECTURE.md) | The full research/engineering history — every redesign, bug found and fixed, and experiment run, in chronological parts, with evidence. Start here for "why does the code look like this." |
| [`backend/docs/API.md`](backend/docs/API.md) | Route reference: auth requirements, what each endpoint does and why. |
| [`backend/docs/REPRODUCE.md`](backend/docs/REPRODUCE.md) | Exact commands to regenerate every result cited in this project. |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | Docker/Compose setup, configuration, security posture. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Local dev setup, pre-PR checklist, code-quality tooling, test philosophy. |

## License

MIT — see [`LICENSE`](LICENSE), including the scope note on why this isn't safety-certified
for real emergency-vehicle deployment.
