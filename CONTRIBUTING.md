# Contributing

Design-review finding (category 26 "Documentation" / category 33 "Open-source quality"):
no contributor guide. This covers local development setup, testing, and code-quality
tooling — see [`DEPLOYMENT.md`](DEPLOYMENT.md) for running this in Docker instead, and
[`backend/docs/REPRODUCE.md`](backend/docs/REPRODUCE.md) for reproducing the research results.

## Local setup (without Docker)

**Backend:**
```bash
cd backend
pip install -r requirements-dev.txt   # includes requirements.txt + lint/type/test tooling
cp .env.example .env                  # optional — every value has a working default
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
cp .env.example .env                  # optional — defaults to http://127.0.0.1:8000
npm run dev
```

**Hardware:** the Arduino firmware (`backend/arduino/traffic_controller.ino`) is optional —
`SerialBridge` falls back to simulation mode automatically if it can't open the configured
port. If you do have the hardware, see the pin map in `README.md`.

## Before opening a PR

The CI workflow (`.github/workflows/ci.yml`) runs all of this automatically; running it
locally first saves a round-trip:

```bash
# Backend
cd backend
python -m ruff check .              # lint
python -m black --check .           # format check (use `black .` without --check to fix)
python -m mypy rl/safety_shield.py rl/emergency_scoring.py routers/auth.py models/schemas.py
python -m pytest tests/ test_accuracy.py test_reward.py test_simulator.py -q --cov=.

# Frontend
cd frontend
npm run lint
npm test
npm run build
```

## Code-quality tooling and why each exists

| Tool | Scope | Why |
|---|---|---|
| `ruff` | Backend lint | Correctness/bug-risk rules (unused imports, undefined names) — see `backend/pyproject.toml`'s comment on why the rule set starts narrow rather than full-strict. |
| `black` | Backend format | One less thing to bikeshed in review; deterministic. |
| `mypy` | Backend types, strict on a few modules | Applied to `rl/safety_shield.py`, `rl/emergency_scoring.py`, `routers/auth.py`, `models/schemas.py` specifically — the shared safety/scoring logic and the API boundary, where a type error is most costly. Not project-wide yet; see `pyproject.toml`'s incremental-adoption note. |
| `oxlint` | Frontend lint | Fast, zero-config-by-default. |
| `pytest` + `pytest-cov` | Backend tests | 120+ tests as of this writing; coverage reported but not yet gated (`--cov-fail-under=0` in CI) — see "Test coverage philosophy" below. |
| `vitest` + `@testing-library/react` + `jest-dom` | Frontend tests | `jest-dom` was added specifically because `.toBeInTheDocument()`-style matchers didn't exist before — see `frontend/src/test-setup.js`. |

## Test coverage philosophy

Coverage is measured and reported in CI but not gated on a hard percentage threshold. The
project's own history is the reason why: a single, real, high-impact bug (an emergency-
accuracy counting error duplicated across nine files — see `backend/ARCHITECTURE.md` Part 9)
survived two prior manual audit passes specifically because no test exercised that logic at
all, and it was only caught by deliberately writing a test for the exact scenario the code
handles incorrectly. **When you touch RL/scoring/safety logic, add a test that would have
caught the bug you're fixing or the regression you're worried about** — that's the standard
this project holds itself to, not a coverage percentage.

## Commit conventions

No enforced format, but PRs should explain *why*, not just *what* — see the commit history
and `backend/ARCHITECTURE.md`'s root-cause/fix/verification structure for the house style.
If you're fixing a bug, say what the original (wrong) behavior was, not just what the new
code does.

## Architecture orientation

Start with `backend/main.py`'s module-map docstring and the diagram at the top of
`backend/ARCHITECTURE.md` — both describe the same layering (routers -> services -> rl/
evaluation -> traffic/sumo environments) from different angles.
