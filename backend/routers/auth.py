# routers/auth.py
"""
Minimal shared-secret API-key authentication for state-mutating routes.

Why a shared API key instead of a full user/session/OAuth system: RescueWay
has no concept of distinct human users in its actual deployment — it is a
single-site traffic controller with one dashboard and one Arduino talking
to it. A bearer-token-style shared secret is the right-sized mechanism for
that topology (proportionate auth, not security theater); a full identity
provider would be over-engineering for a system with exactly one trust
boundary (the local network / operator vs. the public internet).

Design-review finding this addresses (Part 8 review, category 17 "Backend
engineering"): "zero authentication or authorization on any endpoint —
anyone who can reach the API can trigger or clear an emergency."

Scope: applied to every route that mutates server state or triggers
non-trivial compute (POST /zone/select, /zone/clear, /compare/run,
/eval/benchmark, /eval/stress-test, /logs/export, /simulation/pattern,
/simulation/arrival-rates, /simulation/timings, /simulation/reset).
Read-only GET routes stay open so the live dashboard can poll/observe
without embedding the key in every viewer — see ARCHITECTURE.md Part 10
for the full read/write trust-boundary rationale.
"""
from typing import Optional

from fastapi import Header, HTTPException

import config


async def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    FastAPI dependency enforcing config.API_KEY on state-mutating routes.

    No-op when config.AUTH_ENABLED is False (RESCUEWAY_API_KEY unset) —
    this is a deliberate default so local development and the existing
    test suite keep working unchanged unless an operator opts in. main.py
    logs a startup warning whenever auth is disabled so this is a visible,
    intentional choice rather than a silent gap in a real deployment.
    """
    if not config.AUTH_ENABLED:
        return
    if x_api_key is None or x_api_key != config.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-API-Key header. This route requires "
            "authentication because RESCUEWAY_API_KEY is set on the server.",
        )
