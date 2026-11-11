# routers/data_api.py
import asyncio

from fastapi import APIRouter

from data import db
from models.schemas import DataStatsResponse

router = APIRouter(prefix="/data", tags=["data"])

# Every handler below wraps its blocking sqlite3 call in asyncio.to_thread.
# These are synchronous, blocking disk I/O calls — previously invoked
# directly inside `async def` handlers, which blocks the whole event loop
# (including the /ws/live websocket and the background TrafficController
# loop) for the call's duration. services/traffic_controller.py already
# offloads its own db.* calls this way; these routers didn't, which was
# the actual inconsistency (see the audit).


@router.get("/stats", response_model=DataStatsResponse)
async def get_database_stats():
    return await asyncio.to_thread(db.get_stats)


@router.get("/emergencies")
async def get_emergency_history(limit: int = 50):
    return await asyncio.to_thread(db.get_emergency_history, limit=limit)


@router.get("/decisions")
async def get_decision_history(limit: int = 100, mode: str = None):
    return await asyncio.to_thread(db.get_decision_history, limit=limit, mode=mode)


@router.get("/benchmarks")
async def get_benchmark_history(limit: int = 10):
    return await asyncio.to_thread(db.get_benchmark_history, limit=limit)
