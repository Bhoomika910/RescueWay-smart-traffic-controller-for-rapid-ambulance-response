# routers/comparison.py
import asyncio

from fastapi import APIRouter, Depends

from data import db
from evaluation.comparator import VALID_PATTERNS, ControllerComparator
from models.schemas import CompareResultsResponse
from routers.auth import require_api_key

router = APIRouter(prefix="/compare", tags=["comparison"])

comparator = ControllerComparator()


@router.post("/run", response_model=CompareResultsResponse, dependencies=[Depends(require_api_key)])
async def run_comparison(
    episodes: int = 100, emergency_probability: float = 0.20, pattern: str = None
):
    from config import BENCHMARK_MAX_EPISODES

    if episodes > BENCHMARK_MAX_EPISODES:
        episodes = BENCHMARK_MAX_EPISODES

    pattern = pattern if pattern in VALID_PATTERNS else None

    # comparator.run_comparison() is a synchronous, CPU-bound call that can
    # run up to BENCHMARK_MAX_EPISODES episodes across 5 controllers —
    # potentially seconds of blocking work. Previously called directly,
    # blocking the whole event loop (including /ws/live and the background
    # TrafficController loop) for the entire duration; now offloaded to a
    # worker thread, same pattern services/traffic_controller.py already
    # uses for its own blocking calls.
    # Return value intentionally discarded: run_comparison() mutates
    # comparator's internal state, and get_summary() (below) reads from
    # that state — the episode-level results list itself isn't needed here.
    await asyncio.to_thread(
        comparator.run_comparison,
        episodes=episodes,
        emergency_probability=max(0.05, min(0.5, emergency_probability)),
        pattern=pattern,
    )
    summary = comparator.get_summary()

    # Save to database
    await asyncio.to_thread(
        db.save_benchmark,
        episodes=episodes,
        pattern=pattern or "random",
        emergency_prob=emergency_probability,
        results=summary,
    )

    return summary


@router.get("/results", response_model=CompareResultsResponse)
async def get_results():
    return comparator.get_summary()


@router.get("/history")
async def get_benchmark_history(limit: int = 10):
    return await asyncio.to_thread(db.get_benchmark_history, limit=limit)
