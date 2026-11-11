# routers/evaluation.py
import asyncio

from fastapi import APIRouter, Depends

from models.schemas import EvalSummaryResponse
from routers.auth import require_api_key

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/live")
async def get_live_metrics():
    """Get real-time metrics from the running traffic controller."""
    from main import traffic_controller

    return traffic_controller.metrics_collector.get_current_metrics(
        traffic_controller.emergency_manager.get_stats()
    )


@router.get("/trends")
async def get_trends(window: int = 50):
    """Get recent metric trends for charts."""
    from main import traffic_controller

    return traffic_controller.metrics_collector.get_trends(window=window)


@router.post("/benchmark", dependencies=[Depends(require_api_key)])
async def run_benchmark(episodes: int = 100, traffic_pattern: str = "medium"):
    """DEPRECATED: Use /compare/run instead. This endpoint redirects to the
    unified benchmark pipeline that includes DQN."""
    from data import db
    from evaluation.comparator import ControllerComparator

    comparator = ControllerComparator()
    # Both calls are synchronous, blocking/CPU-bound (up to 500 episodes
    # across 5 controllers, then a disk write) — offloaded to a worker
    # thread so they don't block the event loop, same as /compare/run.
    await asyncio.to_thread(comparator.run_comparison, episodes=min(episodes, 500))
    summary = comparator.get_summary()

    await asyncio.to_thread(
        db.save_benchmark,
        episodes=episodes,
        pattern=traffic_pattern,
        emergency_prob=0.20,
        results=summary,
    )

    return summary


@router.post("/stress-test", dependencies=[Depends(require_api_key)])
async def stress_test(episodes: int = 50):
    """Test RL controller across all traffic patterns."""
    from controllers import RLController
    from metrics.calculator import PerformanceCalculator

    controller = RLController()
    # compare_across_traffic runs `episodes` runs for each of 4 patterns —
    # blocking/CPU-bound, offloaded for the same reason as above.
    results = await asyncio.to_thread(
        PerformanceCalculator.compare_across_traffic,
        controller,
        patterns=["low", "medium", "high", "rush_hour"],
        episodes=episodes,
    )

    return {
        "controller": "Q-Learning RL",
        "results_by_pattern": results,
    }


@router.get("/summary", response_model=EvalSummaryResponse)
async def get_evaluation_summary():
    """Get a quick summary of system performance."""
    from main import traffic_controller

    live = traffic_controller.metrics_collector.get_current_metrics(
        traffic_controller.emergency_manager.get_stats()
    )

    return {
        "mode": traffic_controller.mode,
        "traffic_pattern": traffic_controller.simulator.traffic_pattern,
        "total_steps": live["step"],
        "uptime_seconds": live["uptime_seconds"],
        "performance": {
            "avg_reward": live["avg_reward"],
            "avg_queue_length": live["avg_queue_length"],
            "avg_wait_time": live["avg_wait_time"],
            "clearance_rate": live["clearance_rate"],
            "emergency_accuracy": live["emergency_accuracy"],
            "fairness_index": live["fairness_index"],
        },
        "totals": {
            "vehicles_arrived": live["total_vehicles_arrived"],
            "vehicles_cleared": live["total_vehicles_cleared"],
            "signal_switches": live["total_switches"],
            "emergency_events": live["emergency_events"],
        },
    }
