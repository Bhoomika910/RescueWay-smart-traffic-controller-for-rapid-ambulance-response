# routers/simulation.py
from typing import Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from config import (
    DEFAULT_EMERGENCY_DURATION,
    DEFAULT_GREEN_DURATION,
    DEFAULT_YELLOW_DURATION,
)
from routers.auth import require_api_key

router = APIRouter(prefix="/simulation", tags=["simulation"])

# Single source of truth for each timing's valid range/default, shared
# between set_timings()'s clamp logic and /presets's advertised ranges —
# previously each independently hardcoded its own copy of the same
# min/max/default triple.
GREEN_DURATION_RANGE = {"min": 2, "max": 30, "default": DEFAULT_GREEN_DURATION}
YELLOW_DURATION_RANGE = {"min": 1, "max": 5, "default": DEFAULT_YELLOW_DURATION}
EMERGENCY_DURATION_RANGE = {"min": 5, "max": 60, "default": DEFAULT_EMERGENCY_DURATION}


class TrafficPatternRequest(BaseModel):
    pattern: Literal["low", "medium", "high", "rush_hour"]


class ArrivalRateRequest(BaseModel):
    zone_1: float = 1.0
    zone_2: float = 1.0
    zone_3: float = 1.0
    zone_4: float = 1.0


class TimingRequest(BaseModel):
    green_duration: Optional[int] = None
    yellow_duration: Optional[int] = None
    emergency_duration: Optional[int] = None


@router.get("/status")
async def get_simulation_status():
    """Get current simulation settings and state."""
    from main import traffic_controller

    sim = traffic_controller.simulator
    queue_lengths = {z: sim.zones[z].get_length() for z in [1, 2, 3, 4]}
    system = sim.get_system_metrics()

    return {
        "traffic_pattern": sim.traffic_pattern,
        "arrival_rates": sim.arrival_rates,
        "queue_lengths": queue_lengths,
        "system_metrics": system,
        "timings": {
            "green_duration": traffic_controller.green_duration,
            "yellow_duration": traffic_controller.yellow_duration,
            "emergency_duration": traffic_controller.emergency_duration,
        },
        "mode": traffic_controller.mode,
        "current_zone": traffic_controller.current_zone,
        "step": traffic_controller.step_count,
    }


@router.post("/pattern", dependencies=[Depends(require_api_key)])
async def set_traffic_pattern(body: TrafficPatternRequest):
    """Change traffic pattern during runtime."""
    from main import broadcast, traffic_controller

    traffic_controller.simulator.set_traffic_pattern(body.pattern)

    await broadcast(
        {
            "event": "pattern_change",
            "pattern": body.pattern,
            "message": f"Traffic pattern changed to {body.pattern}",
        }
    )

    return {
        "message": f"Traffic pattern set to {body.pattern}",
        "pattern": body.pattern,
        "arrival_rates": traffic_controller.simulator.arrival_rates,
    }


@router.post("/arrival-rates", dependencies=[Depends(require_api_key)])
async def set_arrival_rates(body: ArrivalRateRequest):
    """Set custom vehicle arrival rates per zone."""
    from main import broadcast, traffic_controller

    custom_rates = {
        "1": body.zone_1,
        "2": body.zone_2,
        "3": body.zone_3,
        "4": body.zone_4,
    }
    traffic_controller.simulator.arrival_rates = custom_rates

    await broadcast(
        {
            "event": "rate_change",
            "rates": custom_rates,
            "message": "Custom arrival rates applied",
        }
    )

    return {
        "message": "Custom arrival rates set",
        "rates": custom_rates,
    }


@router.post("/timings", dependencies=[Depends(require_api_key)])
async def set_timings(body: TimingRequest):
    """Update signal timing parameters."""
    from main import traffic_controller

    if body.green_duration is not None:
        traffic_controller.green_duration = max(
            GREEN_DURATION_RANGE["min"], min(GREEN_DURATION_RANGE["max"], body.green_duration)
        )
    if body.yellow_duration is not None:
        traffic_controller.yellow_duration = max(
            YELLOW_DURATION_RANGE["min"], min(YELLOW_DURATION_RANGE["max"], body.yellow_duration)
        )
    if body.emergency_duration is not None:
        traffic_controller.emergency_duration = max(
            EMERGENCY_DURATION_RANGE["min"],
            min(EMERGENCY_DURATION_RANGE["max"], body.emergency_duration),
        )

    return {
        "message": "Timings updated",
        "green_duration": traffic_controller.green_duration,
        "yellow_duration": traffic_controller.yellow_duration,
        "emergency_duration": traffic_controller.emergency_duration,
    }


@router.post("/reset", dependencies=[Depends(require_api_key)])
async def reset_simulation():
    """Reset the traffic simulator to initial state."""
    from main import traffic_controller

    traffic_controller.simulator.reset()
    traffic_controller.metrics_collector.reset()

    return {
        "message": "Simulation reset to initial state",
        "traffic_pattern": traffic_controller.simulator.traffic_pattern,
    }


@router.get("/presets")
async def get_presets():
    """Get available simulation presets."""
    return {
        "patterns": [
            {
                "id": "low",
                "name": "Low Traffic",
                "description": "Quiet roads, few vehicles",
                "arrival_rate": 0.5,
            },
            {
                "id": "medium",
                "name": "Medium Traffic",
                "description": "Normal daytime traffic",
                "arrival_rate": 1.0,
            },
            {
                "id": "high",
                "name": "Heavy Traffic",
                "description": "Congested roads, long queues",
                "arrival_rate": 1.5,
            },
            {
                "id": "rush_hour",
                "name": "Rush Hour",
                "description": "Peak traffic with uneven distribution",
                "arrival_rate": 2.0,
            },
        ],
        "timing_ranges": {
            "green_duration": GREEN_DURATION_RANGE,
            "yellow_duration": YELLOW_DURATION_RANGE,
            "emergency_duration": EMERGENCY_DURATION_RANGE,
        },
    }


@router.get("/gps")
async def get_gps_status():
    """Get current ambulance GPS status."""
    from main import traffic_controller

    return traffic_controller.gps_model.get_status()
