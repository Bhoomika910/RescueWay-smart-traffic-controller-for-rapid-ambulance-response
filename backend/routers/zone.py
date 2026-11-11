# routers/zone.py
import asyncio

from fastapi import APIRouter, Depends, HTTPException

from models.schemas import (
    ZoneClearResponse,
    ZoneSelectRequest,
    ZoneSelectResponse,
    ZoneStatusResponse,
)
from noise import NoiseApplicator, NoiseConfig
from routers.auth import require_api_key

router = APIRouter(prefix="/zone", tags=["zone"])

# Initialize noise from environment variables
_noise = NoiseApplicator(NoiseConfig.from_env())

ACTION_NAMES = {
    0: "Green Zone 1",
    1: "Green Zone 2",
    2: "Green Zone 3",
    3: "Green Zone 4",
    4: "Extend current green",
    5: "Normal cycle",
}


@router.post("/select", response_model=ZoneSelectResponse, dependencies=[Depends(require_api_key)])
async def select_zone(body: ZoneSelectRequest):
    from logging_system import app_logger
    from main import traffic_controller

    async def _dropped_response(reason: str) -> ZoneSelectResponse:
        app_logger.log_system_event(f"[NOISE] {reason}")
        # Broadcast the drop so frontend gets feedback. Reuses
        # TrafficController's own broadcast payload construction
        # (broadcast_state -> _broadcast_full) instead of hand-building an
        # independent ~30-key copy of it here — the two previously drifted
        # apart silently whenever a field was added to one but not the other.
        try:
            await traffic_controller.broadcast_state(
                mode=traffic_controller.mode,
                zone=traffic_controller.current_zone,
                color=traffic_controller.signal_states.get(traffic_controller.current_zone, "red"),
                reward=0.0,
                phase="noise_dropped",
                phase_description=reason,
                extra={"zone": body.zone, "noise_event": reason, "traffic": body.traffic},
            )
        except Exception:
            pass
        return ZoneSelectResponse(
            zone=body.zone,
            traffic=body.traffic,
            active_signal=traffic_controller.current_zone,
            signal_color=traffic_controller.signal_states.get(
                traffic_controller.current_zone, "red"
            ),
            duration_s=traffic_controller.emergency_green_adaptive,
            arduino_ack=traffic_controller._serial_bridge.last_ack or "pending",
            action=traffic_controller.last_action,
            action_name=ACTION_NAMES.get(traffic_controller.last_action, "None"),
            reward=0.0,
            message=reason,
        )

    # ─── PACKET LOSS CHECK ────────────────────────────
    if _noise.comm.check_packet_loss():
        return await _dropped_response("Emergency request DROPPED (packet loss)")

    # ─── COMMUNICATION LATENCY ────────────────────────
    latency = _noise.comm.get_latency()
    if latency > 0:
        app_logger.log_system_event(f"[NOISE] Communication latency: {latency:.2f}s")
        await asyncio.sleep(latency)

    # ─── GPS FALSE NEGATIVE CHECK ─────────────────────
    if _noise.gps.check_false_negative(ambulance_present=True):
        return await _dropped_response(f"Ambulance in Zone {body.zone} HIDDEN (GPS dropout)")

    # ─── GPS ZONE MISCLASSIFICATION ───────────────────
    reported_zone = _noise.gps.apply_zone(body.zone)
    if reported_zone != body.zone:
        app_logger.log_system_event(
            f"[NOISE] GPS misclassification: true Zone {body.zone} → reported Zone {reported_zone}"
        )

    # RL agent decides (no direct hardware command here). A zone that
    # somehow isn't a known ZONE_CONFIG key (defense in depth — the
    # request body itself is already constrained to 1-4 by ZoneSelectRequest)
    # raises ValueError from EmergencyManager.add_emergency rather than
    # crashing with a bare KeyError; surface it as a clean 400 instead of
    # an unhandled 500.
    try:
        action, reward, new_signal = await traffic_controller.trigger_emergency(
            zone=reported_zone, traffic=body.traffic
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    duration = traffic_controller.emergency_green_adaptive

    # Get queue lengths for the reward breakdown below (still needed
    # locally; the broadcast itself now computes its own internally).
    queue_lengths = traffic_controller._get_queue_lengths()

    # Get reward breakdown
    from rl.reward import get_reward_breakdown

    breakdown = get_reward_breakdown(
        action=action,
        amb_zone=reported_zone,
        prev_signal=traffic_controller.current_zone,
        traffic=body.traffic,
        queue_lengths=queue_lengths,
    )

    # Broadcast emergency event with full state. Reuses
    # TrafficController's own broadcast payload construction instead of
    # hand-building an independent copy — see the _dropped_response comment
    # above for why that duplication was a problem.
    await traffic_controller.broadcast_state(
        mode="emergency",
        zone=new_signal,
        color="green",
        reward=reward,
        phase="emergency_triggered",
        phase_description=f"Ambulance detected in Zone {reported_zone}",
        extra={
            "zone": reported_zone,
            "true_zone": body.zone,
            "reward_breakdown": breakdown,
            "emergency_zone": reported_zone,
        },
    )

    return ZoneSelectResponse(
        zone=reported_zone,
        traffic=body.traffic,
        active_signal=new_signal,
        signal_color="green",
        duration_s=duration,
        arduino_ack=traffic_controller._serial_bridge.last_ack or "pending",
        action=action,
        action_name=ACTION_NAMES[action],
        reward=reward,
        message=f"EMERGENCY: Zone {reported_zone} | "
        f"Distance: {traffic_controller.emergency_distance}m | "
        f"ETA: {traffic_controller.emergency_eta}s | "
        f"{ACTION_NAMES[action]}",
    )


@router.post("/clear", response_model=ZoneClearResponse, dependencies=[Depends(require_api_key)])
async def clear_emergency():
    from main import traffic_controller

    traffic_controller.clear_emergency()
    return ZoneClearResponse(
        message="Emergency cleared.",
        queue=traffic_controller.emergency_manager.get_status(),
    )


@router.get("/emergency-queue")
async def get_emergency_queue():
    from main import traffic_controller

    return traffic_controller.emergency_manager.get_status()


@router.get("/emergency-stats")
async def get_emergency_stats():
    from main import traffic_controller

    return traffic_controller.emergency_manager.get_stats()


@router.get("/status", response_model=ZoneStatusResponse)
async def get_status():
    from main import traffic_controller

    return ZoneStatusResponse(
        mode=traffic_controller.mode,
        active_signal=traffic_controller.current_zone,
        emergency_zone=traffic_controller.emergency_zone,
        step=traffic_controller.step_count,
        queue_lengths=traffic_controller._get_queue_lengths(),
        traffic_pattern=traffic_controller.simulator.traffic_pattern,
        signal_states=dict(traffic_controller.signal_states),
        status="ok",
    )
