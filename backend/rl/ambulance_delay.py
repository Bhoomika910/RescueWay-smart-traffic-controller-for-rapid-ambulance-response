# rl/ambulance_delay.py
"""
Shared "will this ambulance be delayed" primitive.

The research objective is not "prioritize the ambulance once it's at the
intersection" — it's "clear the intersection before the ambulance arrives,
so it never has to stop." That requires comparing how much time the
ambulance has left (its ETA) against how much time is actually needed to
be ready for it: the fixed signal-transition time (yellow + all-red) plus
however long it takes to clear whatever is currently queued in its zone.

This module is the ONE place that comparison is computed, so reward.py,
the state encoders, the live TrafficController, and the benchmark all
agree on what "ambulance delay" means. Two call sites feed it:

  - The simple simulator / benchmark (traffic/simulator.py-based) has no
    per-vehicle motion model, so `remaining_eta` is a countdown-clock
    PROXY (ticking down from the zone's configured ETA) and the resulting
    `deficit` is an ESTIMATE, not a measurement.
  - SUMO (sumo/sumo_env.py) tracks the real ambulance vehicle via TraCI
    and already has a real per-vehicle waiting time
    (sumo_adapter.get_ambulance_waiting_time) — SUMO callers should pass
    that measured value directly as `ambulance_delay` into compute_reward
    rather than going through estimate_clearance() at all. This module
    only supplies the proxy for contexts that have no such measurement.
"""
from config import DEFAULT_ALL_RED_DURATION, DEFAULT_YELLOW_DURATION, VEHICLE_CLEARANCE_RATE

# Default signal-transition time (yellow + all-red) a zone needs before it
# can actually be green, expressed in the same units as ETA (seconds).
TRANSITION_TIME_DEFAULT = float(DEFAULT_YELLOW_DURATION + DEFAULT_ALL_RED_DURATION)


def estimate_clearance(
    remaining_eta: float,
    queue_length: int,
    zone_is_served: bool,
    clearance_rate: float = VEHICLE_CLEARANCE_RATE,
    transition_time: float = TRANSITION_TIME_DEFAULT,
) -> dict:
    """
    Estimate whether an ambulance will have to stop, and by how much.

    Args:
        remaining_eta:   seconds until the ambulance reaches the intersection
        queue_length:    vehicles currently queued in the ambulance's zone
        zone_is_served:  True if that zone already has the green signal right
                         now (so no transition is needed, only clearing)
        clearance_rate:  vehicles clearable per second once green (shared
                          constant with traffic/simulator.py)
        transition_time: seconds needed for yellow+all-red before the zone
                          can turn green, if it doesn't already have it

    Returns:
        lead_time_needed:   seconds required to have the zone green AND
                             clear before the ambulance arrives
        deficit:            predicted/actual delay in seconds (0 if the zone
                             will be ready in time; positive = will stop)
        will_stop:          deficit > 0
        lead_time_available: remaining_eta - lead_time_needed. Positive =
                             margin to spare (this doubles as the
                             "Time-to-Clear-Before-Arrival" metric);
                             negative = how far behind schedule clearing is.
    """
    transition_needed = 0.0 if zone_is_served else transition_time
    clearance_needed = queue_length / max(clearance_rate, 0.01)
    lead_time_needed = transition_needed + clearance_needed
    deficit = max(0.0, lead_time_needed - remaining_eta)

    return {
        "lead_time_needed": round(lead_time_needed, 2),
        "deficit": round(deficit, 2),
        "will_stop": deficit > 0.0,
        "lead_time_available": round(remaining_eta - lead_time_needed, 2),
    }
