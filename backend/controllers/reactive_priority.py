# controllers/reactive_priority.py
"""
Reactive emergency-priority baseline — an independent implementation of
the preemption mechanism described in the base paper:

  D. Veeraiah et al., "Smart Traffic Management Deep Reinforcement
  Learning for Signal Optimization using SUMO," Proc. ICIRCA 2025,
  DOI: 10.1109/ICIRCA65293.2025.11089921, Section III-E.

Their mechanism, as specified: the controller runs its normal phase
cycle; when an ambulance is detected, the ambulance's lane is granted
the NEXT signal phase, "while the present traffic phase finishes its
cycle" (Sec. III-E). Priority is therefore REACTIVE by construction —
the ambulance always absorbs whatever time remains in the current phase
(plus the yellow transition) before its approach turns green.

This module exists so the base paper's mechanism can be evaluated in
RescueWay's own harness under RescueWay's own metric definitions —
the base paper reports no measured ambulance-delay figure of its own
(its comparison tables aggregate ranges from cited literature), so a
same-harness implementation is the only way to compare against it
quantitatively.

Faithfulness notes:
  * The base paper's normal-traffic controller is a DQN; its emergency
    mechanism, however, is a rule grafted on top ("next phase priority").
    We graft the identical rule onto a fixed round-robin cycle. This is
    justified by RescueWay Table II: emergency outcomes are determined
    by the priority MECHANISM, not by the wrapped congestion policy —
    and it avoids re-training an approximation of their unpublished DQN.
  * PHASE_LEN = 25 s matches the Webster/Fixed-Timer green this
    network yields under symmetric medium demand (RescueWay §V-I), the
    same demand pattern used in the comparison.
  * Detection is in-simulation (perfect lane identification), exactly as
    the base paper assumes ("recognizes its existence using vehicle
    count and traffic density data"); the harness may corrupt the
    reported zone with the shared GPSNoise model to test both methods
    under identical sensing error.
"""


class ReactivePriorityController:
    """Round-robin phases; ambulance zone becomes the NEXT phase after
    the current phase completes — the base paper's Section III-E rule."""

    def __init__(self, phase_len=25, n_zones=4):
        self.phase_len = phase_len
        self.n_zones = n_zones
        self.current_zone = 1
        self.elapsed = 0
        self.pending_emergency_zone = 0

    def notify_emergency(self, reported_zone):
        """Called each tick an ambulance is detected. Records the request
        but — per the base paper — never interrupts the running phase."""
        if reported_zone and reported_zone > 0:
            self.pending_emergency_zone = reported_zone

    def clear_emergency(self):
        self.pending_emergency_zone = 0

    def decide(self, dt=1):
        """
        Advance internal clock by dt seconds; return the zone that should
        be green NOW. A zone change is signalled by the return value
        differing from the previous one; the caller applies yellow.
        """
        self.elapsed += dt
        if self.elapsed < self.phase_len:
            return self.current_zone

        # Current phase has finished its cycle — choose the next phase.
        self.elapsed = 0
        if self.pending_emergency_zone:
            self.current_zone = self.pending_emergency_zone
            # One granted phase per request; re-detection re-arms it.
            self.pending_emergency_zone = 0
        else:
            self.current_zone = (self.current_zone % self.n_zones) + 1
        return self.current_zone
