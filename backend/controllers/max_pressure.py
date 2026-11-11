# controllers/max_pressure.py
from controllers.base import BaseController


class MaxPressureController(BaseController):
    """
    Max-Pressure traffic-signal control (Varaiya, "Max pressure control of
    a network of signalized intersections", Transportation Research Part C,
    2013 — the standard backpressure-routing heuristic widely used as a
    non-learned baseline in the traffic-signal-RL literature, e.g. PressLight
    (Wei et al. 2019) is explicitly named after it).

    Added as a genuine non-oracle heuristic baseline. Rule-Based (see
    rule_based.py) hardcodes "always serve the ambulance zone first" —
    i.e. it was authored with direct knowledge of this benchmark's reward
    function and reaps near-maximal emergency-related reward *by
    construction*, not by learning or by an independently-motivated
    traffic-engineering objective. That makes it useful as an upper-bound
    "oracle" reference (see comparator.py's oracle_controllers), but not a
    fair peer to compare a learned policy against. Max-Pressure has no
    privileged knowledge of the reward function and — deliberately — no
    ambulance-awareness at all: it optimizes queue pressure only, the way
    a real traffic engineer's heuristic would, giving DQN/Q-Learning a
    baseline that must be beaten by actually learning to balance
    congestion management against emergency response, not by out-cheating
    a rule that already knows the answer.

    Simplified single-intersection form: this simulator has no downstream/
    outgoing-queue model, so "pressure" reduces to each zone's own queue
    length (the full max-pressure formulation is incoming-queue minus
    outgoing-queue per movement; with no outgoing-queue signal available
    here, incoming queue length alone is the pressure term. This is the
    standard simplification used when applying max-pressure to a
    single-junction, queue-length-only observation space.)
    """

    def __init__(self):
        super().__init__(name="Max-Pressure")
        self.steps_on_current = 0

    def choose_action(
        self,
        active_signal,
        queue_lengths,
        emergency_active,
        ambulance_zone,
        emergency_eta=0.0,
        distance=0.0,
    ):
        self.steps_on_current += 1

        pressure = {z: queue_lengths.get(z, 0) for z in [1, 2, 3, 4]}
        best_zone = max(pressure, key=pressure.get)

        if pressure[best_zone] == 0:
            return 5  # nothing queued anywhere — normal cycle

        if best_zone == active_signal:
            return 4  # already serving the highest-pressure zone — extend

        self.steps_on_current = 0
        return best_zone - 1

    def reset_episode_state(self):
        self.steps_on_current = 0
