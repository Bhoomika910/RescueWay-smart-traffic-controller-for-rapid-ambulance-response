# controllers/rule_based.py
from controllers.base import BaseController


class RuleBasedController(BaseController):
    """
    Baseline 2: Rule-based heuristic controller.

    Rules (in priority order):
    1. If ambulance is active → give green to ambulance zone
    2. If any zone has queue > 15 → give green to that zone
    3. If current green zone still has vehicles → extend green
    4. Otherwise → give green to most congested zone

    This represents the best possible hand-coded logic
    without any learning.
    """

    def __init__(self):
        super().__init__(name="Rule-Based")
        self.min_green_steps = 3
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

        # Rule 1: Emergency priority (always)
        if emergency_active and ambulance_zone > 0:
            target = ambulance_zone - 1  # action 0-3
            if (target + 1) != active_signal:
                self.steps_on_current = 0
            return target

        # Rule 2: Critical congestion (queue > 15)
        for zone in [1, 2, 3, 4]:
            if queue_lengths.get(zone, 0) > 15:
                if zone != active_signal:
                    self.steps_on_current = 0
                return zone - 1

        # Rule 3: Keep current green if it still has vehicles
        current_queue = queue_lengths.get(active_signal, 0)
        if current_queue > 0 and self.steps_on_current < 8:
            return 4  # extend

        # Rule 4: Switch to most congested zone
        if queue_lengths:
            most_congested = max(queue_lengths, key=queue_lengths.get)
            if queue_lengths[most_congested] > 0:
                self.steps_on_current = 0
                return most_congested - 1

        # Default: normal cycle
        return 5

    def reset_episode_state(self):
        self.steps_on_current = 0
