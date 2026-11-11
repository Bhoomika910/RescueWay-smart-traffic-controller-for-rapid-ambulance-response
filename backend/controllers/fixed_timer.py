# controllers/fixed_timer.py
from controllers.base import BaseController


class FixedTimerController(BaseController):
    """
    Baseline 1: Fixed cycle controller.

    Cycles through zones 1 → 2 → 3 → 4 → 1 → ...
    Ignores traffic conditions completely.
    Ignores ambulance emergencies.

    This represents a traditional traffic light with no intelligence.
    """

    def __init__(self):
        super().__init__(name="Fixed Timer")
        self.cycle = [1, 2, 3, 4]
        self.cycle_index = 0
        self.steps_on_current = 0
        self.steps_per_green = 5  # fixed 5 steps per green

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

        if self.steps_on_current >= self.steps_per_green:
            # Move to next zone in cycle
            self.steps_on_current = 0
            self.cycle_index = (self.cycle_index + 1) % 4
            return self.cycle[self.cycle_index] - 1  # action 0-3

        # Stay on current zone
        return 4  # extend current green

    def reset_episode_state(self):
        self.cycle_index = 0
        self.steps_on_current = 0
