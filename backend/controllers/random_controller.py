# controllers/random_controller.py
import random

from controllers.base import BaseController


class RandomController(BaseController):
    """
    Baseline 3: Random controller.

    Picks a random action every step.
    This is the sanity check — every other controller
    should significantly outperform this.

    If your RL agent doesn't beat random, something is wrong.
    """

    def __init__(self):
        super().__init__(name="Random")

    def choose_action(
        self,
        active_signal,
        queue_lengths,
        emergency_active,
        ambulance_zone,
        emergency_eta=0.0,
        distance=0.0,
    ):
        return random.randint(0, 5)
