# controllers/base.py
from abc import ABC, abstractmethod


class BaseController(ABC):
    """
    Base class for all traffic controllers.
    Every controller must implement choose_action().
    This allows fair comparison between different strategies.
    """

    def __init__(self, name: str):
        self.name = name
        self.decision_count = 0
        self.total_reward = 0

    @abstractmethod
    def choose_action(
        self,
        active_signal: int,
        queue_lengths: dict,
        emergency_active: bool,
        ambulance_zone: int,
        emergency_eta: float = 0.0,
        distance: float = 0.0,
    ) -> int:
        """
        Choose which zone should get the green signal.

        Args:
            active_signal:    current green zone (1-4)
            queue_lengths:    {1: 5, 2: 12, 3: 3, 4: 8}
            emergency_active: True if ambulance is present
            ambulance_zone:   1-4 ambulance location (0 if none)
            emergency_eta:    seconds until the ambulance arrives (0 if
                              none/unknown). Optional so non-learned
                              controllers (Fixed/Rule-Based/Random) don't
                              need to accept or use it, but every caller
                              (live controller, benchmark) should supply
                              it so learned controllers (DQN/Q-Learning)
                              receive the same state they were trained on —
                              see rl/ambulance_delay.py and
                              backend/ARCHITECTURE.md for why this matters.
            distance:         ambulance's distance to the intersection in
                              meters (0 if none/unknown)

        Returns:
            action: 0-3 = green zone 1-4, 4 = extend, 5 = normal cycle
        """
        pass

    def record_reward(self, reward: float):
        self.decision_count += 1
        self.total_reward += reward

    def get_avg_reward(self) -> float:
        if self.decision_count == 0:
            return 0
        return self.total_reward / self.decision_count

    def reset_stats(self):
        self.decision_count = 0
        self.total_reward = 0

    def reset_episode_state(self):
        """
        Reset any internal timing/cycling state a controller carries
        between decisions (e.g. a fixed-cycle phase or a "steps on
        current green" counter). Called at the start of every benchmark
        episode so a fresh, independent simulated episode doesn't inherit
        leftover state from whatever the controller was doing at the end
        of the previous episode (or the previous benchmark run, since
        comparator controllers are long-lived singletons). No-op by
        default; override in subclasses that hold such state.
        """
        pass

    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "decisions": self.decision_count,
            "total_reward": round(self.total_reward, 2),
            "avg_reward": round(self.get_avg_reward(), 2),
        }
