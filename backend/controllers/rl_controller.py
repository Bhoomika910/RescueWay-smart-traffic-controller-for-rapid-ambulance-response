# controllers/rl_controller.py
from controllers.base import BaseController
from rl.q_agent import QAgent


class RLController(BaseController):
    """
    The Q-Learning controller.

    Uses the trained Q-table to make decisions.
    This should outperform all baselines if training worked.
    """

    def __init__(self, agent: QAgent = None):
        super().__init__(name="Q-Learning RL")
        if agent is None:
            self.agent = QAgent(epsilon=0.0)  # pure exploitation
            self.agent.load()
        else:
            self.agent = agent

    def choose_action(
        self,
        active_signal,
        queue_lengths,
        emergency_active,
        ambulance_zone,
        emergency_eta=0.0,
        distance=0.0,
    ):
        # QAgent's discrete encoder has no distance dimension (see
        # rl/state_encoder.py) — only emergency_eta is forwarded here.
        state = self.agent.encode_rich_state(
            emergency_active=emergency_active,
            ambulance_zone=ambulance_zone if emergency_active else 0,
            active_signal=active_signal,
            queue_lengths=queue_lengths,
            emergency_eta=emergency_eta if emergency_active else 0.0,
        )
        return self.agent.choose(state)
