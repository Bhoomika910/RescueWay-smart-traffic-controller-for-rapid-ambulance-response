# controllers/dqn_controller.py
"""
DQN Controller for benchmarking.

Wraps the trained DQN agent into the BaseController interface
so it can be compared alongside Fixed, Rule-Based, Random, and Q-Learning.
"""
from controllers.base import BaseController
from rl.dqn_agent import DQNAgent
from rl.safety_shield import shield_emergency_action


class DQNController(BaseController):
    """
    Double DQN controller using trained neural network, RAW policy — no
    safety shield applied. This is the right controller for measuring the
    learned policy's own competence in isolation (e.g. "did the network
    learn to prioritize the ambulance zone"), which is what most of this
    benchmark suite exists to measure.

    Uses the trained model from rl/dqn_model.pt.

    See DQNShieldedController below for the variant that matches what
    the LIVE system actually does in production.
    """

    def __init__(self, agent: DQNAgent = None):
        super().__init__(name="Double DQN")
        if agent is None:
            self.agent = DQNAgent(epsilon=0.0, inference_mode=True)
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
        state = self.agent.encode_rich_state(
            emergency_active=emergency_active,
            ambulance_zone=ambulance_zone if emergency_active else 0,
            active_signal=active_signal,
            queue_lengths=queue_lengths,
            emergency_eta=emergency_eta if emergency_active else 0.0,
            distance=distance if emergency_active else 0.0,
        )
        return self.agent.choose(state)


class DQNShieldedController(DQNController):
    """
    Double DQN + the emergency safety shield (rl/safety_shield.py) — the
    SAME hard override services/traffic_controller.py already applies in
    production, wrapped around the raw DQN policy for benchmarking.

    Why this exists (design-review finding): the review recommended
    adding "a shielded/constrained action-selection layer ... combining
    DQN's general traffic-management competence with the safety guarantee
    Rule-Based already demonstrates," after noting the project's own data
    shows Rule-Based's hard-priority approach beating DQN's learned
    approach on emergency accuracy. That layer already existed in the
    live system, but the benchmark suite only ever measured DQNController
    (the raw, un-shielded policy) — so a reader comparing DQN's ~85%
    benchmark accuracy against Rule-Based's 100% could easily conclude
    production has the same gap, which is false: production has
    guaranteed 100% emergency accuracy via this override since the shield
    was added. This controller lets the benchmark report that explicitly,
    honestly labeled, alongside (not instead of) the raw DQNController —
    both numbers matter for different questions ("how good is the learned
    policy alone" vs. "what does deployment actually guarantee").
    """

    def __init__(self, agent: DQNAgent = None):
        super().__init__(agent=agent)
        self.name = "Double DQN + Safety Shield"

    def choose_action(
        self,
        active_signal,
        queue_lengths,
        emergency_active,
        ambulance_zone,
        emergency_eta=0.0,
        distance=0.0,
    ):
        raw_action = super().choose_action(
            active_signal,
            queue_lengths,
            emergency_active,
            ambulance_zone,
            emergency_eta,
            distance,
        )
        enacted_action, _overridden = shield_emergency_action(
            raw_action,
            ambulance_zone if emergency_active else 0,
            active_signal,
        )
        return enacted_action
