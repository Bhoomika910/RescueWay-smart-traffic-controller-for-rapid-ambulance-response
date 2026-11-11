# tests/test_dueling_and_per.py
"""
Unit tests for the opt-in Dueling DQN architecture (rl/dqn_network.py) and
Prioritized Experience Replay buffer (rl/replay_buffer.py) — design-review
finding (category 5/36 "RL implementation"/"Innovation": "evaluate and
implement modern DQN improvements where appropriate"). See
ARCHITECTURE.md's Dueling+PER comparison section for the actual
retrain-and-compare experimental result; these tests only verify the
mechanisms are implemented correctly, not that they improve performance
on this specific problem (a smaller network / different problem could
plausibly not benefit — the tests can't and shouldn't assume the answer).
"""
import numpy as np
import pytest
import torch

from rl.dqn_agent import DQNAgent
from rl.dqn_network import DQNetwork, DuelingDQNetwork, build_network
from rl.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from rl.state_features import N_ACTIONS, STATE_DIM


class TestBuildNetwork:
    def test_dueling_false_returns_standard_network(self):
        assert isinstance(build_network(dueling=False), DQNetwork)

    def test_dueling_true_returns_dueling_network(self):
        assert isinstance(build_network(dueling=True), DuelingDQNetwork)

    def test_both_architectures_produce_the_same_output_shape(self):
        x = torch.randn(4, STATE_DIM)
        standard_out = build_network(STATE_DIM, N_ACTIONS, 128, dueling=False)(x)
        dueling_out = build_network(STATE_DIM, N_ACTIONS, 128, dueling=True)(x)
        assert standard_out.shape == dueling_out.shape == (4, N_ACTIONS)

    def test_dueling_network_handles_unbatched_input(self):
        x = torch.randn(STATE_DIM)
        out = DuelingDQNetwork(STATE_DIM, N_ACTIONS, 128)(x)
        assert out.shape == (N_ACTIONS,)


class TestDuelingDecomposition:
    """The whole point of the architecture: verify Q(s,a) actually
    decomposes as V(s) + (A(s,a) - mean(A(s,:))), not just that a network
    with two heads happens to exist."""

    def test_advantage_stream_is_zero_mean_after_combination(self):
        net = DuelingDQNetwork(STATE_DIM, N_ACTIONS, 128)
        x = torch.randn(8, STATE_DIM)
        with torch.no_grad():
            features = net.shared(x)
            advantage = net.advantage_stream(features)
            q = net(x)
            value = net.value_stream(features)

        # Q - V must equal (A - mean(A)), which by construction has
        # per-row mean exactly zero — this is what makes V(s) and A(s,a)
        # separately identifiable rather than an unconstrained combination
        # that could shift value into either stream arbitrarily.
        centered_advantage = q - value
        assert torch.allclose(centered_advantage.mean(dim=-1), torch.zeros(8), atol=1e-5)
        assert torch.allclose(
            centered_advantage, advantage - advantage.mean(dim=-1, keepdim=True), atol=1e-5
        )


class TestDQNAgentWithDueling:
    def test_agent_trains_without_error_using_dueling_architecture(self):
        agent = DQNAgent(dueling=True, batch_size=4, buffer_size=100)
        assert isinstance(agent.online_net, DuelingDQNetwork)

        state = np.zeros(STATE_DIM, dtype=np.float32)
        for _ in range(10):
            agent.update(state, action=0, reward=1.0, next_state=state, done=False)
        assert agent.train_step_count == 10
        assert len(agent.training_losses) > 0

    def test_save_load_round_trip_preserves_dueling_architecture(self, tmp_path):
        agent = DQNAgent(dueling=True, inference_mode=False, batch_size=4, buffer_size=100)
        path = tmp_path / "dueling_test.pt"
        agent.save(path=str(path))

        reloaded = DQNAgent(dueling=False, inference_mode=True)  # deliberately wrong flag
        reloaded.load(path=str(path))

        assert reloaded.dueling is True  # corrected from the checkpoint, not the constructor arg
        assert isinstance(reloaded.online_net, DuelingDQNetwork)


class TestPrioritizedReplayBuffer:
    def test_new_transitions_get_max_priority(self):
        buf = PrioritizedReplayBuffer(capacity=10)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        buf.push(state, 0, 1.0, state, False)
        assert buf.priorities[0] == 1.0  # first-ever push, no prior max, defaults to 1.0

        buf.priorities[0] = 5.0  # simulate an update_priorities() call
        buf.push(state, 1, 2.0, state, False)
        assert buf.priorities[1] == 5.0  # new push inherits the current max

    def test_sample_returns_seven_elements_including_weights_and_indices(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        for i in range(20):
            buf.push(state, i % N_ACTIONS, float(i), state, False)

        result = buf.sample(batch_size=8)
        assert len(result) == 7
        states, actions, rewards, next_states, dones, weights, indices = result
        assert weights.shape == (8,)
        assert indices.shape == (8,)
        assert weights.max() <= 1.0 + 1e-6  # normalized by batch max, per the paper

    def test_update_priorities_changes_future_sampling_distribution(self):
        # Statistical check: give one transition a much higher priority
        # than the rest and confirm it gets sampled disproportionately
        # often — the actual behavioral point of prioritized replay, not
        # just that the API doesn't crash.
        buf = PrioritizedReplayBuffer(capacity=50, alpha=1.0)  # alpha=1: fully proportional
        state = np.zeros(STATE_DIM, dtype=np.float32)
        for i in range(50):
            buf.push(state, 0, 0.0, state, False)

        high_priority_index = 7
        buf.priorities[:] = 1e-5  # everything else near-zero priority
        buf.priorities[high_priority_index] = 100.0

        # batch_size=4, sampled without replacement (n=50 >= batch_size),
        # so each of the 200 calls contributes exactly one "slot" per
        # index at most — counting how many of the 200 CALLS included
        # this index (not raw draw count, which is diluted by the other 3
        # slots every call also fills) is the right dominance measure.
        appeared_in_call = 0
        for _ in range(200):
            _, _, _, _, _, _, indices = buf.sample(batch_size=4)
            if high_priority_index in indices:
                appeared_in_call += 1

        assert appeared_in_call > 190  # selected in nearly every call (>95%)

    def test_beta_anneals_from_beta_start_toward_one(self):
        buf = PrioritizedReplayBuffer(capacity=100, beta_start=0.4, beta_frames=10)
        assert buf._beta() == 0.4
        for _ in range(10):
            buf._sample_calls += 1
        assert buf._beta() == pytest.approx(1.0)

    def test_agent_trains_without_error_using_prioritized_replay(self):
        agent = DQNAgent(prioritized_replay=True, batch_size=4, buffer_size=100)
        assert isinstance(agent.replay_buffer, PrioritizedReplayBuffer)

        state = np.zeros(STATE_DIM, dtype=np.float32)
        for _ in range(10):
            agent.update(state, action=0, reward=1.0, next_state=state, done=False)
        assert agent.train_step_count == 10
        assert len(agent.training_losses) > 0
        # Priorities for the sampled transitions must have moved off
        # their initial value of 1.0 once update_priorities() has run.
        assert not np.allclose(agent.replay_buffer.priorities[:10], 1.0)

    def test_default_agent_still_uses_uniform_replay_buffer(self):
        # Backward-compatibility guard: the default configuration (no
        # dueling=, no prioritized_replay=) must be unaffected by this
        # feature's addition.
        agent = DQNAgent(batch_size=4, buffer_size=100)
        assert isinstance(agent.replay_buffer, ReplayBuffer)
        assert isinstance(agent.online_net, DQNetwork)


def pytest_approx(value, rel=1e-4):
    import pytest

    return pytest.approx(value, rel=rel)
