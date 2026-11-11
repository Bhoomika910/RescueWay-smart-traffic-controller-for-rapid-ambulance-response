# tests/test_q_agent.py
"""
Unit tests for rl/q_agent.py — design-review finding (category 27
"Testing"): no unit tests existed for either RL agent's core update rule.
"""
import numpy as np

from rl.q_agent import QAgent
from rl.state_encoder import N_ACTIONS, N_STATES


class TestTerminalStateMasking:
    """update()'s `done` parameter zeroes the bootstrap term on terminal
    transitions (ARCHITECTURE.md Part 7, finding M16) — previously always
    bootstrapped off best_next regardless of episode end, systematically
    overvaluing terminal states. This is the fix's own regression test."""

    def test_terminal_transition_ignores_next_state_value(self):
        agent = QAgent(alpha=1.0, gamma=0.95)  # alpha=1.0: update replaces the Q-value outright
        state, next_state, action = 0, 1, 0
        agent.q[next_state, :] = 1000.0  # an enormous next-state value that must NOT leak in

        agent.update(state, action, reward=5.0, next_state=next_state, done=True)

        assert agent.q[state, action] == 5.0  # reward alone, no gamma * best_next term

    def test_non_terminal_transition_bootstraps_off_next_state(self):
        agent = QAgent(alpha=1.0, gamma=0.95)
        state, next_state, action = 0, 1, 0
        agent.q[next_state, :] = 10.0

        agent.update(state, action, reward=5.0, next_state=next_state, done=False)

        assert agent.q[state, action] == 5.0 + 0.95 * 10.0

    def test_done_defaults_to_false_for_backward_compatibility(self):
        agent = QAgent(alpha=1.0, gamma=0.95)
        state, next_state, action = 0, 1, 0
        agent.q[next_state, :] = 10.0

        agent.update(state, action, reward=5.0, next_state=next_state)  # no `done` passed

        assert agent.q[state, action] == 5.0 + 0.95 * 10.0


class TestEpsilonGreedyChoice:
    def test_epsilon_zero_is_always_greedy(self):
        agent = QAgent(epsilon=0.0)
        state = 0
        agent.q[state, :] = [1, 2, 3, 9, 4, 5]  # action 3 is the clear best
        for _ in range(20):
            assert agent.choose(state) == 3

    def test_choose_increments_decision_count(self):
        agent = QAgent(epsilon=0.0)
        agent.choose(0)
        agent.choose(0)
        assert agent.decision_count == 2


class TestSaveLoadRoundTrip:
    def test_save_then_load_preserves_q_values(self, tmp_path):
        agent = QAgent()
        agent.q = np.random.RandomState(42).rand(N_STATES, N_ACTIONS)
        path = tmp_path / "q_table_test.pkl"

        agent.save(path=str(path))

        reloaded = QAgent()
        reloaded.load(path=str(path))

        assert np.allclose(reloaded.q, agent.q)

    def test_load_missing_file_starts_fresh_without_raising(self, tmp_path):
        agent = QAgent()
        agent.load(path=str(tmp_path / "does_not_exist.pkl"))
        assert agent.q.shape == (N_STATES, N_ACTIONS)
        assert np.all(agent.q == 0)
