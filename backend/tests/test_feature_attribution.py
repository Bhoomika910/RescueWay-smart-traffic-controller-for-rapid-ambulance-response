# tests/test_feature_attribution.py
"""
Unit tests for rl/feature_attribution.py — design-review finding
(category 14 "Explainability"): "implement and evaluate an actual
attribution method." These verify the attribution methods behave
correctly on constructed cases with a KNOWN answer (e.g. a feature the
network's output provably cannot depend on must get ~zero importance),
not just that they run without error.
"""
import numpy as np
import pytest
import torch

from rl.dqn_agent import DQNAgent
from rl.feature_attribution import gradient_saliency, permutation_importance
from rl.state_features import STATE_DIM


def _agent_ignoring_one_feature(ignored_idx):
    """Construct a DQNAgent whose online_net provably cannot depend on
    feature `ignored_idx` — its input-layer weight column for that
    feature is zeroed out. Used to verify attribution methods correctly
    assign it near-zero importance, a real correctness check rather than
    "the code ran."""
    agent = DQNAgent(inference_mode=True)
    with torch.no_grad():
        first_layer = agent.online_net.network[0]
        first_layer.weight[:, ignored_idx] = 0.0
    return agent


class TestGradientSaliency:
    def test_returns_all_feature_names_summing_to_one(self):
        agent = DQNAgent(inference_mode=True)
        state = np.random.RandomState(0).rand(STATE_DIM).astype(np.float32)
        result = gradient_saliency(agent, state)

        assert set(result.keys()) == set(agent.feature_extractor.get_feature_names())
        assert sum(result.values()) == pytest.approx(1.0)

    def test_is_sorted_by_descending_importance(self):
        agent = DQNAgent(inference_mode=True)
        state = np.random.RandomState(1).rand(STATE_DIM).astype(np.float32)
        result = gradient_saliency(agent, state)
        values = list(result.values())
        assert values == sorted(values, reverse=True)

    def test_a_feature_the_network_cannot_depend_on_gets_near_zero_importance(self):
        ignored_idx = 5  # arbitrary; "signal_z2" at the real deployment's N_ZONES=4
        agent = _agent_ignoring_one_feature(ignored_idx)
        state = np.random.RandomState(2).rand(STATE_DIM).astype(np.float32)

        result = gradient_saliency(agent, state)
        ignored_name = agent.feature_extractor.get_feature_names()[ignored_idx]

        assert result[ignored_name] == pytest.approx(0.0, abs=1e-6)

    def test_does_not_mutate_agent_training_state(self):
        # Backward pass must not leave stray .grad accumulation on the
        # network's own parameters that would corrupt a subsequent real
        # training step.
        agent = DQNAgent(inference_mode=False, batch_size=4, buffer_size=100)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        gradient_saliency(agent, state)

        for _ in range(4):
            agent.update(state, action=0, reward=1.0, next_state=state, done=False)
        assert agent.train_step_count == 4  # trained normally afterward, no crash/corruption


class TestPermutationImportance:
    def test_returns_all_feature_names_summing_to_one(self):
        agent = DQNAgent(inference_mode=True)
        states = np.random.RandomState(3).rand(50, STATE_DIM).astype(np.float32)
        result = permutation_importance(agent, states, n_repeats=3)

        assert set(result.keys()) == set(agent.feature_extractor.get_feature_names())
        assert sum(result.values()) == pytest.approx(1.0)

    def test_a_feature_the_network_cannot_depend_on_gets_near_zero_importance(self):
        ignored_idx = 3
        agent = _agent_ignoring_one_feature(ignored_idx)
        states = np.random.RandomState(4).rand(50, STATE_DIM).astype(np.float32)

        result = permutation_importance(agent, states, n_repeats=5)
        ignored_name = agent.feature_extractor.get_feature_names()[ignored_idx]

        assert result[ignored_name] == pytest.approx(0.0, abs=1e-6)

    def test_is_deterministic_given_the_same_seed(self):
        agent = DQNAgent(inference_mode=True)
        states = np.random.RandomState(5).rand(30, STATE_DIM).astype(np.float32)

        result_a = permutation_importance(agent, states, n_repeats=3, seed=42)
        result_b = permutation_importance(agent, states, n_repeats=3, seed=42)

        assert result_a == result_b

    def test_does_not_perturb_global_numpy_random_state(self):
        # Uses a local RandomState specifically so calling this during an
        # evaluation run doesn't desync whatever global np.random sequence
        # the caller's own seeding depends on (the same class of bug
        # documented in QAgent/DQNAgent.choose()'s epsilon-draw comments).
        agent = DQNAgent(inference_mode=True)
        states = np.random.RandomState(6).rand(20, STATE_DIM).astype(np.float32)

        np.random.seed(123)
        before = np.random.get_state()[1].copy()
        permutation_importance(agent, states, n_repeats=3)
        after = np.random.get_state()[1]

        assert np.array_equal(before, after)
