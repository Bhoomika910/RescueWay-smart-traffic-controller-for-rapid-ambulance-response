# rl/feature_attribution.py
"""
Real feature-attribution explainability for the DQN agent.

Design-review finding (category 14 "Explainability"): the existing
`xai_info` payload (services/traffic_controller.py) is genuine decision-
TRANSPARENCY instrumentation — real Q-values, a structured reward
breakdown, whether the safety shield fired — but it is not attribution in
the ML-XAI research sense: nothing in the project previously computed
which INPUT FEATURES actually drove a given decision. The review is
explicit that self-reporting the model's own outputs is not the same
claim as explaining them, and recommended either dropping the
"explainability" framing entirely or implementing an actual attribution
method.

Two methods, both standard and both implemented (not just one, so the
choice of method is a documented tradeoff rather than a single arbitrary
pick):

  gradient_saliency(): |dQ(a*)/dx_i|, a single backward pass — the
      Simonyan et al., 2013 ("Deep Inside Convolutional Networks:
      Visualising Image Classification Models and Saliency Maps") approach
      adapted from pixel inputs to this project's tabular state vector.
      Cheap enough (one backward pass, ~1ms) to compute on every live
      decision without meaningfully affecting the control loop's latency
      — this is the one wired into services/traffic_controller.py's
      xai_info.

  permutation_importance(): Breiman, 2001 ("Random Forests," which
      popularized this as a general-purpose, model-agnostic technique) —
      shuffle one feature across a batch of representative states at a
      time and measure how much the chosen action's Q-value moves on
      average. Model-agnostic (doesn't need gradients, so it would work
      for the tabular QAgent too, unlike saliency) and more robust to
      saturated/flat gradient regions, at the cost of being much more
      expensive (many forward passes per feature) — used for offline
      analysis (tests, ARCHITECTURE.md's explainability validation), not
      the live per-decision path.

Both report attribution over rl.state_features.StateFeatureExtractor's
named features (get_feature_names()), so results are directly
interpretable rather than raw indices.
"""
import numpy as np
import torch


def gradient_saliency(agent, state):
    """
    Single-backward-pass saliency: |dQ(chosen_action)/dx_i| for each
    input feature x_i, normalized to sum to 1 (a relative-importance
    distribution, not raw gradient magnitudes, which aren't comparable
    across differently-scaled features).

    Args:
        agent: a rl.dqn_agent.DQNAgent (needs .online_net, .device).
        state: numpy array of shape (state_dim,) — typically from
            agent.encode_rich_state(...).

    Returns:
        dict[str, float] — feature_name -> normalized importance,
        ordered by descending importance.
    """
    state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    state_t.requires_grad_(True)

    q_values = agent.online_net(state_t)
    chosen_action = int(q_values.argmax(dim=1).item())

    agent.online_net.zero_grad(set_to_none=True)
    q_values[0, chosen_action].backward()

    gradients = state_t.grad[0].abs().detach().cpu().numpy()
    total = gradients.sum()
    normalized = gradients / total if total > 0 else gradients

    names = agent.feature_extractor.get_feature_names()
    importance = dict(zip(names, (float(v) for v in normalized)))
    return dict(sorted(importance.items(), key=lambda kv: -kv[1]))


def permutation_importance(agent, states, n_repeats=10, seed=42):
    """
    Model-agnostic permutation importance over a batch of representative
    states.

    For each feature i: repeatedly shuffle that feature's values across
    the batch (breaking its association with the outcome while preserving
    its marginal distribution), re-run the network, and measure the mean
    absolute change in the CHOSEN action's Q-value versus the unperturbed
    batch. Averaged over n_repeats shuffles per feature to reduce the
    variance of any single random shuffle.

    Args:
        agent: a rl.dqn_agent.DQNAgent.
        states: numpy array of shape (n_states, state_dim) — a batch of
            representative states (e.g. sampled from a replay buffer or
            an evaluation run), not a single state — permutation
            importance is inherently a batch-level, not per-decision,
            technique.
        n_repeats: number of independent shuffles per feature to average
            over.
        seed: local RandomState seed — does not touch global numpy/random
            state, so this can be called during training/evaluation
            without desyncing whatever RNG sequence the caller depends on.

    Returns:
        dict[str, float] — feature_name -> mean |Q change| when that
        feature is shuffled, normalized to sum to 1, ordered by
        descending importance.
    """
    rng = np.random.RandomState(seed)
    states = np.asarray(states, dtype=np.float32)
    n_states, state_dim = states.shape

    with torch.no_grad():
        baseline_q = agent.online_net(torch.FloatTensor(states).to(agent.device))
        baseline_chosen = baseline_q.gather(1, baseline_q.argmax(dim=1, keepdim=True)).squeeze(1)

    raw_importance = np.zeros(state_dim)
    for feature_idx in range(state_dim):
        deltas = []
        for _ in range(n_repeats):
            perturbed = states.copy()
            perturbed[:, feature_idx] = rng.permutation(perturbed[:, feature_idx])
            with torch.no_grad():
                perturbed_q = agent.online_net(torch.FloatTensor(perturbed).to(agent.device))
                # Compare against the SAME action index chosen on the
                # unperturbed input — the question is "how much does this
                # feature affect the value of the action we actually
                # took," not "how much does the argmax action's identity
                # change" (a related but different question).
                original_actions = baseline_q.argmax(dim=1, keepdim=True)
                perturbed_chosen = perturbed_q.gather(1, original_actions).squeeze(1)
            delta = (perturbed_chosen - baseline_chosen).abs().mean().item()
            deltas.append(delta)
        raw_importance[feature_idx] = np.mean(deltas)

    total = raw_importance.sum()
    normalized = raw_importance / total if total > 0 else raw_importance

    names = agent.feature_extractor.get_feature_names()
    importance = dict(zip(names, (float(v) for v in normalized)))
    return dict(sorted(importance.items(), key=lambda kv: -kv[1]))
