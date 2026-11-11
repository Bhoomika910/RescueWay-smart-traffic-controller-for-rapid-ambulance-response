# rl/dqn_network.py
"""
Deep Q-Network architectures for RescueWay.

DQNetwork: the original standard feedforward network (state_dim -> 128 ->
128 -> action_dim), unchanged — every existing checkpoint (dqn_model.pt,
sumo/checkpoints/*.pt) was trained against this class and continues to
load and run identically. Nothing about the default agent configuration
changed by adding the option below.

DuelingDQNetwork: an additive, opt-in alternative (design-review finding,
category 5/6/36 "RL implementation"/"DQN implementation"/"Innovation":
"evaluate and implement modern DQN improvements where appropriate").
Implements Wang et al., 2016, "Dueling Network Architectures for Deep
Reinforcement Learning" (ICML) — splits the shared feature representation
into a scalar state-value stream V(s) and a per-action advantage stream
A(s,a), recombined as:

    Q(s,a) = V(s) + (A(s,a) - mean_a' A(s,a'))

Why this is a reasonable fit for this specific problem, not novelty for
its own sake: RescueWay's state includes long stretches (no active
emergency) where the VALUE of the state matters far more than which of
the 6 actions is taken — a well-managed intersection with low queues is
good regardless of which particular zone gets the next green, and the
dueling split lets the network learn "how good is this traffic situation"
(V) separately from "which action within it matters" (A), which the
original paper shows converges faster than a monolithic Q-head
specifically in domains with many state regions where action choice has
low marginal impact. See ARCHITECTURE.md's Dueling+PER comparison for
whether this predicted benefit actually materialized on this project's
own training budget — the comparison is reported honestly either way.
"""
import torch.nn as nn


def _xavier_init(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)


class DQNetwork(nn.Module):
    """
    Two-layer feedforward Q-network (original architecture, unchanged).

    Input:  Continuous state vector (19 features)
    Output: Q-value for each of 6 actions
    """

    def __init__(self, state_dim=19, action_dim=6, hidden_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        _xavier_init(self)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Q-values of shape (batch_size, action_dim) or (action_dim,)
        """
        return self.network(x)


class DuelingDQNetwork(nn.Module):
    """
    Dueling architecture: shared trunk, then split value/advantage heads
    recombined into Q-values. Same input/output shape contract as
    DQNetwork, so it's a drop-in replacement wherever DQNetwork is
    constructed — DQNAgent selects between them via its `dueling=`
    constructor flag rather than callers needing to know which class is in
    use.

    Parameter count is kept comparable to DQNetwork at the same hidden_dim
    (shared trunk is one layer instead of two, with the second layer's
    capacity redistributed into the two heads) so any observed difference
    in training outcome is attributable to the architectural split, not
    simply to having more parameters.
    """

    def __init__(self, state_dim=19, action_dim=6, hidden_dim=128):
        super().__init__()
        self.action_dim = action_dim

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        _xavier_init(self)

    def forward(self, x):
        features = self.shared(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Subtracting the mean advantage (rather than the max, the paper's
        # other proposed variant) is the more common and more stable
        # choice in practice — it doesn't force any single action's
        # advantage to exactly zero, which the max-based version does and
        # which can make that action's individual advantage estimate
        # harder to identify during training.
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


def build_network(state_dim=19, action_dim=6, hidden_dim=128, dueling=False):
    """Single construction point so DQNAgent doesn't need an if/else at
    every one of its (two) call sites — see DQNAgent.__init__/load()."""
    cls = DuelingDQNetwork if dueling else DQNetwork
    return cls(state_dim, action_dim, hidden_dim)
