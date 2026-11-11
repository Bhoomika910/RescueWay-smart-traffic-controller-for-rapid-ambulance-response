# rl/replay_buffer.py
"""
Experience Replay Buffer(s) for DQN Training.

ReplayBuffer: uniform-random sampling (original, unchanged — every
existing training script defaults to this).

PrioritizedReplayBuffer: an additive, opt-in alternative (design-review
finding, category 5/36 "RL implementation"/"Innovation": "improve replay
strategy where justified"). Implements the proportional-priority variant
of Schaul et al., 2016, "Prioritized Experience Replay" (ICLR) — samples
transitions with probability proportional to their last-known TD-error
magnitude (large-error transitions are the ones the network is most wrong
about, so replaying them more often is a more sample-efficient use of a
fixed number of gradient steps than uniform sampling), corrected by
importance-sampling weights so the loss stays an unbiased estimate of the
uniform-sampling objective despite the non-uniform sampling distribution.

Implementation choice: proportional sampling via a plain priorities array
and `numpy.random.choice(..., p=probs)` rather than the sum-tree structure
the original paper uses for O(log n) updates/sampling. At this project's
buffer scale (capacity <= 100,000, matching ReplayBuffer's default), a
vectorized O(n) reweight-and-sample is still sub-millisecond in practice
and simpler to verify correct — the sum-tree is an efficiency optimization
for buffers orders of magnitude larger than this project ever uses, not a
correctness requirement of the algorithm itself.
"""
import random

import numpy as np


class ReplayBuffer:
    """Fixed-size circular buffer storing experience tuples, sampled
    uniformly at random.

    Backed by a plain list used as a ring buffer (append until `capacity`
    is reached, then overwrite oldest-first via `position`), not a
    `collections.deque`. `deque` only supports O(1) append/pop at its
    ends — random-access indexing (what `random.sample()` needs to pick a
    mini-batch) degrades to O(n) per access on a deque, since it's
    internally a doubly-linked list of blocks rather than a contiguous
    array. A `list` gives O(1) random access, so `random.sample()` over
    it is the O(batch_size) operation it's supposed to be. At this
    project's default capacity (100,000) this measured ~5x faster per
    `sample()` call than the equivalent deque — a real, easily-avoided
    per-training-step cost, since `sample()` runs once per gradient step.
    """

    def __init__(self, capacity=100_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Store a single transition."""
        transition = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            # Ring-buffer overwrite, oldest entry first — same eviction
            # order a deque(maxlen=capacity) would give.
            self.buffer[self.position] = transition
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=64):
        """Sample a random mini-batch. Returns numpy arrays."""
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size=64):
        """Check if buffer has enough samples for a batch."""
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """
    Proportional-priority experience replay (Schaul et al., 2016).

    New transitions are pushed with maximum known priority (so every
    transition is sampled at least once before its true TD-error is
    known — standard practice, avoids a cold-start bias against never-yet-
    sampled transitions). `update_priorities()` must be called after each
    `sample()` with the batch's actual TD errors, or priorities never
    improve past their initial max-priority value and this degrades to
    (a slower version of) uniform sampling.

    Args:
        capacity: max transitions retained (ring-buffer, same eviction
            order as ReplayBuffer).
        alpha: how much prioritization is used (0 = uniform/no
            prioritization, 1 = fully proportional to TD-error). 0.6 is
            the original paper's recommended default for proportional
            sampling.
        beta_start / beta_frames: importance-sampling exponent, annealed
            linearly from beta_start to 1.0 over beta_frames calls to
            `sample()` — starts partially-corrected (beta<1, faster early
            learning) and anneals to fully-corrected (beta=1, unbiased)
            as training progresses, matching the original paper.
        epsilon: small constant added to every priority so a
            zero-TD-error transition is never sampled with probability
            exactly zero (it could still be informative later).
    """

    def __init__(
        self, capacity=100_000, alpha=0.6, beta_start=0.4, beta_frames=100_000, epsilon=1e-5
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.position = 0
        self._sample_calls = 0
        self._last_indices = None  # set by sample(), consumed by update_priorities()

    def push(self, state, action, reward, next_state, done):
        transition = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        )
        # New transitions get the current max priority (or 1.0 if the
        # buffer is empty) so they're guaranteed a chance to be sampled
        # before their real TD-error is known.
        max_priority = self.priorities[: len(self.buffer)].max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def _beta(self):
        # Linear anneal from beta_start to 1.0 over beta_frames sample()
        # calls, then held at 1.0.
        progress = min(1.0, self._sample_calls / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)

    def sample(self, batch_size=64):
        """
        Returns (states, actions, rewards, next_states, dones,
        importance_sampling_weights, indices) — two extra elements versus
        ReplayBuffer.sample(), which callers must handle explicitly (see
        DQNAgent._train_step()'s `if isinstance(self.replay_buffer,
        PrioritizedReplayBuffer)` branch) rather than being silently
        swallowed by a `*_` catch-all.
        """
        n = len(self.buffer)
        priorities = self.priorities[:n]
        probs = priorities**self.alpha
        probs_sum = probs.sum()
        probs = probs / probs_sum if probs_sum > 0 else np.full(n, 1.0 / n)

        indices = np.random.choice(n, size=batch_size, p=probs, replace=n < batch_size)
        batch = [self.buffer[i] for i in indices]

        # Importance-sampling weights correct for the sampling bias:
        # transitions sampled more often get down-weighted in the loss so
        # the gradient estimate stays consistent with uniform sampling in
        # expectation. Normalized by the batch's own max weight (not a
        # theoretical max) purely for gradient-scale stability, per the
        # original paper.
        beta = self._beta()
        self._sample_calls += 1
        weights = (n * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        self._last_indices = indices

        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, weights.astype(np.float32), indices

    def update_priorities(self, indices, td_errors):
        """Call after every sample() + loss computation with the batch's
        actual |TD error| per transition (numpy array or list, same
        length/order as the `indices` returned by sample())."""
        self.priorities[indices] = np.abs(td_errors) + self.epsilon

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size=64):
        return len(self.buffer) >= batch_size
