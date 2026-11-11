# rl/dqn_agent.py
"""
Double DQN Agent for RescueWay.

Drop-in replacement for QAgent. Matches the same interface so
TrafficController doesn't need changes:
  - agent.choose(state) → int
  - agent.update(state, action, reward, next_state) → None
  - agent.get_best_action_name(action) → str
  - agent.encode_rich_state(...) → state representation
  - agent.load() / agent.save()

Key differences from QAgent:
  - State is a 19-dim continuous vector (not a single integer)
  - Q-values computed by neural network (not table lookup)
  - Double DQN: uses online network to SELECT action,
    target network to EVALUATE value (reduces overestimation)
  - Experience replay for stable training
  - Target network soft-updated every N steps
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.dqn_network import build_network
from rl.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from rl.state_features import ACTION_NAMES, N_ACTIONS, STATE_DIM, StateFeatureExtractor

# Default model path
MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_model.pt")


class DQNAgent:
    """
    Double DQN agent for traffic signal control.

    Architecture:
      - Online network: selects actions and is updated every step
      - Target network: provides stable Q-value targets, updated every
        target_update_freq steps via hard copy

    Training:
      - Experience replay buffer (100k transitions)
      - Mini-batch gradient descent (batch_size=64)
      - Epsilon-greedy exploration with linear decay

    Inference:
      - Epsilon fixed at inference_epsilon (default 0.0) whenever
        inference_mode=True, regardless of what `epsilon=` was passed —
        see inference_epsilon's docstring below for why this is enforced
        rather than left to the caller.
      - Forward pass through online network (~1ms on CPU)
    """

    def __init__(
        self,
        alpha=5e-4,
        gamma=0.95,
        epsilon=1.0,
        batch_size=64,
        buffer_size=100_000,
        target_update_freq=200,
        hidden_dim=128,
        inference_mode=False,
        inference_epsilon=0.0,
        dueling=False,
        prioritized_replay=False,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=100_000,
    ):
        """
        Args:
            alpha: Learning rate for Adam optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate (training only — ignored
                when inference_mode=True, see inference_epsilon)
            batch_size: Mini-batch size for replay sampling
            buffer_size: Maximum replay buffer capacity
            target_update_freq: Steps between target network hard updates.
                Default lowered from 1000 to 200 (technical audit finding —
                see backend/ARCHITECTURE.md): 1000 is a reasonable default
                for large-scale (100k+ step) training, but against this
                project's realistic training budgets (a few thousand to
                tens of thousands of steps) it left the target network
                updating only a handful of times total, undermining Double
                DQN's stabilization mechanism.
            hidden_dim: Hidden layer width
            inference_mode: If True, skip replay buffer/optimizer setup and
                force epsilon to inference_epsilon (see below).
            inference_epsilon: Epsilon actually used whenever
                inference_mode=True. Defaults to 0.0 (fully greedy),
                matching every controller/evaluation call site in this
                codebase except one: main.py (the live production
                TrafficController) used to construct this agent with
                epsilon=config.RL_EPSILON (0.05) instead of 0.0, so the
                deployed policy took a random action ~5% of the time in
                normal-mode cycling while every benchmark/evaluation
                script scored it at epsilon=0.0 — the benchmarked numbers
                described a policy the live system didn't actually run.
                Making inference_mode authoritative over epsilon (rather
                than trusting every call site to remember to pass 0.0)
                fixes that mismatch at the root and prevents it from
                recurring in a future call site.
            dueling: use DuelingDQNetwork (Wang et al., 2016) instead of
                the original DQNetwork. Default False — the production
                checkpoint and every existing evaluation script keep using
                the original architecture unless explicitly opted in; see
                rl/dqn_network.py's module docstring and ARCHITECTURE.md's
                Dueling+PER comparison for why this is offered as an
                option rather than a wholesale replacement.
            prioritized_replay: use PrioritizedReplayBuffer (Schaul et
                al., 2016) instead of uniform ReplayBuffer. Default False,
                same rationale as `dueling`.
            per_alpha / per_beta_start / per_beta_frames: forwarded to
                PrioritizedReplayBuffer when prioritized_replay=True;
                ignored otherwise.
        """
        self.gamma = gamma
        self.epsilon = inference_epsilon if inference_mode else epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.inference_mode = inference_mode
        self.dueling = dueling
        self.prioritized_replay = prioritized_replay
        self.hidden_dim = hidden_dim

        # Device (CPU for this project — Intel UHD not supported by PyTorch)
        self.device = torch.device("cpu")

        # Networks
        self.online_net = build_network(STATE_DIM, N_ACTIONS, hidden_dim, dueling=dueling).to(
            self.device
        )
        self.target_net = build_network(STATE_DIM, N_ACTIONS, hidden_dim, dueling=dueling).to(
            self.device
        )
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target network is never in training mode

        # Feature extractor (replaces StateEncoder)
        self.feature_extractor = StateFeatureExtractor()

        # Training components (skip if inference only)
        if not inference_mode:
            self.optimizer = optim.Adam(self.online_net.parameters(), lr=alpha)
            # reduction="none": PER needs the per-sample loss (to derive
            # per-sample TD errors for priority updates and to apply
            # per-sample importance-sampling weights) before it gets
            # reduced to a scalar; uniform-buffer training reduces it
            # itself via .mean() in _train_step(). Behaviorally identical
            # to the previous reduction="mean" Huber loss for the
            # non-PER path — see _train_step()'s branch.
            self.loss_fn = nn.SmoothL1Loss(reduction="none")
            if prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(
                    capacity=buffer_size,
                    alpha=per_alpha,
                    beta_start=per_beta_start,
                    beta_frames=per_beta_frames,
                )
            else:
                self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        else:
            self.optimizer = None
            self.loss_fn = None
            self.replay_buffer = None

        # Counters
        self.decision_count = 0
        self.correct_count = 0
        self.train_step_count = 0
        self.training_losses = []

    # ══════════════════════════════════════════════════════
    # INTERFACE MATCHING QAgent (used by TrafficController)
    # ══════════════════════════════════════════════════════

    def encode_rich_state(
        self,
        emergency_active,
        ambulance_zone,
        active_signal,
        queue_lengths,
        emergency_eta=0.0,
        distance=0.0,
    ):
        """
        Encode traffic state into continuous feature vector.

        Returns numpy array (not integer like QAgent).
        TrafficController calls this, then passes result to choose().
        """
        return self.feature_extractor.extract(
            emergency_active=emergency_active,
            ambulance_zone=ambulance_zone,
            active_signal=active_signal,
            queue_lengths=queue_lengths,
            emergency_eta=emergency_eta,
            distance=distance,
        )

    def encode_state(self, zone, traffic, signal):
        """Backward-compatible simple encoding."""
        base = {"low": 3, "medium": 8, "high": 15}
        level = base.get(traffic, 8)
        queue_lengths = {
            1: level + np.random.randint(-2, 3),
            2: level + np.random.randint(-2, 3),
            3: level + np.random.randint(-2, 3),
            4: level + np.random.randint(-2, 3),
        }
        return self.encode_rich_state(
            emergency_active=(zone > 0),
            ambulance_zone=zone,
            active_signal=signal,
            queue_lengths=queue_lengths,
        )

    def choose(self, state):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: numpy array of shape (19,) from encode_rich_state()

        Returns:
            int: action index 0-5
        """
        self.decision_count += 1

        # Epsilon-greedy exploration. Skip the RNG draw entirely when
        # epsilon <= 0 (pure exploitation, e.g. benchmark/inference mode) —
        # see QAgent.choose() for why an unconditional draw here would
        # desync the shared numpy RNG stream from non-RNG-consuming
        # controllers under the same benchmark seed.
        if self.epsilon > 0 and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)

        # Greedy action from online network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def update(self, state, action, reward, next_state, done=False, diagnostics=None):
        """
        Store transition and perform one training step.

        Args:
            state: numpy array (19,)
            action: int
            reward: float
            next_state: numpy array (19,)
            done: bool — whether episode ended
            diagnostics: optional rl.training_diagnostics.TrainingDiagnostics —
                when provided, records an overestimation-bias sample each
                gradient step. No-op (and zero overhead) when omitted, so
                this is purely additive instrumentation.
        """
        if self.inference_mode:
            return  # No training during inference

        # Store in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)

        # Train if buffer has enough samples
        if self.replay_buffer.is_ready(self.batch_size):
            loss = self._train_step(diagnostics=diagnostics)
            self.training_losses.append(loss)

        # Update target network periodically
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self._update_target_network()

    def _train_step(self, diagnostics=None):
        """
        Sample mini-batch and perform one gradient update.

        Uses Double DQN:
          - Online network SELECTS the best action for next_state
          - Target network EVALUATES the Q-value of that action
          - This reduces Q-value overestimation

        With prioritized_replay=True, also applies importance-sampling
        weights to the loss and pushes the batch's resulting |TD error|
        back into the buffer's priorities — see rl/replay_buffer.py's
        PrioritizedReplayBuffer docstring for why both halves of that
        round-trip are required, not just the sampling half.
        """
        if self.prioritized_replay:
            states, actions, rewards, next_states, dones, is_weights, indices = (
                self.replay_buffer.sample(self.batch_size)
            )
            is_weights_t = torch.FloatTensor(is_weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size
            )
            is_weights_t = None

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a) for the actions that were taken
        current_q = self.online_net(states_t)
        current_q = current_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # ─── DOUBLE DQN TARGET ───────────────────────────
        with torch.no_grad():
            # Online network SELECTS best action for next state
            online_next_q_all = self.online_net(next_states_t)
            next_actions = online_next_q_all.argmax(dim=1)

            # Target network EVALUATES Q-value of that action
            next_q = self.target_net(next_states_t)
            next_q = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # TD target: r + γ * Q_target(s', argmax Q_online(s', a'))
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

            if diagnostics is not None:
                # Overestimation-bias proxy: how much higher does the
                # online network rate its own chosen action than the
                # (more conservative) target network does, on the same
                # batch? A persistently large positive gap indicates the
                # classic DQN overestimation failure mode.
                online_q_at_next = online_next_q_all.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                diagnostics.record_overestimation_sample(
                    online_q_at_next.mean().item(), next_q.mean().item()
                )

        # Per-sample Huber loss (reduction="none" — see __init__'s comment).
        per_sample_loss = self.loss_fn(current_q, target_q)

        if self.prioritized_replay:
            # Priorities updated from this batch's TD errors BEFORE the
            # loss is reduced, using the same per-sample values the
            # weighted loss below is computed from — both derive from one
            # td_errors tensor so they can't silently disagree.
            td_errors = (current_q.detach() - target_q).abs().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)
            loss = (per_sample_loss * is_weights_t).mean()
        else:
            loss = per_sample_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        return loss.item()

    def _update_target_network(self):
        """Hard copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ══════════════════════════════════════════════════════
    # QUERY METHODS (used by TrafficController + Dashboard)
    # ══════════════════════════════════════════════════════

    def get_q_values(self, state):
        """Return Q-values for all actions at this state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_tensor).squeeze(0).numpy()

        return {ACTION_NAMES[a]: round(float(q_values[a]), 2) for a in range(N_ACTIONS)}

    def get_best_action_name(self, action):
        """Return human-readable action name."""
        return ACTION_NAMES.get(action, "Unknown")

    def get_accuracy(self):
        if self.decision_count == 0:
            return 0
        return (self.correct_count / self.decision_count) * 100

    def get_training_stats(self):
        """Return training statistics for monitoring."""
        recent_losses = self.training_losses[-100:] if self.training_losses else []
        return {
            "train_steps": self.train_step_count,
            "buffer_size": len(self.replay_buffer) if self.replay_buffer else 0,
            "avg_loss_100": round(np.mean(recent_losses), 4) if recent_losses else 0,
            "epsilon": round(self.epsilon, 4),
            "decisions": self.decision_count,
        }

    # ══════════════════════════════════════════════════════
    # SAVE / LOAD
    # ══════════════════════════════════════════════════════

    def save(self, path=None):
        """Save trained model weights and metadata."""
        save_path = path or MODEL_PATH
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "state_dim": STATE_DIM,
                "action_dim": N_ACTIONS,
                "hidden_dim": self.hidden_dim,
                "dueling": self.dueling,
                "prioritized_replay": self.prioritized_replay,
                "train_steps": self.train_step_count,
                "epsilon": self.epsilon,
                "version": "dqn_v1",
            },
            save_path,
        )
        print(f"DQN model saved to {save_path}")

    def load(self, path=None):
        """Load trained model weights."""
        load_path = path or MODEL_PATH

        if not os.path.exists(load_path):
            print(f"No DQN model found at {load_path}. Starting with random weights.")
            return

        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)

        if checkpoint.get("version") != "dqn_v1":
            print(f"Unknown model version: {checkpoint.get('version')}")
            return

        # Reconstruct network if dimensions OR architecture (dueling vs.
        # standard) differ from what this agent was constructed with.
        # `dueling` defaults to False in older checkpoints that predate
        # this flag (saved before DuelingDQNetwork existed), which
        # correctly resolves to "standard architecture" for them.
        hidden_dim = checkpoint.get("hidden_dim", 128)
        checkpoint_dueling = checkpoint.get("dueling", False)
        if hidden_dim != self.hidden_dim or checkpoint_dueling != self.dueling:
            if checkpoint_dueling != self.dueling:
                print(
                    f"DQN checkpoint at {load_path} was saved with "
                    f"dueling={checkpoint_dueling}, but this agent was constructed "
                    f"with dueling={self.dueling} — reconstructing to match the "
                    f"checkpoint's architecture so the saved weights actually apply."
                )
            self.hidden_dim = hidden_dim
            self.dueling = checkpoint_dueling
            self.online_net = build_network(
                STATE_DIM, N_ACTIONS, hidden_dim, dueling=checkpoint_dueling
            ).to(self.device)
            self.target_net = build_network(
                STATE_DIM, N_ACTIONS, hidden_dim, dueling=checkpoint_dueling
            ).to(self.device)

        try:
            self.online_net.load_state_dict(checkpoint["online_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_state_dict"])
        except RuntimeError as e:
            # Checkpoint's input layer doesn't match STATE_DIM (e.g. a new
            # state feature was added since this checkpoint was trained) —
            # the saved weights are for a different-shaped network
            # entirely and cannot be loaded into this one. Fall back to
            # freshly-initialized weights instead of crashing every caller
            # that constructs a DQNAgent.
            print(
                f"DQN checkpoint at {load_path} is incompatible with the "
                f"current STATE_DIM={STATE_DIM} ({e}). Starting with random "
                f"weights — this checkpoint needs retraining."
            )
            return

        self.online_net.eval()
        self.target_net.eval()

        self.train_step_count = checkpoint.get("train_steps", 0)
        self.prioritized_replay = checkpoint.get("prioritized_replay", self.prioritized_replay)

        arch = "dueling" if self.dueling else "standard"
        print(
            f"DQN model loaded ({STATE_DIM} features -> {N_ACTIONS} actions, "
            f"{hidden_dim} hidden [{arch}], {self.train_step_count} train steps)"
        )
