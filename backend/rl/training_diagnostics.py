# rl/training_diagnostics.py
"""
Training-time diagnostics for the DQN pipeline.

Purpose: let every tuning decision be based on *evidence* (action
distribution, reward distribution, Q-value evolution, overestimation
signal) instead of guessing from the final reward/delay numbers alone.
Two runs can have similar final reward but wildly different policies —
this module is what actually shows the difference.

Usage (see sumo/train_dqn_sumo.py):
    diag = TrainingDiagnostics()
    ...
    diag.record_step(action, reward, emergency_active)
    ...
    if episode % 10 == 0:
        diag.snapshot_q_values(episode, agent)
    ...
    diag.summary()  # -> dict, JSON-serializable, saved into the training log
"""
import numpy as np

ACTION_NAMES = {0: "Zone1", 1: "Zone2", 2: "Zone3", 3: "Zone4", 4: "Extend", 5: "Normal"}

# Four representative "an ambulance is here" states, one per zone, with a
# moderate queue and mid-range ETA — used to track how the policy's
# preference ordering (which action has the highest Q-value) evolves over
# training. If "Extend" wins here throughout training even as the
# ambulance zone changes, that's a direct, visible signature of policy
# collapse — not an inference from aggregate reward numbers.
PROBE_STATES = {
    f"zone{z}_amb": dict(
        emergency_active=True,
        ambulance_zone=z,
        active_signal=(z % 4) + 1,
        queue_lengths={1: 10, 2: 10, 3: 10, 4: 10},
        emergency_eta=10.0,
        distance=140.0,
    )
    for z in [1, 2, 3, 4]
}


class TrainingDiagnostics:
    def __init__(self, probe_states=None):
        self.probe_states = probe_states if probe_states is not None else PROBE_STATES

        self.action_counts = {a: 0 for a in range(6)}
        self.emergency_action_counts = {a: 0 for a in range(6)}
        self.emergency_correct_count = 0
        self.emergency_step_count = 0

        self.rewards = []
        self.emergency_rewards = []

        self.q_value_snapshots = []  # [{episode, probe: {action_name: q}}]
        self.online_target_gaps = (
            []
        )  # overestimation proxy: online max-Q minus target max-Q on the same batch

    def record_step(self, action, reward, emergency_active, ambulance_zone=0):
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        self.rewards.append(float(reward))
        if emergency_active:
            self.emergency_action_counts[action] = self.emergency_action_counts.get(action, 0) + 1
            self.emergency_rewards.append(float(reward))
            self.emergency_step_count += 1
            if ambulance_zone > 0 and action == ambulance_zone - 1:
                self.emergency_correct_count += 1

    def record_overestimation_sample(self, online_max_q, target_max_q):
        """
        Classic DQN overestimation diagnostic: track the gap between the
        online network's own max-Q estimate and the (more conservative,
        Double-DQN-style) target network's evaluation of that same action,
        on sampled batches during _train_step. A persistently large,
        growing positive gap indicates Q-value overestimation.
        """
        self.online_target_gaps.append(float(online_max_q - target_max_q))

    def snapshot_q_values(self, episode, agent):
        snap = {}
        for name, kwargs in self.probe_states.items():
            state = agent.encode_rich_state(**kwargs)
            q = agent.get_q_values(state)
            snap[name] = q
        self.q_value_snapshots.append({"episode": episode, "q_values": snap})

    @staticmethod
    def _dist_stats(values):
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "std": 0, "p1": 0, "p50": 0, "p99": 0}
        arr = np.array(values, dtype=np.float64)
        return {
            "min": round(float(arr.min()), 2),
            "max": round(float(arr.max()), 2),
            "mean": round(float(arr.mean()), 2),
            "std": round(float(arr.std()), 2),
            "p1": round(float(np.percentile(arr, 1)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
        }

    def action_distribution(self, counts):
        total = sum(counts.values())
        if total == 0:
            return {}
        return {ACTION_NAMES[a]: round(c / total * 100, 1) for a, c in counts.items()}

    def probe_winner_history(self):
        """For each probe state, which action had the highest Q-value at
        each snapshot — the clearest possible view of policy collapse:
        if this stays "Extend" for a zone across the whole run, the agent
        never learned to prefer serving that zone over waiting."""
        history = {name: [] for name in self.probe_states}
        for snap in self.q_value_snapshots:
            for name, q in snap["q_values"].items():
                winner = max(q, key=q.get)
                history[name].append({"episode": snap["episode"], "winner": winner, "q_values": q})
        return history

    def summary(self):
        return {
            "action_distribution_all": self.action_distribution(self.action_counts),
            "action_distribution_emergency": self.action_distribution(self.emergency_action_counts),
            # Denominator context for emergency_accuracy_during_training —
            # was computed but silently dropped before reaching the
            # returned dict (a real, if minor, lost-diagnostic bug caught
            # by adding lint coverage; see ARCHITECTURE.md Part 10).
            "total_emergency_actions": sum(self.emergency_action_counts.values()),
            "emergency_accuracy_during_training": (
                round(self.emergency_correct_count / self.emergency_step_count * 100, 2)
                if self.emergency_step_count > 0
                else 0
            ),
            "reward_stats_all": self._dist_stats(self.rewards),
            "reward_stats_emergency": self._dist_stats(self.emergency_rewards),
            "overestimation_gap_stats": self._dist_stats(self.online_target_gaps),
            "probe_winner_history": self.probe_winner_history(),
        }

    def print_report(self):
        s = self.summary()
        print("\n" + "=" * 70)
        print("TRAINING DIAGNOSTICS")
        print("=" * 70)
        print("Action distribution (all steps):      ", s["action_distribution_all"])
        print("Action distribution (emergency steps):", s["action_distribution_emergency"])
        print(
            f"Emergency correct-zone rate during training: {s['emergency_accuracy_during_training']}%"
        )
        print(f"Reward stats (all steps):      {s['reward_stats_all']}")
        print(f"Reward stats (emergency steps): {s['reward_stats_emergency']}")
        if s["overestimation_gap_stats"]["mean"] or s["overestimation_gap_stats"]["std"]:
            print(f"Online-vs-target Q gap (overestimation proxy): {s['overestimation_gap_stats']}")
        print("\nProbe-state Q-value winner at final snapshot:")
        for name, hist in s["probe_winner_history"].items():
            if hist:
                last = hist[-1]
                print(f"  {name:12s}: winner={last['winner']:8s} q={last['q_values']}")
        print("=" * 70)
