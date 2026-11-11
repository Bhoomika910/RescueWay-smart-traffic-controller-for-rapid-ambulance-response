# rl/q_agent.py
import os
import pickle

import numpy as np

from rl.state_encoder import ACTION_NAMES, N_ACTIONS, N_STATES, StateEncoder

QTABLE_PATH = os.path.join(os.path.dirname(__file__), "q_table_v2.pkl")
QTABLE_PATH_OLD = os.path.join(os.path.dirname(__file__), "q_table.pkl")

TRAFFIC_MAP = {"low": 0, "medium": 1, "high": 2}


class QAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0):
        self.q = np.zeros((N_STATES, N_ACTIONS))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.encoder = StateEncoder()
        self.decision_count = 0
        self.correct_count = 0

    def encode_state(self, zone: int, traffic: str, signal: int) -> int:
        """Backward-compatible: encode from simple inputs"""
        return self.encoder.encode_simple(zone, traffic, signal)

    def encode_rich_state(
        self,
        emergency_active: bool,
        ambulance_zone: int,
        active_signal: int,
        queue_lengths: dict,
        emergency_eta: float = 0.0,
    ) -> int:
        """New: encode from full simulator metrics"""
        return self.encoder.encode(
            emergency_active=emergency_active,
            ambulance_zone=ambulance_zone,
            active_signal=active_signal,
            queue_lengths=queue_lengths,
            emergency_eta=emergency_eta,
        )

    def choose(self, state: int) -> int:
        self.decision_count += 1
        # Skip the RNG draw entirely when epsilon <= 0 (pure exploitation,
        # e.g. benchmark/inference mode): `np.random.random() < 0.0` can
        # never be True, so drawing anyway just burns one extra sample off
        # the shared numpy RNG stream per decision. In a benchmark that
        # seeds np.random once per episode and then also uses it to
        # generate vehicle arrivals, that extra draw desyncs the arrival
        # sequence this agent sees from the one non-RNG-consuming
        # controllers (Fixed Timer, Rule-Based) see under the same seed,
        # breaking the "identical conditions" comparison.
        if self.epsilon > 0 and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool = False):
        """
        `done` defaults to False for backward compatibility with existing
        callers. When True, the bootstrap term (gamma * best_next) is
        zeroed — matching DQNAgent.update()'s terminal masking, which this
        method previously lacked entirely despite dqn_agent.py's own
        docstring calling QAgent and DQNAgent "drop-in replacements" for
        each other.
        """
        best_next = np.max(self.q[next_state])
        td_target = reward + self.gamma * best_next * (1.0 - float(done))
        self.q[state, action] += self.alpha * (td_target - self.q[state, action])

    def get_q_values(self, state: int) -> dict:
        """Return Q-values for all actions at this state"""
        return {ACTION_NAMES[a]: round(float(self.q[state, a]), 2) for a in range(N_ACTIONS)}

    def get_best_action_name(self, action: int) -> str:
        return ACTION_NAMES.get(action, "Unknown")

    def get_accuracy(self) -> float:
        if self.decision_count == 0:
            return 0
        return (self.correct_count / self.decision_count) * 100

    def save(self, path=None):
        save_path = path or QTABLE_PATH
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q,
                    "n_states": N_STATES,
                    "n_actions": N_ACTIONS,
                    "version": 2,
                },
                f,
            )
        print(f"Q-table v2 saved to {save_path}")

    def load(self, path=None):
        load_path = path or QTABLE_PATH

        # Try loading v2 first
        if os.path.exists(load_path):
            with open(load_path, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict) and data.get("version") == 2:
                loaded = data["q_table"]
                if loaded.shape != (N_STATES, N_ACTIONS):
                    # The state space changed (e.g. a new feature dimension
                    # was added) since this table was saved — loading it
                    # anyway would silently misindex every lookup rather
                    # than raise, since numpy won't complain about a
                    # differently-shaped array being assigned here. Fall
                    # through to starting fresh instead.
                    print(
                        f"Q-table shape {loaded.shape} does not match current "
                        f"state space ({N_STATES}, {N_ACTIONS}) — state encoding "
                        f"changed since this table was trained. Starting fresh."
                    )
                else:
                    self.q = loaded
                    print(f"Q-table v2 loaded ({N_STATES} states × {N_ACTIONS} actions)")
                return
            elif isinstance(data, np.ndarray):
                print("Found old v1 Q-table — will retrain with new state space")

        # Try old path
        if os.path.exists(QTABLE_PATH_OLD):
            print("Found old q_table.pkl — need to retrain for v2 state space")

        print(f"Starting fresh Q-table ({N_STATES} states × {N_ACTIONS} actions)")
