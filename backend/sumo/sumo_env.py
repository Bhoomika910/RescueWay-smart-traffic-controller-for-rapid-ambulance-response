# sumo/sumo_env.py
"""
RescueWay SUMO Training Environment for DQN.

Wraps SUMO + TraCI into a gym-like interface:
    env = SumoTrafficEnv(pattern="medium")
    state = env.reset()
    while not done:
        action = agent.choose(state)
        next_state, reward, done = env.step(action)

Training architecture (Option B2):
    SUMO → SumoTrafficEnv → DQN Training → dqn_model.pt
    Then dqn_model.pt is loaded into the live system.

Each episode:
    1. Generates fresh routes with deterministic seed
    2. Starts SUMO in headless mode
    3. Runs for `duration` seconds
    4. Agent makes decisions every `green_duration` seconds
    5. Safe transitions (yellow → all-red → green) between phases
    6. Ambulance injected at random time for emergency training
    7. Episode ends when SUMO simulation completes
"""
import os
import random
import sys

import numpy as np

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import traci
from generate_routes import generate_routes
from sumo_adapter import EDGE_TO_ZONE, SUMOAdapter

from gps.zone_model import ZONE_CONFIG
from rl.reward import compute_reward
from rl.state_features import StateFeatureExtractor

# ─── Configuration ────────────────────────────────────

SUMO_BINARY = "sumo"  # headless for training (no GUI)
NET_FILE = os.path.join(os.path.dirname(__file__), "intersection.net.xml")
TEMP_ROUTE_FILE = os.path.join(os.path.dirname(__file__), "_train_routes.rou.xml")

# Signal timing (must match live system)
GREEN_DURATION = 5  # seconds per green phase
YELLOW_DURATION = 2  # seconds for yellow
ALL_RED_DURATION = 1  # seconds for all-red clearance

# Episode defaults
DEFAULT_DURATION = 300  # seconds per episode
DEFAULT_PATTERN = "medium"


class SumoTrafficEnv:
    """
    Gym-like environment for DQN training in SUMO.

    Interface:
        state = env.reset(seed=42)           → numpy array (19,)
        state, reward, done = env.step(action)  → (array, float, bool)
        env.close()

    The environment handles:
        - SUMO process lifecycle (start/stop per episode)
        - Route generation with deterministic seeds
        - Safe signal transitions (yellow → all-red → green)
        - Ambulance detection and emergency reward
        - State extraction via StateFeatureExtractor
    """

    def __init__(
        self,
        pattern="medium",
        duration=DEFAULT_DURATION,
        ambulance_prob=0.7,
        use_gui=False,
        use_discrete_encoder=False,
    ):
        """
        Args:
            pattern: Traffic pattern ("low", "medium", "high", "rush_hour")
            duration: Episode length in SUMO seconds
            ambulance_prob: Probability of ambulance appearing per episode
            use_gui: Whether to show SUMO GUI (slow, for debugging only)
            use_discrete_encoder: use rl.state_encoder.StateEncoder (the
                tabular Q-learning agent's discrete state space) instead
                of the continuous StateFeatureExtractor (DQN's). Default
                False — every existing caller trains DQN here and is
                unaffected. Added so tabular Q-learning can train natively
                in SUMO (sumo/train_qlearning_sumo.py) instead of only
                ever training in the simplified custom simulator, closing
                half of the asymmetry behind the documented SUMO-vs-
                simplified-environment domain gap: DQN previously trained
                exclusively in SUMO with no simplified-only counterpart,
                while Q-learning trained exclusively in the simplified
                simulator with no SUMO counterpart — neither agent had
                ever been trained in both environments the same way,
                which is what made a like-for-like domain-gap comparison
                impossible. See ARCHITECTURE.md's domain-gap section.
        """
        self.pattern = pattern
        self.duration = duration
        self.ambulance_prob = ambulance_prob
        self.use_gui = use_gui
        self.use_discrete_encoder = use_discrete_encoder

        # State extractor: continuous 19-dim vector (DQN) by default, or
        # the discrete tabular encoder (Q-learning) when opted in.
        if use_discrete_encoder:
            from rl.state_encoder import StateEncoder

            self.encoder = StateEncoder()
            self.feature_extractor = None
        else:
            self.feature_extractor = StateFeatureExtractor()
            self.encoder = None

        # Internal state
        self.adapter = None
        self.current_zone = 1
        self.step_count = 0
        self.decision_count = 0
        self.emergency_active = False
        self.ambulance_zone = 0
        self.emergency_eta = 0.0
        self.ambulance_distance = 0.0
        # Real per-vehicle waiting time from TraCI
        # (sumo_adapter.get_ambulance_waiting_time) — the ground-truth
        # ambulance delay for this environment, as opposed to the
        # countdown-clock proxy the non-SUMO simple simulator/benchmark
        # use (rl/ambulance_delay.estimate_clearance) when no real vehicle
        # simulation is available.
        #
        # NOTE: this is a LIVE SNAPSHOT, not an aggregate — it resets to
        # 0.0 the instant an ambulance is no longer detected (see
        # _detect_ambulance below). It must not be read at episode end as
        # "the ambulance's delay for this episode": by the time a 300s
        # episode ends, any ambulance that appeared has almost always
        # already cleared, so this value is 0.0 essentially always at
        # that point. Use `ambulance_delay_events` / `get_metrics()`'s
        # `avg_ambulance_delay` for episode-level reporting instead.
        self.ambulance_waiting_time = 0.0
        # Peak waiting time observed so far during the emergency currently
        # in progress (if any). Flushed into `ambulance_delay_events` the
        # moment the emergency ends (ambulance no longer detected).
        self._current_emergency_peak_wait = 0.0
        # One entry per ambulance emergency that occurred this episode:
        # the peak per-vehicle waiting time TraCI measured while that
        # ambulance was present. This is the real, ground-truth "how much
        # was this ambulance actually delayed" signal — see get_metrics().
        self.ambulance_delay_events = []
        self.is_running = False

        # Episode tracking
        self.episode_reward = 0
        self.episode_switches = 0

    def reset(self, seed=42):
        """
        Start a new episode.

        Generates routes, launches SUMO, and returns initial state.

        Args:
            seed: Random seed for reproducible traffic generation

        Returns:
            numpy array of shape (19,) — initial state
        """
        # Close any running SUMO instance
        self.close()

        # Seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Decide whether ambulance appears this episode
        ambulance_zone = None
        ambulance_time = None
        if random.random() < self.ambulance_prob:
            ambulance_zone = random.randint(1, 4)
            # Ambulance arrives between 20% and 80% of episode
            ambulance_time = random.randint(int(self.duration * 0.2), int(self.duration * 0.8))

        # Generate routes
        generate_routes(
            pattern=self.pattern,
            duration=self.duration,
            ambulance_zone=ambulance_zone,
            ambulance_time=ambulance_time,
            seed=seed,
            output_file=TEMP_ROUTE_FILE,
        )

        # Start SUMO
        binary = "sumo-gui" if self.use_gui else SUMO_BINARY
        sumo_cmd = [
            binary,
            "-n",
            NET_FILE,
            "-r",
            TEMP_ROUTE_FILE,
            "--step-length",
            "1",
            "--no-step-log",
            "true",
            "--time-to-teleport",
            "300",
            "--waiting-time-memory",
            "1000",
            "--no-warnings",
            "true",
            # Pins SUMO's own internal RNG (departure jitter, any
            # sublane/car-following stochasticity) to the same seed used
            # for route generation and Python's random/np.random above.
            # Previously only the Python-side seeding was set — SUMO's own
            # process was left to its own default (time-based) seed, so
            # two "identical seed" episodes were not actually fully
            # deterministic.
            "--seed",
            str(seed),
        ]

        traci.start(sumo_cmd)
        self.adapter = SUMOAdapter()
        self.is_running = True

        # Reset state
        self.current_zone = 1
        self.step_count = 0
        self.decision_count = 0
        self.emergency_active = False
        self.ambulance_zone = 0
        self.emergency_eta = 0.0
        self.ambulance_distance = 0.0
        self.ambulance_waiting_time = 0.0
        self._current_emergency_peak_wait = 0.0
        self.ambulance_delay_events = []
        self.episode_reward = 0
        self.episode_switches = 0

        # Set initial signal
        self.adapter.set_signal(1)

        # Advance a few steps to let vehicles enter
        for _ in range(3):
            self.adapter.step()
            self.step_count += 1

        # Return initial state
        return self._get_state()

    def step(self, action):
        """
        Execute one decision cycle.

        1. Interpret action (0-5)
        2. Execute safe transition if zone changes
        3. Hold green for GREEN_DURATION seconds
        4. Return new state, reward, done

        Args:
            action: int 0-5 (same action space as Q-agent)

        Returns:
            (next_state, reward, done) tuple
        """
        if not self.is_running:
            return self._get_state(), 0.0, True

        self.decision_count += 1

        # Interpret action
        if action < 4:
            target_zone = action + 1
        else:
            target_zone = self.current_zone  # extend or normal cycle

        # Get pre-decision queue lengths for reward
        queue_lengths = self.adapter.get_queue_lengths()

        # Check for ambulance
        self._detect_ambulance()

        # Compute reward using same function as custom simulator. Unlike
        # the simple simulator/benchmark (which only ever has a proxy
        # estimate of ambulance delay), SUMO measures it directly via
        # TraCI (self.ambulance_waiting_time, set in _detect_ambulance) —
        # this is real per-vehicle waiting time, not an estimate, so it's
        # passed straight through as the ground-truth training signal.
        reward = compute_reward(
            action=action,
            amb_zone=self.ambulance_zone if self.emergency_active else 0,
            prev_signal=self.current_zone,
            traffic=self.pattern,
            queue_lengths=queue_lengths,
            ambulance_delay=self.ambulance_waiting_time if self.emergency_active else None,
        )

        self.episode_reward += reward

        # Execute safe transition if zone changes
        if target_zone != self.current_zone:
            done = self._safe_transition(self.current_zone, target_zone)
            if done:
                return self._get_state(), reward, True
            self.episode_switches += 1

        self.current_zone = target_zone

        # Hold green for GREEN_DURATION
        done = self._hold_green(GREEN_DURATION)

        # Get next state
        next_state = self._get_state()

        return next_state, reward, done

    def _safe_transition(self, from_zone, to_zone):
        """
        Execute yellow → all-red → green transition.

        Returns True if simulation ended during transition.
        """
        # Phase 1: Yellow
        self.adapter.set_yellow()
        for _ in range(YELLOW_DURATION):
            if not self._advance_step():
                return True
            self._detect_ambulance()

        # Phase 2: All Red
        self.adapter.set_all_red()
        for _ in range(ALL_RED_DURATION):
            if not self._advance_step():
                return True
            self._detect_ambulance()

        # Phase 3: New Green
        self.adapter.set_signal(to_zone)
        return False

    def _hold_green(self, seconds):
        """
        Hold current green for specified seconds.

        Returns True if simulation ended.
        """
        for _ in range(seconds):
            if not self._advance_step():
                return True
            self._detect_ambulance()
        return False

    def _advance_step(self):
        """
        Advance SUMO by one step.

        Returns False if simulation is over.
        """
        try:
            self.adapter.step()
            self.step_count += 1

            # Check if simulation ended
            if self.step_count >= self.duration:
                return False
            if traci.simulation.getMinExpectedNumber() <= 0:
                return False

            return True
        except traci.exceptions.FatalTraCIError:
            self.is_running = False
            return False

    def _detect_ambulance(self):
        """Check for ambulance in the simulation."""
        try:
            amb_present, amb_zone = self.adapter.detect_ambulance()

            # Also check by vehicle list for ambulances on internal edges
            if not amb_present:
                for vid in traci.vehicle.getIDList():
                    if "ambulance" in vid.lower():
                        amb_present = True
                        road = traci.vehicle.getRoadID(vid)
                        if road in EDGE_TO_ZONE:
                            amb_zone = EDGE_TO_ZONE[road]
                        else:
                            amb_zone = self.ambulance_zone or 3
                        break

            was_active = self.emergency_active
            self.emergency_active = amb_present and amb_zone > 0
            if self.emergency_active:
                self.ambulance_zone = amb_zone
                # Estimate ETA and distance from the real vehicle position
                try:
                    for vid in traci.vehicle.getIDList():
                        if "ambulance" in vid.lower():
                            speed = traci.vehicle.getSpeed(vid)
                            dist = traci.vehicle.getLanePosition(vid)
                            lane_len = traci.lane.getLength(traci.vehicle.getLaneID(vid))
                            remaining = max(0, lane_len - dist)
                            self.emergency_eta = remaining / max(speed, 1.0)
                            self.ambulance_distance = remaining
                            break
                except Exception:
                    self.emergency_eta = 10.0
                    self.ambulance_distance = ZONE_CONFIG.get(amb_zone, {}).get("avg_distance", 0.0)
                # Real per-vehicle waiting time — the ground-truth ambulance
                # delay this environment can measure directly (see __init__).
                self.ambulance_waiting_time = self.adapter.get_ambulance_waiting_time()
                self._current_emergency_peak_wait = max(
                    self._current_emergency_peak_wait, self.ambulance_waiting_time
                )
            else:
                # Emergency just ended (was tracked last tick, isn't now):
                # flush the peak delay observed during it into the
                # episode's record before resetting. This is what makes
                # ambulance delay reporting survive past the moment the
                # ambulance clears, instead of always reading 0 at episode
                # end (see the note on ambulance_waiting_time in __init__).
                if was_active and self._current_emergency_peak_wait > 0:
                    self.ambulance_delay_events.append(self._current_emergency_peak_wait)
                self._current_emergency_peak_wait = 0.0
                self.ambulance_zone = 0
                self.emergency_eta = 0.0
                self.ambulance_distance = 0.0
                self.ambulance_waiting_time = 0.0

        except Exception:
            pass

    def _get_state(self):
        """Extract the configured state representation (continuous
        StateFeatureExtractor by default, or the discrete StateEncoder
        when use_discrete_encoder=True) from current SUMO state."""
        try:
            queue_lengths = self.adapter.get_queue_lengths()
        except Exception:
            queue_lengths = {1: 0, 2: 0, 3: 0, 4: 0}

        if self.use_discrete_encoder:
            # StateEncoder.encode() has no `distance` parameter — the
            # tabular agent's state space doesn't include raw distance,
            # only the ETA-derived clearance-risk bucket computed
            # internally from emergency_eta.
            return self.encoder.encode(
                emergency_active=self.emergency_active,
                ambulance_zone=self.ambulance_zone,
                active_signal=self.current_zone,
                queue_lengths=queue_lengths,
                emergency_eta=self.emergency_eta,
            )

        return self.feature_extractor.extract(
            emergency_active=self.emergency_active,
            ambulance_zone=self.ambulance_zone,
            active_signal=self.current_zone,
            queue_lengths=queue_lengths,
            emergency_eta=self.emergency_eta,
            distance=self.ambulance_distance,
        )

    def get_metrics(self):
        """Return episode metrics for logging.

        `ambulance_waiting_time` is kept for backward compatibility but is
        a live snapshot (see its docstring in __init__) — it is normally
        0.0 by the time an episode ends. For "how delayed were the
        ambulances that appeared this episode", use `avg_ambulance_delay`
        / `max_ambulance_delay` / `ambulance_delay_events`, which are
        built from the peak waiting time recorded during each emergency
        (including one still in progress when the episode ends).
        """
        try:
            queue_lengths = self.adapter.get_queue_lengths()
            total_queue = sum(queue_lengths.values())
        except Exception:
            total_queue = 0

        completed_delays = list(self.ambulance_delay_events)
        if self.emergency_active and self._current_emergency_peak_wait > 0:
            completed_delays.append(self._current_emergency_peak_wait)

        avg_ambulance_delay = (
            round(sum(completed_delays) / len(completed_delays), 2) if completed_delays else 0.0
        )
        max_ambulance_delay = round(max(completed_delays), 2) if completed_delays else 0.0

        return {
            "step": self.step_count,
            "decisions": self.decision_count,
            "episode_reward": round(self.episode_reward, 2),
            "switches": self.episode_switches,
            "total_queue": total_queue,
            "emergency_active": self.emergency_active,
            "ambulance_zone": self.ambulance_zone,
            "ambulance_waiting_time": round(self.ambulance_waiting_time, 2),
            "avg_ambulance_delay": avg_ambulance_delay,
            "max_ambulance_delay": max_ambulance_delay,
            "ambulance_delay_events": [round(d, 2) for d in completed_delays],
            "ambulance_count": len(completed_delays),
        }

    def close(self):
        """Shut down SUMO process."""
        if self.is_running:
            try:
                traci.close()
            except Exception:
                pass
            self.is_running = False
