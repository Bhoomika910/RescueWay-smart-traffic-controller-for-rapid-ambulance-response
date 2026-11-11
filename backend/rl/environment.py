# rl/environment.py
import random

from gps.zone_model import ZONE_CONFIG
from rl.ambulance_delay import estimate_clearance
from rl.reward import compute_reward
from rl.state_encoder import StateEncoder
from traffic.simulator import TrafficSimulator

ZONES = [1, 2, 3, 4]
# All 4 patterns traffic/simulator.py actually supports. TRAFFICS below is
# the DEFAULT training rotation and previously silently excluded
# "rush_hour" with no explanation — meaning Q-Learning had in practice
# never been trained on it at all, an undocumented, accidental train/test
# split. This is now made deliberate: ALL_PATTERNS is the full set,
# TRAFFICS is the default (backward-compatible) training rotation, and
# TrafficEnv(training_patterns=...) lets a caller explicitly hold one out
# — see train_offline.py's --held-out-pattern.
ALL_PATTERNS = ["low", "medium", "high", "rush_hour"]
TRAFFICS = ["low", "medium", "high"]
AMBULANCE_SPAWN_PROBABILITY = 0.08


class TrafficEnv:
    def __init__(self, disable_ambulance_delay_reward=False, training_patterns=None):
        """
        disable_ambulance_delay_reward: when True, compute_reward() is
        always called with ambulance_delay=None regardless of the real
        computed delay — i.e. the agent never sees the ambulance-delay
        reward term during training at all, only the zone-correctness
        term. Exists specifically to support a genuine, retrained ablation
        of the ambulance-delay reward component (evaluation/ablation_studies.py's
        Study C) — unlike Study A/B, which only re-score an already-trained,
        frozen policy under a different reward *scale*, this actually
        changes what a freshly-trained agent optimizes for, so the
        resulting policy's real ambulance-delay/stop-rate outcome is
        genuine causal evidence about the reward term's contribution, not
        a post-hoc rescoring. `info["ambulance_delay"]` still reports the
        real measured delay regardless of this flag, so training curves
        can show the causal effect (or lack of it) directly.

        training_patterns: which traffic patterns reset() randomly draws
        from (defaults to TRAFFICS, the pre-existing 3-pattern rotation,
        for backward compatibility). Pass a subset of ALL_PATTERNS to
        deliberately hold one out for a genuine generalization test — see
        train_offline.py's --held-out-pattern, which is exactly what this
        parameter exists to support (design-review finding: this codebase
        previously had zero train/test separation anywhere; every
        evaluation used traffic patterns statistically identical to what
        the agent trained on).
        """
        self.disable_ambulance_delay_reward = disable_ambulance_delay_reward
        self.training_patterns = list(training_patterns) if training_patterns else list(TRAFFICS)
        self.simulator = TrafficSimulator()
        self.encoder = StateEncoder()
        self.max_steps = 30
        self.step_count = 0
        self.active_signal = 1
        self.ambulance_active = False
        self.ambulance_zone = 0
        # Countdown-clock proxy: seconds remaining until the ambulance
        # reaches the intersection, seeded from the same per-zone ETA the
        # live system uses (gps/zone_model.ZONE_CONFIG), and ticking down
        # once per step — NOT a real position/speed simulation (see
        # rl/ambulance_delay.py), but enough to let the agent be scored on
        # whether it actually cleared the zone before arrival, not just
        # whether it reacted once the ambulance showed up.
        self.ambulance_eta = 0
        self.total_reward = 0
        self.episode_ambulance_delays = []
        self.episode_ambulance_stops = 0
        self.episode_ambulance_trips = 0

    def reset(self):
        self.simulator = TrafficSimulator()
        self.simulator.set_traffic_pattern(random.choice(self.training_patterns))
        self.step_count = 0
        self.active_signal = random.choice(ZONES)
        self.ambulance_active = False
        self.ambulance_zone = 0
        self.ambulance_eta = 0
        self.total_reward = 0
        self.episode_ambulance_delays = []
        self.episode_ambulance_stops = 0
        self.episode_ambulance_trips = 0

        for _ in range(5):
            self.simulator.step(active_signal=self.active_signal, is_green=True)

        return self._get_state()

    def step(self, action):
        self.step_count += 1
        prev_signal = self.active_signal

        if action < 4:
            new_signal = action + 1
        else:
            new_signal = self.active_signal

        self.active_signal = new_signal

        metrics = self.simulator.step(
            active_signal=new_signal, is_green=True, current_time=self.step_count
        )

        queue_lengths = {z: metrics[z].queue_length for z in ZONES}
        system_metrics = self.simulator.get_system_metrics()

        ambulance_delay = None
        clearance = None
        if self.ambulance_active:
            clearance = estimate_clearance(
                remaining_eta=self.ambulance_eta,
                queue_length=queue_lengths.get(self.ambulance_zone, 0),
                zone_is_served=(new_signal == self.ambulance_zone),
            )
            ambulance_delay = clearance["deficit"]

        # Use new rich reward function. See __init__'s
        # disable_ambulance_delay_reward docstring — when set, the reward
        # never receives the real ambulance_delay, so the training signal
        # itself has no delay-minimization term at all (a genuine
        # component ablation), while `info["ambulance_delay"]` below still
        # reports the real measured outcome for evaluation.
        reward = compute_reward(
            action=action,
            amb_zone=self.ambulance_zone if self.ambulance_active else 0,
            prev_signal=prev_signal,
            traffic=self.simulator.traffic_pattern,
            queue_lengths=queue_lengths,
            system_metrics=system_metrics,
            ambulance_delay=(None if self.disable_ambulance_delay_reward else ambulance_delay),
        )

        self.total_reward += reward
        self._update_ambulance(queue_lengths, new_signal)

        if self.step_count % 10 == 0:
            self.simulator.set_traffic_pattern(random.choice(self.training_patterns))

        done = self.step_count >= self.max_steps
        if done and self.ambulance_active:
            # Episode boundary reached with an ambulance still in flight.
            # Previously this trip was silently dropped from
            # episode_ambulance_delays, so short episodes could end with
            # zero recorded ambulance outcomes even though the controller
            # had spent most of the episode handling an emergency — the
            # ambulance-delay metric was then undefined for that episode
            # and quietly excluded from any average computed over it.
            # Record the trip's outcome as measured at the boundary.
            final = estimate_clearance(
                remaining_eta=max(0.0, self.ambulance_eta),
                queue_length=queue_lengths.get(self.ambulance_zone, 0),
                zone_is_served=(new_signal == self.ambulance_zone),
            )
            self.episode_ambulance_delays.append(final["deficit"])
            self.episode_ambulance_trips += 1
            if final["will_stop"]:
                self.episode_ambulance_stops += 1

        info = {
            "queue_lengths": queue_lengths,
            "ambulance_active": self.ambulance_active,
            "ambulance_zone": self.ambulance_zone,
            "ambulance_eta": self.ambulance_eta,
            "ambulance_delay": ambulance_delay,
            "ambulance_will_stop": clearance["will_stop"] if clearance else False,
            "traffic_pattern": self.simulator.traffic_pattern,
            "system_metrics": system_metrics,
        }

        next_state = self._get_state()
        return next_state, reward, done, info

    def _get_state(self):
        metrics = self.simulator.get_metrics()
        queue_lengths = {z: metrics[z].queue_length for z in ZONES}

        return self.encoder.encode(
            emergency_active=self.ambulance_active,
            ambulance_zone=self.ambulance_zone if self.ambulance_active else 0,
            active_signal=self.active_signal,
            queue_lengths=queue_lengths,
            emergency_eta=self.ambulance_eta if self.ambulance_active else 0,
        )

    def _update_ambulance(self, queue_lengths=None, active_signal=None):
        if self.ambulance_active:
            self.ambulance_eta -= 1.0
            if self.ambulance_eta <= 0:
                # Trip concludes now — record the final outcome (did it
                # actually clear in time) for episode-level trend tracking
                # used by training scripts to verify the proactive
                # objective is being learned, not just reward going up.
                if queue_lengths is not None and active_signal is not None:
                    final = estimate_clearance(
                        remaining_eta=0.0,
                        queue_length=queue_lengths.get(self.ambulance_zone, 0),
                        zone_is_served=(active_signal == self.ambulance_zone),
                    )
                    self.episode_ambulance_delays.append(final["deficit"])
                    self.episode_ambulance_trips += 1
                    if final["will_stop"]:
                        self.episode_ambulance_stops += 1
                self.ambulance_active = False
                self.ambulance_zone = 0
                self.ambulance_eta = 0
        else:
            if random.random() < AMBULANCE_SPAWN_PROBABILITY:
                self.ambulance_active = True
                self.ambulance_zone = random.choice(ZONES)
                self.ambulance_eta = float(ZONE_CONFIG[self.ambulance_zone]["eta_seconds"])
