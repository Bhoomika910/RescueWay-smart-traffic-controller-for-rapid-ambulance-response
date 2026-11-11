# rl/state_features.py
"""
Continuous State Feature Extractor for DQN.

Replaces the discrete StateEncoder (1440 integer states) with
a continuous feature vector suitable for neural network input.

The discrete encoder loses information through binning:
  - Queue lengths: 0-50 → 3 bins (low/medium/high)
  - Congestion: exact counts → 3 levels
  - Imbalance: coefficient of variation → 3 levels

The continuous encoder preserves full information:
  - Exact queue lengths (normalized)
  - One-hot encoded categorical features
  - Continuous ETA and phase timing

State vector layout (STATE_DIM = 3*N_ZONES + 7 features; N_ZONES=4 in the
current deployment, so STATE_DIM=19 — see gps.zone_model.N_ZONES for the
single source of truth this is derived from, design-review finding
category 24 "Scalability"):

  [0 : N_ZONES]                    Queue lengths per zone (normalized 0-1)
  [N_ZONES : 2*N_ZONES]            Active signal one-hot
  [2*N_ZONES]                      Emergency active flag (0 or 1)
  [2*N_ZONES+1 : 3*N_ZONES+1]      Ambulance zone one-hot (all zero if no emergency)
  [3*N_ZONES+1]                    Emergency ETA normalized (0-1, 0 if no emergency)
  [3*N_ZONES+2]                    Total congestion normalized (0-1)
  [3*N_ZONES+3]                    Queue imbalance (coefficient of variation, 0-1 capped)
  [3*N_ZONES+4]                    Ambulance distance normalized (0-1, 0 if no emergency)
  [3*N_ZONES+5]                    Seconds needed to clear the ambulance's zone queue, normalized
  [3*N_ZONES+6]                    Predicted clearance deficit (rl/ambulance_delay.py), normalized
                                    — 0 means "will clear in time / no emergency", >0 means "will stop"

The last three features are what make proactive preemption learnable:
they give the agent a continuous signal for "is this zone going to be
ready before the ambulance gets here," derived the same way
rl/ambulance_delay.py and the benchmark compute it, not just "is the flag
currently on."

Scope note on N_ZONES parametrization: this module constructs correctly-
shaped state vectors for any N_ZONES (verified in
tests/test_zone_parametrization.py by constructing a StateFeatureExtractor
at N_ZONES=6 and checking the resulting shape/one-hot placement) — but the
rest of the physical system (Arduino firmware pin wiring, the SUMO network
topology, the frontend's zone panels) remains built for the actual
4-zone deployment. See gps/zone_model.py's N_ZONES docstring for why that
split is a deliberate scope boundary, not an oversight.
"""
import numpy as np

from config import VEHICLE_CLEARANCE_RATE
from gps.zone_model import N_ZONES
from noise import NoiseConfig, SensorNoise
from rl.ambulance_delay import estimate_clearance


class _Layout:
    """Zone-count-dependent index layout, computed once per distinct
    n_zones value. The module-level constants below (QUEUE_START,
    STATE_DIM, N_ACTIONS, ...) are this class evaluated at the actual
    deployment's N_ZONES; StateFeatureExtractor(n_zones=...) can construct
    a different one, which is what makes it possible to genuinely test
    (not just formula-check) this module's behavior at a different zone
    count — see tests/test_zone_parametrization.py."""

    def __init__(self, n_zones):
        self.n_zones = n_zones
        self.queue_start = 0
        self.signal_start = n_zones
        self.emergency_flag_idx = 2 * n_zones
        self.amb_zone_start = 2 * n_zones + 1
        self.eta_idx = 3 * n_zones + 1
        self.congestion_idx = 3 * n_zones + 2
        self.imbalance_idx = 3 * n_zones + 3
        self.distance_idx = 3 * n_zones + 4
        self.clearance_time_idx = 3 * n_zones + 5
        self.deficit_idx = 3 * n_zones + 6
        self.state_dim = 3 * n_zones + 7
        self.n_actions = n_zones + 2  # one "switch to zone i" per zone, + Extend + Normal cycle
        self.extend_action = n_zones
        self.normal_action = n_zones + 1
        self.action_names = {i: f"Green Zone {i + 1}" for i in range(n_zones)}
        self.action_names[self.extend_action] = "Extend current green"
        self.action_names[self.normal_action] = "Normal cycle"


_DEFAULT_LAYOUT = _Layout(N_ZONES)

# ─── Module-level constants for the actual deployment (N_ZONES from
# gps.zone_model) — every existing caller in the codebase imports these
# directly and is unaffected by _Layout/StateFeatureExtractor(n_zones=...)
# existing. ─────────────────────────────────────────────
QUEUE_START = _DEFAULT_LAYOUT.queue_start
SIGNAL_START = _DEFAULT_LAYOUT.signal_start
EMERGENCY_FLAG_IDX = _DEFAULT_LAYOUT.emergency_flag_idx
AMB_ZONE_START = _DEFAULT_LAYOUT.amb_zone_start
ETA_IDX = _DEFAULT_LAYOUT.eta_idx
CONGESTION_IDX = _DEFAULT_LAYOUT.congestion_idx
IMBALANCE_IDX = _DEFAULT_LAYOUT.imbalance_idx
DISTANCE_IDX = _DEFAULT_LAYOUT.distance_idx
CLEARANCE_TIME_IDX = _DEFAULT_LAYOUT.clearance_time_idx
DEFICIT_IDX = _DEFAULT_LAYOUT.deficit_idx
STATE_DIM = _DEFAULT_LAYOUT.state_dim
N_ACTIONS = _DEFAULT_LAYOUT.n_actions
EXTEND_ACTION = _DEFAULT_LAYOUT.extend_action
NORMAL_ACTION = _DEFAULT_LAYOUT.normal_action
ACTION_NAMES = _DEFAULT_LAYOUT.action_names

MAX_QUEUE = 50  # Maximum queue capacity per zone
MAX_ETA = 35.0  # Maximum ETA in seconds (farthest zone's eta_seconds + margin)
MAX_DISTANCE = 600.0  # meters (farthest zone's distance_max in gps/zone_model.ZONE_CONFIG)
MAX_CLEARANCE_TIME = 20.0  # seconds, for normalizing queue-clearance-time
MAX_DEFICIT = 30.0  # seconds, for normalizing predicted delay


class StateFeatureExtractor:
    """
    Extracts continuous feature vectors from traffic state.

    Used by DQN agent instead of the discrete StateEncoder.
    Provides the same interface methods for compatibility.
    """

    def __init__(self, noise_config=None, n_zones=None):
        """
        Args:
            n_zones: overrides gps.zone_model.N_ZONES for this instance.
                Every real caller in the codebase omits this (uses the
                actual deployment's zone count) — it exists so
                tests/test_zone_parametrization.py can construct an
                extractor for a hypothetical differently-sized
                intersection and verify the resulting state vector is
                correctly shaped/populated, proving the encoding scheme
                itself (not just the default-N_ZONES formula) generalizes.
        """
        self._layout = _DEFAULT_LAYOUT if n_zones is None else _Layout(n_zones)
        self.n_zones = self._layout.n_zones
        self.state_dim = self._layout.state_dim
        if noise_config is None:
            noise_config = NoiseConfig.from_env()
        self.sensor_noise = SensorNoise(noise_config)

    def extract(
        self,
        emergency_active,
        ambulance_zone,
        active_signal,
        queue_lengths,
        emergency_eta=0.0,
        distance=0.0,
    ):
        """
        Extract continuous state vector from traffic observations.

        Args:
            emergency_active: bool — whether ambulance is present
            ambulance_zone: int — 0 (none) or 1..N_ZONES
            active_signal: int — 1..N_ZONES (which zone has green)
            queue_lengths: dict — {1: count, 2: count, ..., N_ZONES: count}
            emergency_eta: float — estimated time of arrival in seconds
            distance: float — ambulance's distance to the intersection in
                meters (e.g. gps.zone_model.ZONE_CONFIG[zone]["avg_distance"]);
                0 if no emergency. Provided by the caller rather than looked
                up here, so this module stays decoupled from the GPS zone
                model — see rl/environment.py / services/traffic_controller.py
                for where it's supplied.

        Returns:
            numpy array of shape (self.state_dim,) with float32 values
        """
        layout = self._layout
        n_zones = self.n_zones
        features = np.zeros(layout.state_dim, dtype=np.float32)

        # Apply sensor noise if configured
        noisy_queues = self.sensor_noise.apply(queue_lengths)

        # Queue lengths normalized to [0, 1]
        for z in range(n_zones):
            raw = noisy_queues.get(z + 1, 0)
            features[layout.queue_start + z] = min(1.0, raw / MAX_QUEUE)

        # Active signal one-hot
        signal_idx = max(0, min(n_zones - 1, active_signal - 1))
        features[layout.signal_start + signal_idx] = 1.0

        # Emergency flag
        features[layout.emergency_flag_idx] = 1.0 if emergency_active else 0.0

        # Ambulance zone one-hot (all zeros if no emergency)
        if emergency_active and 1 <= ambulance_zone <= n_zones:
            features[layout.amb_zone_start + ambulance_zone - 1] = 1.0

        # Emergency ETA normalized
        if emergency_active and emergency_eta > 0:
            features[layout.eta_idx] = min(1.0, emergency_eta / MAX_ETA)

        # Total congestion normalized
        total_queue = sum(noisy_queues.values())
        features[layout.congestion_idx] = min(1.0, total_queue / (MAX_QUEUE * n_zones))

        # Queue imbalance (coefficient of variation, capped at 1.0)
        values = list(noisy_queues.values())
        mean_q = np.mean(values) if values else 0
        if mean_q > 0:
            cv = np.std(values) / mean_q
            features[layout.imbalance_idx] = min(1.0, cv)

        # Ambulance distance normalized
        if emergency_active and distance > 0:
            features[layout.distance_idx] = min(1.0, distance / MAX_DISTANCE)

        # Queue-clearance-time and predicted delay, both derived from the
        # same estimate_clearance() the reward/benchmark use, so the
        # agent's state matches what it's being scored against.
        if emergency_active and 1 <= ambulance_zone <= n_zones:
            zone_queue = noisy_queues.get(ambulance_zone, 0)
            features[layout.clearance_time_idx] = min(
                1.0, (zone_queue / VEHICLE_CLEARANCE_RATE) / MAX_CLEARANCE_TIME
            )

            clearance = estimate_clearance(
                remaining_eta=emergency_eta,
                queue_length=zone_queue,
                zone_is_served=(active_signal == ambulance_zone),
            )
            features[layout.deficit_idx] = min(1.0, clearance["deficit"] / MAX_DEFICIT)

        return features

    def get_feature_names(self):
        """Return human-readable feature names for debugging."""
        n_zones = self.n_zones
        names = [f"queue_z{z + 1}" for z in range(n_zones)]
        names += [f"signal_z{z + 1}" for z in range(n_zones)]
        names += ["emergency_active"]
        names += [f"amb_z{z + 1}" for z in range(n_zones)]
        names += [
            "emergency_eta",
            "total_congestion",
            "queue_imbalance",
            "ambulance_distance",
            "queue_clearance_time",
            "clearance_deficit",
        ]
        return names
