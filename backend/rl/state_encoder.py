# rl/state_encoder.py
import numpy as np

from noise import NoiseConfig, SensorNoise
from rl.ambulance_delay import estimate_clearance

EMERGENCY_STATUS = 2
AMBULANCE_ZONES = 5
ACTIVE_SIGNALS = 4
CONGESTION_LEVELS = 3
CONGESTED_ZONES = 4
QUEUE_IMBALANCE = 3
# Coarse "will this ambulance make it through in time" bucket, derived from
# rl/ambulance_delay.py's lead_time_available: 0 = ample lead time to
# spare, 1 = marginal (cutting it close but should still clear in time),
# 2 = predicted to stop. This is what lets the tabular agent condition its
# policy on proactive clearance urgency instead of only on whether an
# emergency is currently flagged active.
CLEARANCE_RISK_LEVELS = 3

N_STATES = (
    EMERGENCY_STATUS
    * AMBULANCE_ZONES
    * ACTIVE_SIGNALS
    * CONGESTION_LEVELS
    * CONGESTED_ZONES
    * QUEUE_IMBALANCE
    * CLEARANCE_RISK_LEVELS
)

N_ACTIONS = 6
ACTION_NAMES = {
    0: "Green Zone 1",
    1: "Green Zone 2",
    2: "Green Zone 3",
    3: "Green Zone 4",
    4: "Extend current green",
    5: "Normal cycle",
}


class StateEncoder:
    def __init__(self, noise_config=None):
        if noise_config is None:
            noise_config = NoiseConfig.from_env()
        self._sensor_noise = SensorNoise(noise_config)
        self.state_dims = [
            EMERGENCY_STATUS,
            AMBULANCE_ZONES,
            ACTIVE_SIGNALS,
            CONGESTION_LEVELS,
            CONGESTED_ZONES,
            QUEUE_IMBALANCE,
            CLEARANCE_RISK_LEVELS,
        ]
        self.total_states = 1
        for d in self.state_dims:
            self.total_states *= d

    def encode(
        self, emergency_active, ambulance_zone, active_signal, queue_lengths, emergency_eta=0.0
    ):
        f_emergency = 1 if emergency_active else 0

        # Validate ambulance_zone: must be 0 (none) or 1-4
        if ambulance_zone < 0 or ambulance_zone > 4:
            import logging

            logging.getLogger("rescueway").warning(
                f"[STATE] ambulance_zone={ambulance_zone} out of range [0-4], clamping"
            )
        f_ambulance = max(0, min(4, ambulance_zone))

        # Validate active_signal: must be 1-4
        if active_signal < 1 or active_signal > 4:
            import logging

            logging.getLogger("rescueway").warning(
                f"[STATE] active_signal={active_signal} out of range [1-4], clamping"
            )
        f_signal = max(0, min(3, active_signal - 1))

        # Apply sensor noise to simulate real-world detector imprecision
        noisy_queues = self._apply_sensor_noise(queue_lengths)

        f_congestion = self._compute_congestion(noisy_queues)
        f_most_congested = self._find_most_congested(noisy_queues)
        f_imbalance = self._compute_imbalance(noisy_queues)
        f_clearance_risk = self._compute_clearance_risk(
            emergency_active, f_ambulance, active_signal, noisy_queues, emergency_eta
        )

        state = 0
        state = state * EMERGENCY_STATUS + f_emergency
        state = state * AMBULANCE_ZONES + f_ambulance
        state = state * ACTIVE_SIGNALS + f_signal
        state = state * CONGESTION_LEVELS + f_congestion
        state = state * CONGESTED_ZONES + f_most_congested
        state = state * QUEUE_IMBALANCE + f_imbalance
        state = state * CLEARANCE_RISK_LEVELS + f_clearance_risk

        # Final safety check: state must be within Q-table bounds
        if state < 0 or state >= N_STATES:
            import logging

            logging.getLogger("rescueway").error(
                f"[STATE] Encoded state {state} out of bounds [0-{N_STATES-1}], "
                f"inputs: emergency={emergency_active}, amb={ambulance_zone}, "
                f"signal={active_signal}, queues={queue_lengths}"
            )
            state = max(0, min(N_STATES - 1, state))

        return state

    def encode_simple(self, ambulance_zone, traffic, active_signal):
        base = {"low": 3, "medium": 8, "high": 15}
        level = base.get(traffic, 8)
        queue_lengths = {
            1: level + np.random.randint(-2, 3),
            2: level + np.random.randint(-2, 3),
            3: level + np.random.randint(-2, 3),
            4: level + np.random.randint(-2, 3),
        }
        return self.encode(
            emergency_active=(ambulance_zone > 0),
            ambulance_zone=ambulance_zone,
            active_signal=active_signal,
            queue_lengths=queue_lengths,
        )

    def decode(self, state):
        clearance_risk = state % CLEARANCE_RISK_LEVELS
        state //= CLEARANCE_RISK_LEVELS
        imbalance = state % QUEUE_IMBALANCE
        state //= QUEUE_IMBALANCE
        most_congested = state % CONGESTED_ZONES
        state //= CONGESTED_ZONES
        congestion = state % CONGESTION_LEVELS
        state //= CONGESTION_LEVELS
        signal = state % ACTIVE_SIGNALS
        state //= ACTIVE_SIGNALS
        ambulance = state % AMBULANCE_ZONES
        state //= AMBULANCE_ZONES
        emergency = state % EMERGENCY_STATUS

        congestion_names = {0: "low", 1: "medium", 2: "high"}
        imbalance_names = {0: "balanced", 1: "moderate", 2: "severe"}
        clearance_risk_names = {0: "ample_lead_time", 1: "marginal", 2: "predicted_stop"}

        return {
            "emergency": bool(emergency),
            "ambulance_zone": ambulance,
            "active_signal": signal + 1,
            "congestion": congestion_names[congestion],
            "most_congested_zone": most_congested + 1,
            "queue_imbalance": imbalance_names[imbalance],
            "clearance_risk": clearance_risk_names[clearance_risk],
        }

    def _compute_congestion(self, queue_lengths):
        total = sum(queue_lengths.values())
        if total < 15:
            return 0
        elif total < 35:
            return 1
        else:
            return 2

    def _find_most_congested(self, queue_lengths):
        if not queue_lengths:
            return 0
        most = max(queue_lengths, key=queue_lengths.get)
        return most - 1

    def _compute_imbalance(self, queue_lengths):
        values = list(queue_lengths.values())
        if not values or max(values) == 0:
            return 0
        std_dev = np.std(values)
        mean = np.mean(values)
        if mean == 0:
            return 0
        cv = std_dev / mean
        if cv < 0.3:
            return 0
        elif cv < 0.6:
            return 1
        else:
            return 2

    def _apply_sensor_noise(self, queue_lengths):
        """Apply sensor noise using the shared NoiseApplicator module."""
        return self._sensor_noise.apply(queue_lengths)

    def _compute_clearance_risk(
        self, emergency_active, ambulance_zone, active_signal, queue_lengths, emergency_eta
    ):
        """
        Bucket "will this ambulance make it through in time" into 3 coarse
        levels using rl/ambulance_delay.py's estimate — see
        CLEARANCE_RISK_LEVELS. Returns 0 (no emergency / plenty of lead
        time) when there's nothing to be at risk of.
        """
        if not emergency_active or ambulance_zone <= 0:
            return 0
        result = estimate_clearance(
            remaining_eta=emergency_eta,
            queue_length=queue_lengths.get(ambulance_zone, 0),
            zone_is_served=(active_signal == ambulance_zone),
        )
        if result["deficit"] > 0:
            return 2  # predicted to stop
        if result["lead_time_available"] <= 3.0:
            return 1  # marginal — will likely clear, but little margin
        return 0  # ample lead time
