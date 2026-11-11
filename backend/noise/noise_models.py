# noise/noise_models.py
"""
RescueWay Noise Models for Robustness Analysis.

Implements 12 noise types across three categories:

SENSOR NOISE (applied at state encoding):
  1. Gaussian noise       — random ±σ% error on queue counts
  2. Detector dropout     — sensor returns 0 with probability p
  3. Systematic bias      — always over/undercount by fixed %

GPS NOISE (applied at emergency trigger):
  4. Zone misclassification  — ambulance reported in wrong zone
  5. ETA error               — reported ETA differs from true ETA
  6. False positive           — phantom ambulance detected
  7. False negative           — real ambulance not detected for N seconds

COMMUNICATION NOISE (applied at API boundary):
  8. Constant latency     — fixed delay (already implemented)
  9. Variable latency     — delay drawn from gamma distribution
  10. Bursty latency      — normally low, occasional spikes
  11. Packet loss          — emergency request dropped entirely

Research question:
  "Under what conditions of sensor noise, GPS error, and communication
  latency does an RL-based emergency traffic signal controller degrade
  below the performance of a hand-designed rule-based controller?"

All models are controlled via environment variables or NoiseConfig.
Default is all noise OFF (clean baseline).
"""

import numpy as np


class NoiseConfig:
    """
    Centralized noise configuration.

    All parameters default to 0 (no noise).
    Set via constructor or environment variables.
    """

    def __init__(
        self,
        # Sensor noise
        sensor_gaussian_std=0.0,
        sensor_dropout_prob=0.0,
        sensor_bias_pct=0.0,
        # GPS noise
        gps_misclass_prob=0.0,
        gps_eta_error_seconds=0.0,
        gps_false_positive_prob=0.0,
        gps_false_negative_prob=0.0,
        gps_false_negative_duration=3.0,
        # Communication noise
        comm_constant_latency=0.0,
        comm_variable_latency_mean=0.0,
        comm_variable_latency_shape=2.0,
        comm_burst_prob=0.0,
        comm_burst_latency=3.0,
        comm_base_latency=0.1,
        comm_packet_loss_prob=0.0,
    ):

        # Sensor noise parameters
        self.sensor_gaussian_std = sensor_gaussian_std
        self.sensor_dropout_prob = sensor_dropout_prob
        self.sensor_bias_pct = sensor_bias_pct

        # GPS noise parameters
        self.gps_misclass_prob = gps_misclass_prob
        self.gps_eta_error_seconds = gps_eta_error_seconds
        self.gps_false_positive_prob = gps_false_positive_prob
        self.gps_false_negative_prob = gps_false_negative_prob
        self.gps_false_negative_duration = gps_false_negative_duration

        # Communication noise parameters
        self.comm_constant_latency = comm_constant_latency
        self.comm_variable_latency_mean = comm_variable_latency_mean
        self.comm_variable_latency_shape = comm_variable_latency_shape
        self.comm_burst_prob = comm_burst_prob
        self.comm_burst_latency = comm_burst_latency
        self.comm_base_latency = comm_base_latency
        self.comm_packet_loss_prob = comm_packet_loss_prob

    def is_clean(self):
        """Return True if all noise is disabled."""
        return (
            self.sensor_gaussian_std == 0
            and self.sensor_dropout_prob == 0
            and self.sensor_bias_pct == 0
            and self.gps_misclass_prob == 0
            and self.gps_eta_error_seconds == 0
            and self.gps_false_positive_prob == 0
            and self.gps_false_negative_prob == 0
            and self.comm_constant_latency == 0
            and self.comm_variable_latency_mean == 0
            and self.comm_burst_prob == 0
            and self.comm_packet_loss_prob == 0
        )

    def to_dict(self):
        """Serialize for logging and reproducibility."""
        return {
            "sensor_gaussian_std": self.sensor_gaussian_std,
            "sensor_dropout_prob": self.sensor_dropout_prob,
            "sensor_bias_pct": self.sensor_bias_pct,
            "gps_misclass_prob": self.gps_misclass_prob,
            "gps_eta_error_seconds": self.gps_eta_error_seconds,
            "gps_false_positive_prob": self.gps_false_positive_prob,
            "gps_false_negative_prob": self.gps_false_negative_prob,
            "gps_false_negative_duration": self.gps_false_negative_duration,
            "comm_constant_latency": self.comm_constant_latency,
            "comm_variable_latency_mean": self.comm_variable_latency_mean,
            "comm_variable_latency_shape": self.comm_variable_latency_shape,
            "comm_burst_prob": self.comm_burst_prob,
            "comm_burst_latency": self.comm_burst_latency,
            "comm_base_latency": self.comm_base_latency,
            "comm_packet_loss_prob": self.comm_packet_loss_prob,
        }

    @staticmethod
    def from_env():
        """Load configuration from environment variables."""
        import os

        return NoiseConfig(
            sensor_gaussian_std=float(os.getenv("RESCUEWAY_SENSOR_NOISE", "0.0")),
            sensor_dropout_prob=float(os.getenv("RESCUEWAY_SENSOR_DROPOUT", "0.0")),
            sensor_bias_pct=float(os.getenv("RESCUEWAY_SENSOR_BIAS", "0.0")),
            gps_misclass_prob=float(os.getenv("RESCUEWAY_GPS_MISCLASS", "0.0")),
            gps_eta_error_seconds=float(os.getenv("RESCUEWAY_GPS_ETA_ERROR", "0.0")),
            gps_false_positive_prob=float(os.getenv("RESCUEWAY_GPS_FALSE_POS", "0.0")),
            gps_false_negative_prob=float(os.getenv("RESCUEWAY_GPS_FALSE_NEG", "0.0")),
            gps_false_negative_duration=float(os.getenv("RESCUEWAY_GPS_FN_DURATION", "3.0")),
            comm_constant_latency=float(os.getenv("RESCUEWAY_COMM_LATENCY", "0.0")),
            comm_variable_latency_mean=float(os.getenv("RESCUEWAY_COMM_VAR_LATENCY", "0.0")),
            comm_variable_latency_shape=float(os.getenv("RESCUEWAY_COMM_VAR_SHAPE", "2.0")),
            comm_burst_prob=float(os.getenv("RESCUEWAY_COMM_BURST_PROB", "0.0")),
            comm_burst_latency=float(os.getenv("RESCUEWAY_COMM_BURST_LATENCY", "3.0")),
            comm_base_latency=float(os.getenv("RESCUEWAY_COMM_BASE_LATENCY", "0.1")),
            comm_packet_loss_prob=float(os.getenv("RESCUEWAY_COMM_PACKET_LOSS", "0.0")),
        )

    def __repr__(self):
        if self.is_clean():
            return "NoiseConfig(clean=True)"
        active = {k: v for k, v in self.to_dict().items() if v != 0}
        return f"NoiseConfig({active})"


# ══════════════════════════════════════════════════════
# PRESET SCENARIOS for Study 3 (combined stress tests)
#
# Labeling note (design-review finding): these severity levels are
# hand-tuned, monotonically-increasing round numbers chosen to give the
# Study 3 sweep a reasonable spread from "no noise" to "everything
# degraded at once" — they are NOT calibrated against measured real-world
# GNSS/detector/network error-rate datasets for this project's specific
# hardware or deployment context (no such measurement campaign has been
# run). Preset names like "urban_dense" describe the INTENDED qualitative
# scenario, not a claim that e.g. 10% GPS misclassification is the
# measured rate for real urban-canyon conditions. Treat every result
# using these presets as a synthetic robustness stress test — useful for
# comparing controllers' relative degradation under increasing noise —
# not as a prediction of real-world failure rates. (Real GNSS urban-canyon
# horizontal error is commonly reported in the ~10-50m range for
# consumer/automotive-grade receivers, which is the general order of
# magnitude these presets' zone-misclassification probabilities are
# loosely motivated by, but that is a qualitative sanity check, not a
# calibration.)
# ══════════════════════════════════════════════════════

NOISE_PRESETS = {
    "clean": NoiseConfig(),
    "suburban": NoiseConfig(
        sensor_gaussian_std=0.05,
        sensor_dropout_prob=0.02,
        gps_misclass_prob=0.05,
        gps_eta_error_seconds=2.0,
        comm_constant_latency=0.5,
    ),
    "urban_dense": NoiseConfig(
        sensor_gaussian_std=0.15,
        sensor_dropout_prob=0.05,
        sensor_bias_pct=0.05,
        gps_misclass_prob=0.10,
        gps_eta_error_seconds=4.0,
        comm_variable_latency_mean=1.5,
        comm_burst_prob=0.05,
        comm_burst_latency=3.0,
    ),
    "poor_gps": NoiseConfig(
        sensor_gaussian_std=0.05,
        gps_misclass_prob=0.25,
        gps_eta_error_seconds=8.0,
        gps_false_negative_prob=0.10,
        gps_false_negative_duration=5.0,
        comm_constant_latency=1.0,
    ),
    "degraded": NoiseConfig(
        sensor_gaussian_std=0.20,
        sensor_dropout_prob=0.10,
        sensor_bias_pct=0.10,
        gps_misclass_prob=0.15,
        gps_eta_error_seconds=5.0,
        gps_false_positive_prob=0.05,
        gps_false_negative_prob=0.08,
        comm_variable_latency_mean=2.0,
        comm_burst_prob=0.10,
        comm_burst_latency=5.0,
        comm_packet_loss_prob=0.03,
    ),
    "extreme": NoiseConfig(
        sensor_gaussian_std=0.30,
        sensor_dropout_prob=0.20,
        sensor_bias_pct=0.15,
        gps_misclass_prob=0.30,
        gps_eta_error_seconds=10.0,
        gps_false_positive_prob=0.10,
        gps_false_negative_prob=0.15,
        gps_false_negative_duration=8.0,
        comm_variable_latency_mean=3.0,
        comm_burst_prob=0.15,
        comm_burst_latency=5.0,
        comm_packet_loss_prob=0.10,
    ),
}


# ══════════════════════════════════════════════════════
# SENSOR NOISE MODELS
# Applied at state encoding (before RL agent sees data)
# ══════════════════════════════════════════════════════


class SensorNoise:
    """
    Applies sensor noise to queue length observations.

    Real-world detectors (inductive loops, cameras, radar) produce
    noisy measurements. This class simulates three failure modes.
    """

    def __init__(self, config: NoiseConfig):
        self.config = config

    def apply(self, queue_lengths):
        """
        Apply all configured sensor noise models to queue lengths.

        Args:
            queue_lengths: dict {1: count, 2: count, 3: count, 4: count}

        Returns:
            dict with noisy queue lengths (same keys)
        """
        if (
            self.config.sensor_gaussian_std == 0
            and self.config.sensor_dropout_prob == 0
            and self.config.sensor_bias_pct == 0
        ):
            return queue_lengths

        noisy = {}
        for zone, count in queue_lengths.items():
            value = float(count)

            # 1. Systematic bias (always over/undercount)
            if self.config.sensor_bias_pct != 0:
                value = value * (1.0 + self.config.sensor_bias_pct)

            # 2. Gaussian noise (random per-step variation)
            if self.config.sensor_gaussian_std > 0:
                noise = np.random.normal(0, self.config.sensor_gaussian_std * max(1, value))
                value = value + noise

            # 3. Detector dropout (sensor fails entirely)
            if self.config.sensor_dropout_prob > 0:
                if np.random.random() < self.config.sensor_dropout_prob:
                    value = 0  # Sensor reports zero

            noisy[zone] = max(0, int(round(value)))

        return noisy


# ══════════════════════════════════════════════════════
# GPS NOISE MODELS
# Applied at emergency detection/trigger point
# ══════════════════════════════════════════════════════


class GPSNoise:
    """
    Applies GPS-related noise to ambulance detection.

    Real GPS systems have spatial error (wrong zone), timing
    error (wrong ETA), false alarms, and dropout periods.
    """

    # Controls how sharply zone-misclassification probability falls off
    # with distance-band separation in apply_zone() below — see that
    # method's docstring. Not empirically fit to measured GNSS error
    # data (no such dataset is available for this prototype's specific
    # zone geometry); it's a documented, physically-motivated shape
    # (adjacent bands dominate, distant bands are rare but not
    # impossible) rather than a claim of calibrated real-world accuracy.
    # Treat the resulting noise sweep as a synthetic stress test, not a
    # measured real-world error rate.
    ADJACENCY_DECAY = 1.0

    def __init__(self, config: NoiseConfig):
        self.config = config
        # Track false negative state
        self._fn_active = False
        self._fn_remaining = 0.0

    def apply_zone(self, true_zone):
        """
        Apply zone misclassification noise.

        With probability gps_misclass_prob, report the ambulance in a
        DIFFERENT zone than the true one — weighted toward the
        radially-adjacent zone(s) (one distance-band away), not picked
        uniformly among all three other zones (design-review finding:
        this was previously an exact uniform draw over the other 3
        zones, while this docstring already claimed — incorrectly — that
        it favored an adjacent zone).

        Physical grounding: this project's 4 zones are ordered distance
        bands from the intersection (gps/zone_model.ZONE_CONFIG: Zone 1 =
        0-80m, Zone 2 = 80-150m, Zone 3 = 200-300m, Zone 4 = 400-500m).
        Real GNSS positioning error (chiefly multipath/urban-canyon
        reflection) is a roughly continuous distance error, so a
        true-Zone-2 ambulance is far more likely to be misread as the
        neighboring Zone 1 or Zone 3 than as the much-farther Zone 4 —
        confusing radially-adjacent bands is the dominant error mode.
        Weights fall off exponentially with band-distance
        (ADJACENCY_DECAY above) so adjacent zones get most of the
        misclassification mass while still allowing the occasional
        larger GNSS jump (e.g. a bad multipath reflection) rather than
        making it impossible.

        Args:
            true_zone: int 1-4 (actual ambulance zone)

        Returns:
            int 1-4 (possibly wrong zone)
        """
        if self.config.gps_misclass_prob <= 0:
            return true_zone

        if np.random.random() < self.config.gps_misclass_prob:
            other_zones = [z for z in [1, 2, 3, 4] if z != true_zone]
            weights = np.array(
                [np.exp(-self.ADJACENCY_DECAY * abs(z - true_zone)) for z in other_zones],
                dtype=float,
            )
            weights /= weights.sum()
            return int(np.random.choice(other_zones, p=weights))

        return true_zone

    def apply_eta(self, true_eta):
        """
        Apply ETA estimation error.

        Real GPS-based ETA depends on map matching, route prediction,
        and speed estimation — all sources of error.

        Args:
            true_eta: float (true ETA in seconds)

        Returns:
            float (noisy ETA, always >= 0)
        """
        if self.config.gps_eta_error_seconds <= 0:
            return true_eta

        error = np.random.uniform(
            -self.config.gps_eta_error_seconds,
            self.config.gps_eta_error_seconds,
        )
        return max(0.0, true_eta + error)

    def check_false_positive(self):
        """
        Check if a false positive ambulance detection should occur.

        Returns:
            (bool, int): (is_false_positive, phantom_zone)
            If True, a non-existent ambulance is "detected" in phantom_zone.
        """
        if self.config.gps_false_positive_prob <= 0:
            return False, 0

        if np.random.random() < self.config.gps_false_positive_prob:
            phantom_zone = np.random.randint(1, 5)
            return True, phantom_zone

        return False, 0

    def check_false_negative(self, ambulance_present):
        """
        Check if a real ambulance should be hidden (GPS dropout).

        Simulates GPS loss in tunnels, urban canyons, or network failure.
        Once triggered, the ambulance is invisible for a duration.

        Args:
            ambulance_present: bool (is there really an ambulance?)

        Returns:
            bool: True if ambulance should be hidden from the controller
        """
        if self.config.gps_false_negative_prob <= 0:
            return False

        # If false negative is currently active, count down
        if self._fn_active:
            self._fn_remaining -= 0.5  # assumes 0.5s step
            if self._fn_remaining <= 0:
                self._fn_active = False
                self._fn_remaining = 0
            return True  # Still hidden

        # Check if new false negative should trigger
        if ambulance_present:
            if np.random.random() < self.config.gps_false_negative_prob:
                self._fn_active = True
                self._fn_remaining = self.config.gps_false_negative_duration
                return True  # Start hiding

        return False

    def reset(self):
        """Reset false negative state between episodes."""
        self._fn_active = False
        self._fn_remaining = 0.0


# ══════════════════════════════════════════════════════
# COMMUNICATION NOISE MODELS
# Applied at API boundary (emergency request handling)
# ══════════════════════════════════════════════════════


class CommNoise:
    """
    Applies communication latency and packet loss.

    Real V2I (vehicle-to-infrastructure) communication has
    variable delays, occasional spikes, and packet drops.
    """

    def __init__(self, config: NoiseConfig):
        self.config = config

    def get_latency(self):
        """
        Compute the communication delay for this request.

        Returns:
            float: delay in seconds (0.0 if no noise configured)
        """
        # Constant latency (simplest model)
        if self.config.comm_constant_latency > 0:
            return self.config.comm_constant_latency

        # Variable latency (gamma distribution)
        if self.config.comm_variable_latency_mean > 0:
            shape = self.config.comm_variable_latency_shape
            scale = self.config.comm_variable_latency_mean / shape
            latency = np.random.gamma(shape, scale)

            # Bursty overlay: occasional large spikes
            if self.config.comm_burst_prob > 0:
                if np.random.random() < self.config.comm_burst_prob:
                    latency = self.config.comm_burst_latency

            return max(0.0, latency)

        # Bursty latency only (no variable base)
        if self.config.comm_burst_prob > 0:
            if np.random.random() < self.config.comm_burst_prob:
                return self.config.comm_burst_latency
            return self.config.comm_base_latency

        return 0.0

    def check_packet_loss(self):
        """
        Check if this emergency request should be dropped.

        Returns:
            bool: True if the packet is lost (request should be ignored)
        """
        if self.config.comm_packet_loss_prob <= 0:
            return False

        return np.random.random() < self.config.comm_packet_loss_prob


# ══════════════════════════════════════════════════════
# CONVENIENCE: Combined noise applicator
# ══════════════════════════════════════════════════════


class NoiseApplicator:
    """
    Convenience wrapper combining all three noise categories.

    Usage:
        config = NoiseConfig(sensor_gaussian_std=0.15, gps_misclass_prob=0.10)
        noise = NoiseApplicator(config)

        noisy_queues = noise.sensor.apply(queue_lengths)
        noisy_zone = noise.gps.apply_zone(true_zone)
        latency = noise.comm.get_latency()
        dropped = noise.comm.check_packet_loss()
    """

    def __init__(self, config: NoiseConfig = None):
        if config is None:
            config = NoiseConfig()  # All noise off
        self.config = config
        self.sensor = SensorNoise(config)
        self.gps = GPSNoise(config)
        self.comm = CommNoise(config)

    def reset(self):
        """Reset stateful noise (false negative tracking)."""
        self.gps.reset()

    def __repr__(self):
        return f"NoiseApplicator({self.config})"
