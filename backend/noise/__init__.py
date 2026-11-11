# noise/__init__.py
"""
RescueWay Noise Models for Robustness Analysis.

Usage:
    from noise import NoiseConfig, NoiseApplicator, NOISE_PRESETS

    # Clean (no noise)
    config = NoiseConfig()

    # Custom noise
    config = NoiseConfig(sensor_gaussian_std=0.15, gps_misclass_prob=0.10)

    # Preset scenarios
    config = NOISE_PRESETS["urban_dense"]

    # From environment variables
    config = NoiseConfig.from_env()

    # Apply
    noise = NoiseApplicator(config)
    noisy_queues = noise.sensor.apply(queue_lengths)
    noisy_zone = noise.gps.apply_zone(true_zone)
    latency = noise.comm.get_latency()
"""
from noise.noise_models import (
    NOISE_PRESETS,
    CommNoise,
    GPSNoise,
    NoiseApplicator,
    NoiseConfig,
    SensorNoise,
)

__all__ = [
    "NoiseConfig",
    "NoiseApplicator",
    "SensorNoise",
    "GPSNoise",
    "CommNoise",
    "NOISE_PRESETS",
]
