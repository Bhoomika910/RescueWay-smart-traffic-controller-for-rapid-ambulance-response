# rl/perception_aware_shield.py
"""
Perception-Aware Safety Shield (PAS) for RescueWay.

WHAT THIS SOLVES (base-paper limitation + our own Section V-C finding):
Reward-weighted preemption methods (DNLight, IoT-J 2025; CCPVLight,
MT-ITS 2025) make emergency priority a statistical preference and assume
perfect vehicle localisation. Our Section V-C showed the naive hard
shield inherits the same assumption: under zone-misclassification
probability p = 0.317 (dense urban) shielded accuracy falls to 68.2%,
and at p = 0.867 (deep canyon) to 13.7% — identical to the unshielded
policy, because the shield deterministically trusts a single noisy
report.

THE FIX: stop shielding on a point estimate; shield on a *belief*.
GPS fixes arrive every decision tick and their errors are (per our own
noise model, noise/noise_models.GPSNoise.apply_zone) independent across
ticks and adjacency-biased. Independent noisy observations of a
slowly-moving true state is the textbook setting for a discrete Bayes
filter: fuse the fix stream with a motion model (an inbound ambulance
stays in its band or advances one band per tick — it never teleports)
and act on the MAP zone with a quantified posterior confidence.

Guarantee restatement (the honest version Section V-C asked for):
  naive shield : "the reported approach is always served"
  PAS          : "the MAP approach under the full fix history is always
                  served, and the controller KNOWS its own confidence" —
                  the confidence gate turns silent wrong-approach service
                  into a flagged degraded mode a deployment can act on
                  (e.g. all-red + audible fallback below tau).

Design notes:
  * Sensor model mirrors GPSNoise.apply_zone EXACTLY (same exponential
    adjacency decay) so the filter is well-specified w.r.t. the injected
    noise; a real deployment would fit this matrix from logged fixes.
  * Motion model: P(stay) = 1 - advance_rate, P(advance one band inward)
    = advance_rate. Zones are ordered distance bands (4 far ... 1 close),
    so "advance" means zone - 1. Mild and physically true; the filter is
    robust to the exact rate (sweep in the eval script).
  * assumed_misclass_prob is the filter's belief about sensor quality.
    It need not equal the true p — the eval sweeps a mismatched filter
    to show graceful degradation (no oracle knowledge claimed).
  * Latency compensation (optional): if enacted commands lag by L ticks,
    shield the belief PREDICTED L steps ahead (apply the motion model L
    extra times before taking the MAP) — addresses the 8–13 point drop
    at 3–5 s latency found in Section V-F.

This module is dependency-free w.r.t. the live stack: it composes with
rl.safety_shield.shield_emergency_action (unchanged) rather than
replacing it, so every existing result remains reproducible.
"""
from dataclasses import dataclass, field

import numpy as np

try:
    from rl.emergency_scoring import is_emergency_action_correct
    from rl.safety_shield import shield_emergency_action
except ImportError:  # standalone validation outside the repo
    def is_emergency_action_correct(action, emergency_zone, served_zone):
        if action < 4:
            return (action + 1) == emergency_zone
        return served_zone == emergency_zone

    def shield_emergency_action(action, emergency_zone, current_zone):
        if not emergency_zone or emergency_zone <= 0:
            return action, False
        if is_emergency_action_correct(action, emergency_zone, current_zone):
            return action, False
        return emergency_zone - 1, True

N_ZONES = 4
ADJACENCY_DECAY = 1.0  # MUST mirror noise.noise_models.GPSNoise.ADJACENCY_DECAY


def sensor_likelihood_matrix(misclass_prob: float, n_zones: int = N_ZONES) -> np.ndarray:
    """
    L[true-1, reported-1] = P(reported | true), replicating
    GPSNoise.apply_zone: with prob (1-p) report the truth; with prob p
    report one of the other zones with weights exp(-|z - true|)
    normalised over the wrong zones.
    """
    L = np.zeros((n_zones, n_zones), dtype=float)
    for t in range(1, n_zones + 1):
        others = [z for z in range(1, n_zones + 1) if z != t]
        w = np.array([np.exp(-ADJACENCY_DECAY * abs(z - t)) for z in others])
        w = w / w.sum()
        L[t - 1, t - 1] = 1.0 - misclass_prob
        for z, wz in zip(others, w):
            L[t - 1, z - 1] = misclass_prob * wz
    return L


def motion_matrix(advance_rate: float, n_zones: int = N_ZONES) -> np.ndarray:
    """
    M[prev-1, next-1] = P(next | prev) for an inbound ambulance:
    stay with prob (1 - advance_rate), move one band inward (zone-1)
    with prob advance_rate. Zone 1 is absorbing (it is the stop line).
    """
    M = np.eye(n_zones) * (1.0 - advance_rate)
    for z in range(2, n_zones + 1):
        M[z - 1, z - 2] = advance_rate
    M[0, 0] = 1.0  # absorbing at the closest band
    return M


@dataclass
class ZoneBeliefFilter:
    """Discrete Bayes filter over the ambulance's true zone."""

    assumed_misclass_prob: float
    advance_rate: float = 0.15
    n_zones: int = N_ZONES
    belief: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self._L = sensor_likelihood_matrix(self.assumed_misclass_prob, self.n_zones)
        self._M = motion_matrix(self.advance_rate, self.n_zones)
        self.reset()

    def reset(self, prior_zone: int | None = None):
        """New emergency: uniform prior, or peaked on the first report."""
        if prior_zone:
            self.belief = self._L[:, prior_zone - 1].copy()
        else:
            self.belief = np.full(self.n_zones, 1.0 / self.n_zones)
        self.belief /= self.belief.sum()

    def update(self, reported_zone: int) -> np.ndarray:
        """Predict (motion) then correct (sensor) with one fix."""
        self.belief = self._M.T @ self.belief
        self.belief *= self._L[:, reported_zone - 1]
        s = self.belief.sum()
        if s <= 0:  # numerically impossible report stream — reseed
            self.reset(reported_zone)
        else:
            self.belief /= s
        return self.belief

    def predicted(self, steps_ahead: int = 0) -> np.ndarray:
        b = self.belief.copy()
        for _ in range(steps_ahead):
            b = self._M.T @ b
        return b

    def map_zone(self, steps_ahead: int = 0) -> tuple[int, float]:
        b = self.predicted(steps_ahead)
        z = int(np.argmax(b)) + 1
        return z, float(b[z - 1])


class PerceptionAwareShield:
    """
    Drop-in wrapper: same call signature shape as shield_emergency_action
    plus the raw reported zone. Keeps one filter per emergency episode.
    """

    def __init__(
        self,
        assumed_misclass_prob: float,
        advance_rate: float = 0.15,
        confidence_tau: float = 0.6,
        latency_ticks: int = 0,
    ):
        self.filter = ZoneBeliefFilter(assumed_misclass_prob, advance_rate)
        self.tau = confidence_tau
        self.latency_ticks = latency_ticks
        self._active = False

    def notify_emergency(self, active: bool, first_report: int | None = None):
        if active and not self._active:
            self.filter.reset(first_report)
        self._active = active

    def shield(self, action: int, reported_zone: int, current_zone: int):
        """
        Returns (enacted_action, overridden, info) where info carries the
        MAP zone, posterior confidence, and whether the guarantee is in
        its degraded (low-confidence) mode this tick.
        """
        if not reported_zone or reported_zone <= 0:
            return action, False, {"map_zone": 0, "confidence": 0.0, "degraded": False}

        self.filter.update(reported_zone)
        map_zone, conf = self.filter.map_zone(steps_ahead=self.latency_ticks)
        enacted, overridden = shield_emergency_action(action, map_zone, current_zone)
        return enacted, overridden, {
            "map_zone": map_zone,
            "confidence": conf,
            "degraded": conf < self.tau,
        }


def score_tick(action: int, true_zone: int, current_zone: int) -> bool:
    """Correctness against the TRUE zone (what the ambulance experiences)."""
    return is_emergency_action_correct(action, true_zone, current_zone)
