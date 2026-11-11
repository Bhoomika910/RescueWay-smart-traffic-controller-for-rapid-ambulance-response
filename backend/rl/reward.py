# rl/reward.py
"""
Reward function for RescueWay traffic signal control.

See RewardWeights below for the full set of tunable weights — previously
these were bare numeric literals scattered through five functions with no
way to systematically ask "how sensitive is the reward function's design
to these specific values?" (design-review finding, category 8 "Reward
design": "weights are all hand-tuned magic numbers... with no principled
derivation, sensitivity analysis"). Extracting them into one dataclass
lets evaluation/reward_sensitivity.py perturb each weight independently
and check whether the reward function's qualitative behavior (which
action scores higher in representative scenarios) is robust to reasonable
disagreement about the exact values — see that script and
ARCHITECTURE.md's reward-sensitivity section for the actual analysis.

This refactor is behavior-preserving: DEFAULT_WEIGHTS' values are
identical to the previous hardcoded literals, and every existing call
site (dozens, across training/evaluation/live-controller code) calls
compute_reward()/get_reward_breakdown() without a `weights=` argument, so
it keeps using DEFAULT_WEIGHTS and produces bit-identical output to
before — verified via tests/test_reward_invariants.py and
test_reward.py, both re-run unchanged after this refactor.
"""
from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class RewardWeights:
    """
    Every weight compute_reward()'s components use. Defaults match the
    original hardcoded values exactly.

    Actions 4 (Extend) and 5 (Normal cycle) both leave the currently-green
    zone (prev_signal) unchanged — identically to explicitly re-selecting
    that same zone via action 0-3. Congestion/fairness/throughput/switching
    used to score them as a flat 0.0 regardless of how good or bad that
    choice actually was, which made them a risk-free action: re-picking the
    current zone explicitly could score anywhere from -13.5 to +19, but
    "extending" the exact same outcome always scored exactly 0. That made
    Extend/Normal an attractive, zero-variance fallback for a
    not-yet-confident network relative to genuinely risky zone choices,
    and rewarded the agent for describing "stay" as Extend instead of a
    same-zone pick whenever staying was actually a bad idea. Scoring them
    identically to an explicit same-zone re-selection (via chosen_zone =
    prev_signal in each component) closes that loophole: the reward now
    reflects the real traffic outcome regardless of which of the three
    equivalent action encodings produced it.
    """

    # ─── Emergency priority (highest weight) ──────────────
    emergency_correct_zone: float = 25.0  # correct zone gets green
    emergency_extend_correct: float = 10.0  # extending green that already serves ambulance
    emergency_extend_wrong: float = -8.0  # extending wrong green while ambulance waits
    emergency_normal_cycle: float = -15.0  # normal cycle ignores emergency
    emergency_wrong_zone: float = -25.0  # wrong zone gets green, ambulance blocked
    emergency_no_stop_bonus: float = 40.0  # passed through, negligible/no stop
    emergency_delay_penalty_per_second: float = 4.0  # scales with predicted/actual delay

    # ─── Congestion management ─────────────────────────────
    congestion_serve_most_congested: float = 8.0
    congestion_proportional_max: float = 5.0
    congestion_serve_empty_penalty: float = -8.0
    congestion_ignore_congested_penalty: float = -4.0
    congestion_high_total_penalty: float = -3.0
    congestion_low_total_bonus: float = 2.0

    # ─── Fairness between zones ─────────────────────────────
    fairness_balanced_bonus: float = 2.0
    fairness_serve_starved_bonus: float = 3.0
    fairness_ignore_starved_penalty: float = -2.0

    # ─── Throughput efficiency ──────────────────────────────
    throughput_per_vehicle: float = 0.5

    # ─── Switching cost ──────────────────────────────────────
    switching_penalty: float = -1.5
    switching_maintain_bonus: float = 0.5

    def perturbed(self, field_name: str, factor: float) -> "RewardWeights":
        """Return a copy with one named weight scaled by `factor` (e.g.
        1.5 for +50%), all others unchanged — the one-at-a-time
        perturbation evaluation/reward_sensitivity.py sweeps over.
        Mirrors evaluation/comparator.py's weight_sensitivity_sweep()
        methodology for the composite benchmark score, applied here to
        the reward function's own weights instead."""
        return RewardWeights(**{**self.as_dict(), field_name: getattr(self, field_name) * factor})

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


DEFAULT_WEIGHTS = RewardWeights()

# Ceiling on the seconds-of-delay fed into the emergency delay penalty.
# The raw ambulance_delay (SUMO: real per-vehicle waiting time; simple
# simulator: estimate_clearance() deficit) is occasionally much larger
# than the "normal" range the -4.0/s penalty was tuned for (a few to
# ~20s) — e.g. a vehicle stuck behind gridlock can accumulate 60-100+
# seconds of waiting time in SUMO. Left unbounded, one such step produces
# a reward an order of magnitude larger than every other term combined
# (observed: single-episode totals of -6000 to -13000 against a normal
# range of roughly -800 to +1300), which corrupts the TD target for
# whatever state/action produced it and destabilizes the whole Q-function
# (evidenced by training loss climbing steadily rather than converging).
# Capping the input keeps the penalty's ordering/gradient intact over the
# realistic range while removing the catastrophic tail.
MAX_DELAY_FOR_PENALTY = 20.0

# Defensive backstop: clip the total per-step reward to this range. With
# MAX_DELAY_FOR_PENALTY in place this should rarely bind, but it protects
# against any other latent unbounded term (or a future one) blowing up
# the TD target the same way. Range covers the legitimate min/max of all
# components combined (~-120 to ~+90) with headroom.
REWARD_CLIP = 150.0


def compute_reward(
    action,
    amb_zone,
    prev_signal,
    traffic,
    queue_lengths=None,
    system_metrics=None,
    ambulance_delay=None,
    weights: RewardWeights = DEFAULT_WEIGHTS,
):
    """
    Rich reward function for traffic signal control.

    Balances multiple objectives, in priority order:
    1. Ambulance delay (highest weight, when known — see below)
    2. Emergency zone-correctness (dense per-step shaping signal)
    3. Congestion management
    4. Fairness between zones
    5. Throughput efficiency
    6. Switching cost

    Args:
        action:         0-3 = green zone 1-4, 4 = extend, 5 = normal
        amb_zone:       1-4 where ambulance is (0 = no ambulance)
        prev_signal:    1-4 which zone was green before
        traffic:        "low" / "medium" / "high" (fallback)
        queue_lengths:  {1: 5, 2: 12, 3: 3, 4: 8} from simulator
        system_metrics: dict from simulator.get_system_metrics()
        ambulance_delay: predicted or measured seconds the ambulance will
            be/was delayed (0 = passes through without stopping). None
            (the default) omits this term entirely, so any caller that
            doesn't supply it gets the original, unchanged reward — see
            rl/ambulance_delay.py for how to compute it (the simple
            simulator/benchmark path estimates it via
            estimate_clearance(); SUMO measures it directly via
            sumo_adapter.get_ambulance_waiting_time()).
        weights: RewardWeights to use — defaults to DEFAULT_WEIGHTS
            (the original hardcoded values). Only
            evaluation/reward_sensitivity.py passes a perturbed instance;
            every other caller in the codebase is unaffected by this
            parameter existing.
    """
    reward = 0.0

    # Use default queue lengths if simulator data not available
    if queue_lengths is None:
        base = {"low": 3, "medium": 8, "high": 15}.get(traffic, 8)
        queue_lengths = {1: base, 2: base, 3: base, 4: base}

    # ─── 1. EMERGENCY PRIORITY (weight: highest) ────────────────
    emergency_reward = _emergency_reward(action, amb_zone, prev_signal, ambulance_delay, weights)
    reward += emergency_reward

    # ─── 2. CONGESTION MANAGEMENT ────────────────────────────────
    congestion_reward = _congestion_reward(action, queue_lengths, prev_signal, weights)
    reward += congestion_reward

    # ─── 3. FAIRNESS BETWEEN ZONES ───────────────────────────────
    fairness_reward = _fairness_reward(action, queue_lengths, prev_signal, weights)
    reward += fairness_reward

    # ─── 4. THROUGHPUT EFFICIENCY ────────────────────────────────
    throughput_reward = _throughput_reward(action, queue_lengths, prev_signal, weights)
    reward += throughput_reward

    # ─── 5. SWITCHING COST ───────────────────────────────────────
    switching_penalty = _switching_penalty(action, prev_signal, weights)
    reward += switching_penalty

    reward = max(-REWARD_CLIP, min(REWARD_CLIP, reward))

    return round(reward, 2)


def _emergency_reward(action, amb_zone, prev_signal, ambulance_delay=None, weights=DEFAULT_WEIGHTS):
    """
    Emergency vehicle priority.
    This is the highest-weight component.

    The base zone-correctness term (unchanged from before ambulance_delay
    existed) rewards targeting the ambulance's zone this step — a dense,
    always-available shaping signal. When a delay estimate/measurement IS
    available, it is added ON TOP and dominates: the actual outcome
    (did the ambulance stop, and for how long) matters more than whether
    a single step's action nominally targeted the right zone. This
    matters specifically for the reactive-vs-proactive distinction: an
    action that only targets the ambulance's zone once it's already too
    late to clear the backed-up queue in time should NOT be rewarded as
    if the outcome were good.

    NOTE on action==5 ("normal cycle") always scoring
    weights.emergency_normal_cycle regardless of prev_signal, unlike
    action==4 which checks prev_signal==amb_zone: this asymmetry is
    real and was found (not introduced) while adding
    tests/test_reward_invariants.py — see that test's docstring for the
    two plausible readings (intentional passivity penalty vs. a second
    instance of the action-4-only bug already fixed elsewhere) and why it
    was deliberately left unchanged rather than "fixed" on an inference.
    """
    if amb_zone == 0:
        return 0.0  # no ambulance, no emergency reward

    target_action = amb_zone - 1  # action 0 = zone 1

    if action == target_action:
        reward = weights.emergency_correct_zone
    elif action == 4 and prev_signal == amb_zone:
        reward = weights.emergency_extend_correct
    elif action == 4 and prev_signal != amb_zone:
        reward = weights.emergency_extend_wrong
    elif action == 5:
        reward = weights.emergency_normal_cycle
    else:
        reward = weights.emergency_wrong_zone

    if ambulance_delay is not None:
        if ambulance_delay <= 0.5:
            reward += weights.emergency_no_stop_bonus
        else:
            capped_delay = min(ambulance_delay, MAX_DELAY_FOR_PENALTY)
            reward -= weights.emergency_delay_penalty_per_second * capped_delay

    return reward


def _congestion_reward(action, queue_lengths, prev_signal, weights=DEFAULT_WEIGHTS):
    """
    Args:
        prev_signal: the zone currently green. Used as the effective
            "chosen zone" for action>=4 (Extend/Normal) — see
            RewardWeights' docstring on why these actions are scored
            identically to explicitly re-selecting the current zone
            rather than skipped to 0.0.
    """
    chosen_zone = action + 1 if action < 4 else prev_signal
    chosen_queue = queue_lengths.get(chosen_zone, 0)
    max_queue = max(queue_lengths.values()) if queue_lengths else 0
    total_queue = sum(queue_lengths.values())
    max_zone = max(queue_lengths, key=queue_lengths.get) if queue_lengths else 1

    reward = 0.0

    # Strong reward for serving the most congested zone
    if chosen_zone == max_zone and max_queue > 3:
        reward += weights.congestion_serve_most_congested

    # Proportional reward based on queue being served
    if max_queue > 0:
        reward += (chosen_queue / max_queue) * weights.congestion_proportional_max

    # Strong penalty for serving empty zone when others are congested
    if chosen_queue == 0 and total_queue > 10:
        reward += weights.congestion_serve_empty_penalty

    # Penalty for ignoring the most congested zone
    if chosen_zone != max_zone and max_queue > 10:
        reward += weights.congestion_ignore_congested_penalty

    # Overall congestion management
    if total_queue > 40:
        reward += weights.congestion_high_total_penalty
    elif total_queue < 10:
        reward += weights.congestion_low_total_bonus

    return reward


def _fairness_reward(action, queue_lengths, prev_signal, weights=DEFAULT_WEIGHTS):
    """
    Reward fairness between zones.
    Penalize starvation (one zone waiting too long).
    """
    values = list(queue_lengths.values())
    if not values or max(values) == 0:
        return 0.0

    std_dev = np.std(values)
    mean = np.mean(values)

    if mean == 0:
        return 0.0

    cv = std_dev / mean  # coefficient of variation

    # Reward balanced queues
    if cv < 0.3:
        return weights.fairness_balanced_bonus  # very balanced
    elif cv < 0.6:
        return 0.0  # moderate
    else:
        # Severe imbalance — reward serving the starved zone
        chosen_zone = action + 1 if action < 4 else prev_signal
        max_zone = max(queue_lengths, key=queue_lengths.get)
        if chosen_zone == max_zone:
            return weights.fairness_serve_starved_bonus
        return weights.fairness_ignore_starved_penalty


def _throughput_reward(action, queue_lengths, prev_signal, weights=DEFAULT_WEIGHTS):
    """
    Reward high throughput.
    Green signal on a zone with vehicles = vehicles cleared.
    """
    chosen_zone = action + 1 if action < 4 else prev_signal
    chosen_queue = queue_lengths.get(chosen_zone, 0)

    # More vehicles cleared = higher reward
    clearable = min(chosen_queue, 5)  # max 5 vehicles per green
    return clearable * weights.throughput_per_vehicle


def _switching_penalty(action, prev_signal, weights=DEFAULT_WEIGHTS):
    """
    Small penalty for switching signals unnecessarily.
    Frequent switching reduces throughput and confuses drivers.
    """
    new_signal = action + 1 if action < 4 else prev_signal
    if new_signal != prev_signal:
        return weights.switching_penalty
    return weights.switching_maintain_bonus  # small bonus for maintaining current green


def get_reward_breakdown(
    action,
    amb_zone,
    prev_signal,
    traffic,
    queue_lengths=None,
    ambulance_delay=None,
    weights: RewardWeights = DEFAULT_WEIGHTS,
):
    """
    Return detailed breakdown of reward components.
    Useful for debugging and dashboard display.
    """
    if queue_lengths is None:
        base = {"low": 3, "medium": 8, "high": 15}.get(traffic, 8)
        queue_lengths = {1: base, 2: base, 3: base, 4: base}

    breakdown = {
        "emergency": round(
            _emergency_reward(action, amb_zone, prev_signal, ambulance_delay, weights), 2
        ),
        "congestion": round(_congestion_reward(action, queue_lengths, prev_signal, weights), 2),
        "fairness": round(_fairness_reward(action, queue_lengths, prev_signal, weights), 2),
        "throughput": round(_throughput_reward(action, queue_lengths, prev_signal, weights), 2),
        "switching": round(_switching_penalty(action, prev_signal, weights), 2),
    }
    if ambulance_delay is not None:
        breakdown["ambulance_delay_seconds"] = round(ambulance_delay, 2)
    breakdown["total"] = round(
        sum(v for k, v in breakdown.items() if k != "ambulance_delay_seconds"), 2
    )

    return breakdown
