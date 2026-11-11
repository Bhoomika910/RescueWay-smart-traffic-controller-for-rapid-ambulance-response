# evaluation/robustness_runner.py
"""
RescueWay Robustness Experiment Runner.

Systematically evaluates all controllers under all noise conditions
across five studies. Produces structured results for statistical analysis.

Usage:
    python -m evaluation.robustness_runner --study 1
    python -m evaluation.robustness_runner --study 2 --seeds 30
    python -m evaluation.robustness_runner --study all
    python -m evaluation.robustness_runner --study 3 --preset urban_dense

Studies:
    1: Baseline calibration (clean conditions, all controllers)
    2: Single-factor sensitivity (one noise at a time)
    3: Combined stress test (preset scenarios)
    4: Failure mode analysis (extreme noise)
    5: DQN vs Q-Learning head-to-head under noise
"""
import argparse
import json
import os
import random as stdlib_random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BENCHMARK_EMERGENCY_PROBABILITY, BENCHMARK_SEED, ENV_TAG_SIMPLIFIED
from controllers.dqn_controller import DQNController
from controllers.fixed_timer import FixedTimerController
from controllers.random_controller import RandomController
from controllers.rl_controller import RLController
from controllers.rule_based import RuleBasedController
from gps.zone_model import ZONE_CONFIG
from noise import NOISE_PRESETS, NoiseApplicator, NoiseConfig
from rl.emergency_scoring import is_emergency_action_correct
from rl.environment import TrafficEnv
from rl.reward import compute_reward
from rl.state_features import StateFeatureExtractor

# ─── Output ──────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "robustness")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Controller Registry ────────────────────────────


def _make_clean_dqn():
    """Create DQN controller with clean noise config to prevent double noise."""
    from rl.dqn_agent import DQNAgent

    agent = DQNAgent(epsilon=0.0, inference_mode=True)
    # Override the feature extractor with a CLEAN noise config
    # so the runner's explicit noise is the ONLY noise applied
    agent.feature_extractor = StateFeatureExtractor(noise_config=NoiseConfig())
    agent.load()
    return DQNController(agent=agent)


def _make_clean_ql():
    """Create Q-Learning controller with clean noise config."""
    from rl.q_agent import QAgent
    from rl.state_encoder import StateEncoder

    agent = QAgent(epsilon=0.0)
    agent.encoder = StateEncoder(noise_config=NoiseConfig())
    agent.load()
    return RLController(agent=agent)


def get_all_controllers():
    """All 5 controllers for full comparison. Clean noise configs."""
    return {
        "dqn": _make_clean_dqn(),
        "q_learning": _make_clean_ql(),
        "rule_based": RuleBasedController(),
        "fixed": FixedTimerController(),
        "random": RandomController(),
    }


def get_rl_controllers():
    """Only RL controllers for Study 5 head-to-head."""
    return {
        "dqn": _make_clean_dqn(),
        "q_learning": _make_clean_ql(),
    }


# ─── Episode Runner with Noise ───────────────────────


def run_episode(
    controller, noise_config, seed, max_steps=30, emergency_prob=BENCHMARK_EMERGENCY_PROBABILITY
):
    """
    Run one evaluation episode with noise injection.

    Returns dict of metrics for this episode.
    """
    # Seed for reproducibility
    np.random.seed(seed)
    stdlib_random.seed(seed)

    # Create environment
    env = TrafficEnv()
    env.reset()

    # Create noise applicator
    noise = NoiseApplicator(noise_config)
    noise.reset()

    # Metrics
    episode_reward = 0
    total_queue = 0
    emergency_correct = 0
    emergency_total = 0
    emergency_response_steps = []
    signal_switches = 0
    prev_action = -1
    false_positives_triggered = 0
    false_negatives_triggered = 0
    packet_drops = 0
    zone_misclassifications = 0
    latency_total = 0.0
    latency_count = 0

    # Emergency state
    emergency_active = False
    emergency_zone = 0
    true_emergency_zone = 0
    emergency_timer = 0
    emergency_wait_steps = 0
    emergency_eta = 0.0
    pending_emergency = None  # Delayed emergency waiting for latency

    for step in range(max_steps):
        metrics = env.simulator.get_metrics()
        queue_lengths = {z: metrics[z].queue_length for z in [1, 2, 3, 4]}

        # ─── Emergency Generation ─────────────────────
        if not emergency_active:
            # Check if a delayed emergency should now activate
            if pending_emergency is not None:
                pending_emergency["delay_remaining"] -= 1
                if pending_emergency["delay_remaining"] <= 0:
                    emergency_active = True
                    emergency_zone = pending_emergency["zone"]
                    true_emergency_zone = pending_emergency["true_zone"]
                    emergency_timer = pending_emergency["timer"]
                    emergency_wait_steps = pending_emergency["delay_steps"]
                    emergency_eta = pending_emergency["eta"]
                    pending_emergency = None

            elif stdlib_random.random() < emergency_prob:
                true_emergency_zone = stdlib_random.choice([1, 2, 3, 4])
                # Emergency duration is seeded from the same per-zone ETA
                # the live system/flagship benchmark use (gps/zone_model.
                # ZONE_CONFIG), not an arbitrary randint(3,8) window — that
                # arbitrary window tested a systematically narrower/faster
                # emergency-timing distribution than comparator.py's
                # deliberate 5/10/18/30s mix of far/close trips.
                emergency_timer = float(ZONE_CONFIG[true_emergency_zone]["eta_seconds"])
                emergency_wait_steps = 0

                # GPS false negative: ambulance exists but not detected
                if noise.gps.check_false_negative(ambulance_present=True):
                    false_negatives_triggered += 1
                    continue

                # Packet loss: request dropped
                if noise.comm.check_packet_loss():
                    packet_drops += 1
                    continue

                # Communication latency — delay by N steps
                latency = noise.comm.get_latency()
                delay_steps = 0
                if latency > 0:
                    latency_total += latency
                    latency_count += 1
                    delay_steps = max(1, int(latency / 0.5))  # 0.5s per step

                # GPS zone misclassification
                reported_zone = noise.gps.apply_zone(true_emergency_zone)
                if reported_zone != true_emergency_zone:
                    zone_misclassifications += 1

                # GPS ETA error — true_eta comes from the same ZONE_CONFIG
                # table as emergency_timer above (single source of truth,
                # no more hardcoded duplicate of the zone->ETA mapping).
                true_eta = ZONE_CONFIG[true_emergency_zone]["eta_seconds"]
                noisy_eta = noise.gps.apply_eta(float(true_eta))

                if delay_steps > 0:
                    # Queue the emergency — controller doesn't see it yet
                    pending_emergency = {
                        "zone": reported_zone,
                        "true_zone": true_emergency_zone,
                        "timer": emergency_timer,
                        "delay_remaining": delay_steps,
                        "delay_steps": delay_steps,
                        "eta": noisy_eta,
                    }
                else:
                    emergency_active = True
                    emergency_zone = reported_zone
                    emergency_eta = noisy_eta

            else:
                # GPS false positive: no ambulance but one is "detected"
                fp, phantom_zone = noise.gps.check_false_positive()
                if fp:
                    false_positives_triggered += 1
                    emergency_active = True
                    emergency_zone = phantom_zone
                    true_emergency_zone = 0  # No real ambulance
                    emergency_timer = float(ZONE_CONFIG[phantom_zone]["eta_seconds"])
                    # No real ambulance exists, so there's no ground-truth
                    # ETA to draw from — reuse the phantom zone's nominal
                    # ETA so the controller still receives a coherent,
                    # in-range value instead of a stale eta left over from
                    # a previous (unrelated) emergency.
                    emergency_eta = emergency_timer
                    emergency_wait_steps = 0

        else:
            emergency_timer -= 1
            emergency_wait_steps += 1
            emergency_eta = max(0.0, emergency_eta - 1.0)
            if emergency_timer <= 0:
                emergency_active = False
                emergency_zone = 0
                true_emergency_zone = 0
                emergency_eta = 0.0

        # ─── Apply Sensor Noise to Queue Lengths ──────
        noisy_queues = noise.sensor.apply(queue_lengths)

        # Distance is a static per-zone property (gps/zone_model.ZONE_CONFIG),
        # keyed by the *reported* zone since that's what the controller
        # actually perceives (same as ambulance_zone=emergency_zone below).
        distance = (
            float(ZONE_CONFIG[emergency_zone]["avg_distance"])
            if emergency_active and emergency_zone in ZONE_CONFIG
            else 0.0
        )

        # ─── Controller Decision ──────────────────────
        # emergency_eta/distance are forwarded here so DQN/Q-Learning are
        # evaluated on the same state distribution they were trained on
        # (see BaseController.choose_action's emergency_eta/distance kwargs
        # and ARCHITECTURE.md's Bug #1). Previously this call site only
        # passed the first 4 args, silently defaulting both to 0.0 for
        # every noise condition tested here — the exact bug documented as
        # fixed in evaluation/comparator.py, reintroduced by never having
        # been propagated to this file.
        action = controller.choose_action(
            active_signal=env.active_signal,
            queue_lengths=noisy_queues,
            emergency_active=emergency_active,
            ambulance_zone=emergency_zone,
            emergency_eta=emergency_eta if emergency_active else 0.0,
            distance=distance,
        )

        # ─── Compute Reward (using TRUE data, not noisy) ─
        reward = compute_reward(
            action=action,
            amb_zone=true_emergency_zone if emergency_active else 0,
            prev_signal=env.active_signal,
            traffic=env.simulator.traffic_pattern,
            queue_lengths=queue_lengths,  # TRUE queues for reward
        )

        # ─── Apply Action ────────────────────────────
        env.active_signal = action + 1 if action < 4 else env.active_signal
        env.simulator.step(
            active_signal=env.active_signal,
            is_green=True,
            current_time=step,
        )

        # ─── Track Metrics ───────────────────────────
        episode_reward += reward
        total_queue += sum(queue_lengths.values())

        if action != prev_action and action < 4:
            signal_switches += 1
        prev_action = action

        # Emergency accuracy (compare against TRUE zone) via the shared
        # rl/emergency_scoring helper — this call site's own version of
        # this check used to be the one other sites were told to match
        # (see rl/emergency_scoring.py's module docstring for the fuller
        # history of why a shared function replaced nine local copies).
        if emergency_active and true_emergency_zone > 0:
            emergency_total += 1
            if is_emergency_action_correct(action, true_emergency_zone, env.active_signal):
                emergency_correct += 1
                emergency_response_steps.append(emergency_wait_steps)

    # ─── Compile Results ─────────────────────────────
    throughput = env.simulator.get_system_metrics().get("total_vehicles_cleared", 0)

    return {
        "reward": round(float(episode_reward), 2),
        "avg_queue": round(total_queue / max(max_steps, 1), 2),
        "throughput": throughput,
        "switches": signal_switches,
        "emergency_accuracy": round(
            (emergency_correct / emergency_total * 100) if emergency_total > 0 else 100.0, 1
        ),
        "emergency_correct": emergency_correct,
        "emergency_total": emergency_total,
        "avg_emergency_response": round(
            float(np.mean(emergency_response_steps)) if emergency_response_steps else 0.0, 2
        ),
        "false_positives": false_positives_triggered,
        "false_negatives": false_negatives_triggered,
        "packet_drops": packet_drops,
        "zone_misclassifications": zone_misclassifications,
        "avg_latency": round(latency_total / max(latency_count, 1), 3),
    }


# ─── Multi-Seed Evaluation ──────────────────────────


def evaluate_condition(
    controller,
    noise_config,
    n_seeds=30,
    base_seed=None,
    max_steps=30,
    emergency_prob=BENCHMARK_EMERGENCY_PROBABILITY,
):
    """
    Run a controller under one noise condition across multiple seeds.

    Returns aggregated metrics with mean, std, and 95% CI.
    """
    if base_seed is None:
        base_seed = BENCHMARK_SEED

    results = []
    for i in range(n_seeds):
        seed = base_seed + i
        result = run_episode(
            controller,
            noise_config,
            seed,
            max_steps=max_steps,
            emergency_prob=emergency_prob,
        )
        results.append(result)

    # Aggregate with statistics
    def aggregate(key):
        values = [r[key] for r in results]
        mean = float(np.mean(values))
        std = float(np.std(values))
        ci_95 = 1.96 * std / np.sqrt(len(values))
        return {
            "mean": round(mean, 3),
            "std": round(std, 3),
            "ci_95": round(ci_95, 3),
            "min": round(float(np.min(values)), 3),
            "max": round(float(np.max(values)), 3),
        }

    return {
        "reward": aggregate("reward"),
        "avg_queue": aggregate("avg_queue"),
        "throughput": aggregate("throughput"),
        "switches": aggregate("switches"),
        "emergency_accuracy": aggregate("emergency_accuracy"),
        "avg_emergency_response": aggregate("avg_emergency_response"),
        "false_positives": aggregate("false_positives"),
        "false_negatives": aggregate("false_negatives"),
        "packet_drops": aggregate("packet_drops"),
        "zone_misclassifications": aggregate("zone_misclassifications"),
        "avg_latency": aggregate("avg_latency"),
        "n_seeds": n_seeds,
        # Raw per-seed values, in seed order (base_seed, base_seed+1, ...).
        # Required for paired significance testing — every controller runs
        # the SAME seed sequence, so raw[i] across controllers is a matched pair.
        "raw": {
            "reward": [r["reward"] for r in results],
            "emergency_accuracy": [r["emergency_accuracy"] for r in results],
            "avg_queue": [r["avg_queue"] for r in results],
            "throughput": [r["throughput"] for r in results],
        },
    }


# ══════════════════════════════════════════════════════
# STUDY 1: BASELINE CALIBRATION
# ══════════════════════════════════════════════════════


def run_study_1(n_seeds=30):
    """
    Study 1: Baseline calibration under clean conditions.

    All controllers, no noise, establishes reference performance.
    """
    print("=" * 70)
    print("STUDY 1: BASELINE CALIBRATION (Clean Conditions)")
    print(f"Seeds: {n_seeds}")
    print("=" * 70)

    controllers = get_all_controllers()
    clean = NoiseConfig()
    results = {}

    for name, ctrl in controllers.items():
        print(f"  Evaluating: {ctrl.name}...", end="", flush=True)
        result = evaluate_condition(ctrl, clean, n_seeds=n_seeds)
        results[name] = result
        print(
            f" reward={result['reward']['mean']:+.1f} ± {result['reward']['ci_95']:.1f}, "
            f"emerg={result['emergency_accuracy']['mean']:.1f}%"
        )

    return {"study": 1, "description": "Baseline calibration", "results": results}


# ══════════════════════════════════════════════════════
# STUDY 2: SINGLE-FACTOR SENSITIVITY
# ══════════════════════════════════════════════════════

# Define sweep ranges for each noise factor
SENSITIVITY_SWEEPS = {
    "sensor_gaussian_std": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "sensor_dropout_prob": [0.0, 0.02, 0.05, 0.10, 0.15, 0.20],
    "sensor_bias_pct": [0.0, 0.05, 0.10, 0.15, 0.20, -0.10, -0.20],
    "gps_misclass_prob": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    # Note: gps_eta_error_seconds removed — ETA is not part of the benchmark
    # controller decision interface. ETA affects adaptive green duration in the
    # live system but not the step-based benchmark. Documented as a limitation.
    "gps_false_positive_prob": [0.0, 0.02, 0.05, 0.10, 0.15, 0.20],
    "gps_false_negative_prob": [0.0, 0.02, 0.05, 0.10, 0.15, 0.20],
    "comm_constant_latency": [0.0, 0.5, 1.0, 2.0, 3.0, 5.0],
    "comm_variable_latency_mean": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "comm_burst_prob": [0.0, 0.02, 0.05, 0.10, 0.15, 0.20],
    "comm_packet_loss_prob": [0.0, 0.01, 0.03, 0.05, 0.10, 0.15],
}


def run_study_2(n_seeds=20, factors=None):
    """
    Study 2: Single-factor sensitivity analysis.

    Sweep each noise factor independently. All others at zero.
    """
    print("=" * 70)
    print("STUDY 2: SINGLE-FACTOR SENSITIVITY")
    print(f"Seeds per condition: {n_seeds}")
    print("=" * 70)

    controllers = get_all_controllers()

    if factors is None:
        factors = list(SENSITIVITY_SWEEPS.keys())

    results = {}
    total_conditions = sum(len(SENSITIVITY_SWEEPS[f]) for f in factors)
    completed = 0

    for factor in factors:
        sweep_values = SENSITIVITY_SWEEPS[factor]
        print(f"\n  Factor: {factor}")
        results[factor] = {}

        for value in sweep_values:
            # Create config with only this factor active
            config = NoiseConfig(**{factor: value})

            results[factor][str(value)] = {}

            for ctrl_name, ctrl in controllers.items():
                ctrl.reset_stats()
                result = evaluate_condition(ctrl, config, n_seeds=n_seeds)
                results[factor][str(value)][ctrl_name] = result

            completed += 1
            dqn_reward = results[factor][str(value)]["dqn"]["reward"]["mean"]
            rb_reward = results[factor][str(value)]["rule_based"]["reward"]["mean"]

            print(
                f"    {factor}={value:6.2f} | "
                f"DQN: {dqn_reward:+7.1f} | "
                f"Rule: {rb_reward:+7.1f} | "
                f"[{completed}/{total_conditions}]"
            )

    return {
        "study": 2,
        "description": "Single-factor sensitivity",
        "factors": factors,
        "results": results,
    }


# ══════════════════════════════════════════════════════
# STUDY 3: COMBINED STRESS TEST
# ══════════════════════════════════════════════════════


def run_study_3(n_seeds=30, presets=None):
    """
    Study 3: Combined stress test using preset scenarios.
    """
    print("=" * 70)
    print("STUDY 3: COMBINED STRESS TEST")
    print(f"Seeds per condition: {n_seeds}")
    print("=" * 70)

    controllers = get_all_controllers()

    if presets is None:
        presets = list(NOISE_PRESETS.keys())

    results = {}

    for preset_name in presets:
        config = NOISE_PRESETS[preset_name]
        print(f"\n  Scenario: {preset_name}")
        results[preset_name] = {}

        for ctrl_name, ctrl in controllers.items():
            ctrl.reset_stats()
            result = evaluate_condition(ctrl, config, n_seeds=n_seeds)
            results[preset_name][ctrl_name] = result
            print(
                f"    {ctrl.name:20s} | "
                f"reward={result['reward']['mean']:+8.1f} ± {result['reward']['ci_95']:5.1f} | "
                f"emerg={result['emergency_accuracy']['mean']:5.1f}%"
            )

    # Compute performance retention
    if "clean" in results:
        print("\n  Performance Retention (vs clean baseline):")
        for preset_name in presets:
            if preset_name == "clean":
                continue
            for ctrl_name in controllers:
                clean_reward = results["clean"][ctrl_name]["reward"]["mean"]
                noisy_reward = results[preset_name][ctrl_name]["reward"]["mean"]
                if abs(clean_reward) > 0:
                    retention = noisy_reward / clean_reward
                    results[preset_name][ctrl_name]["performance_retention"] = round(retention, 3)
                    if ctrl_name == "dqn":
                        print(f"    {preset_name:15s} DQN: {retention:.1%}")

    return {
        "study": 3,
        "description": "Combined stress test",
        "presets": presets,
        "results": results,
    }


# ══════════════════════════════════════════════════════
# STUDY 4: FAILURE MODE ANALYSIS
# ══════════════════════════════════════════════════════

EXTREME_SWEEPS = {
    "sensor_gaussian_std": [0.30, 0.50, 0.75, 1.00],
    "sensor_dropout_prob": [0.20, 0.40, 0.60, 0.80, 1.00],
    "gps_misclass_prob": [0.30, 0.50, 0.75, 1.00],
    "gps_false_negative_prob": [0.20, 0.40, 0.60, 0.80],
    "comm_constant_latency": [5.0, 10.0, 15.0, 20.0],
    "comm_packet_loss_prob": [0.15, 0.30, 0.50, 0.75, 1.00],
}


def run_study_4(n_seeds=20):
    """
    Study 4: Failure mode analysis.

    Push noise to breaking point. Document when each controller fails.
    """
    print("=" * 70)
    print("STUDY 4: FAILURE MODE ANALYSIS")
    print(f"Seeds per condition: {n_seeds}")
    print("=" * 70)

    controllers = get_all_controllers()
    results = {}

    for factor, values in EXTREME_SWEEPS.items():
        print(f"\n  Factor: {factor} (extreme range)")
        results[factor] = {}

        for value in values:
            config = NoiseConfig(**{factor: value})
            results[factor][str(value)] = {}

            for ctrl_name, ctrl in controllers.items():
                ctrl.reset_stats()
                result = evaluate_condition(ctrl, config, n_seeds=n_seeds)
                results[factor][str(value)][ctrl_name] = result

            dqn_acc = results[factor][str(value)]["dqn"]["emergency_accuracy"]["mean"]
            rb_acc = results[factor][str(value)]["rule_based"]["emergency_accuracy"]["mean"]

            # Detect failure: emergency accuracy below 50%
            dqn_fail = "FAIL" if dqn_acc < 50 else "ok"
            rb_fail = "FAIL" if rb_acc < 50 else "ok"

            print(
                f"    {factor}={value:6.2f} | "
                f"DQN emerg: {dqn_acc:5.1f}% [{dqn_fail}] | "
                f"Rule emerg: {rb_acc:5.1f}% [{rb_fail}]"
            )

    return {
        "study": 4,
        "description": "Failure mode analysis",
        "results": results,
    }


# ══════════════════════════════════════════════════════
# STUDY 5: DQN vs Q-LEARNING HEAD-TO-HEAD
# ══════════════════════════════════════════════════════


def run_study_5(n_seeds=30):
    """
    Study 5: DQN vs Q-Learning under all noise conditions.

    Key research question: Which algorithm is more robust?
    Hypothesis: Q-Learning may be more robust because discretization
    acts as denoising, but DQN has higher peak performance.
    """
    print("=" * 70)
    print("STUDY 5: DQN vs Q-LEARNING HEAD-TO-HEAD")
    print(f"Seeds per condition: {n_seeds}")
    print("=" * 70)

    controllers = get_rl_controllers()
    results = {}

    # Test across all presets
    for preset_name, config in NOISE_PRESETS.items():
        print(f"\n  Scenario: {preset_name}")
        results[preset_name] = {}

        for ctrl_name, ctrl in controllers.items():
            ctrl.reset_stats()
            result = evaluate_condition(ctrl, config, n_seeds=n_seeds)
            results[preset_name][ctrl_name] = result
            print(
                f"    {ctrl.name:20s} | "
                f"reward={result['reward']['mean']:+8.1f} ± {result['reward']['ci_95']:5.1f} | "
                f"emerg={result['emergency_accuracy']['mean']:5.1f}%"
            )

        # Compare
        dqn_r = results[preset_name]["dqn"]["reward"]["mean"]
        ql_r = results[preset_name]["q_learning"]["reward"]["mean"]
        winner = "DQN" if dqn_r > ql_r else "Q-Learning"
        diff = abs(dqn_r - ql_r)
        print(f"    -> Winner: {winner} (+{diff:.1f})")

    # Also test single-factor sensitivity for both
    key_factors = {
        "sensor_gaussian_std": [0.0, 0.10, 0.20, 0.30],
        "gps_misclass_prob": [0.0, 0.10, 0.20, 0.30],
        "comm_constant_latency": [0.0, 1.0, 3.0, 5.0],
    }

    results["sensitivity"] = {}
    for factor, values in key_factors.items():
        print(f"\n  Sensitivity: {factor}")
        results["sensitivity"][factor] = {}

        for value in values:
            config = NoiseConfig(**{factor: value})
            results["sensitivity"][factor][str(value)] = {}

            for ctrl_name, ctrl in controllers.items():
                ctrl.reset_stats()
                result = evaluate_condition(ctrl, config, n_seeds=n_seeds)
                results["sensitivity"][factor][str(value)][ctrl_name] = result

            dqn_r = results["sensitivity"][factor][str(value)]["dqn"]["reward"]["mean"]
            ql_r = results["sensitivity"][factor][str(value)]["q_learning"]["reward"]["mean"]
            print(f"    {factor}={value:.2f} | DQN: {dqn_r:+7.1f} | Q-Learn: {ql_r:+7.1f}")

    return {
        "study": 5,
        "description": "DQN vs Q-Learning head-to-head",
        "results": results,
    }


# ══════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════


def save_results(study_data, study_num):
    """Save study results to JSON."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"study_{study_num}_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)

    # Environment provenance (design-review finding): this whole harness
    # evaluates every controller — DQN included — against
    # traffic/simulator.py's simplified physics, not SUMO. DQN trains
    # exclusively in SUMO, so these numbers reflect an out-of-distribution
    # evaluation for DQN specifically; see comparator.py's
    # environment_caveat and sumo/evaluate_in_sumo_v2.py for the
    # same-environment comparison.
    study_data["environment"] = ENV_TAG_SIMPLIFIED

    with open(filepath, "w") as f:
        json.dump(study_data, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


def run_all_studies(n_seeds=30):
    """Run all 5 studies sequentially."""
    start = time.time()
    all_results = {}

    for study_num in [1, 2, 3, 4, 5]:
        study_start = time.time()
        result = run_study(study_num, n_seeds=n_seeds)
        all_results[f"study_{study_num}"] = result
        save_results(result, study_num)
        elapsed = time.time() - study_start
        print(f"\n  Study {study_num} completed in {elapsed/60:.1f} minutes")

    total = time.time() - start
    print(f"\n{'='*70}")
    print(f"ALL STUDIES COMPLETE | Total time: {total/60:.1f} minutes")
    print(f"{'='*70}")

    # Save combined results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    combined_path = os.path.join(RESULTS_DIR, f"all_studies_{timestamp}.json")
    all_results["environment"] = ENV_TAG_SIMPLIFIED
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined results: {combined_path}")

    return all_results


def run_study(study_num, **kwargs):
    """Run a specific study by number."""
    runners = {
        1: run_study_1,
        2: run_study_2,
        3: run_study_3,
        4: run_study_4,
        5: run_study_5,
    }
    if study_num not in runners:
        raise ValueError(f"Unknown study: {study_num}. Must be 1-5.")
    return runners[study_num](**kwargs)


# ─── CLI ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RescueWay Robustness Experiment Runner")
    parser.add_argument("--study", type=str, default="all", help="Study number (1-5) or 'all'")
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds per condition")
    parser.add_argument("--preset", type=str, default=None, help="Specific preset for Study 3")
    parser.add_argument("--factor", type=str, default=None, help="Specific factor for Study 2")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("RescueWay Robustness Experiment Runner")
    print(f"Base seed: {BENCHMARK_SEED}")
    print(f"Seeds per condition: {args.seeds}")
    print("=" * 70)

    start = time.time()

    if args.study == "all":
        run_all_studies(n_seeds=args.seeds)
    else:
        study_num = int(args.study)
        kwargs = {"n_seeds": args.seeds}

        if study_num == 2 and args.factor:
            kwargs["factors"] = [args.factor]
        if study_num == 3 and args.preset:
            kwargs["presets"] = [args.preset]

        result = run_study(study_num, **kwargs)
        save_results(result, study_num)

    total = time.time() - start
    print(f"\nTotal time: {total/60:.1f} minutes")
