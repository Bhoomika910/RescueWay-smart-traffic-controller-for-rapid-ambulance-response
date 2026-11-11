"""
RescueWay GPS Misclassification Confusion Analysis.

Records EVERY (true_zone → reported_zone → chosen_action) triple during
the GPS misclassification sweep, producing a confusion matrix showing
exactly which wrong zones the system redirects emergencies to.

This answers the reviewer question: "when GPS misclassification occurs,
WHERE does the ambulance get sent — is it random, or is there a pattern?"

Usage:
    python -m evaluation.gps_confusion_analysis --seeds 30
    python -m evaluation.gps_confusion_analysis --seeds 50 --probs 0.1 0.2 0.3 0.5

Output:
    results/robustness/gps_confusion_<timestamp>.json
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
from controllers.rule_based import RuleBasedController
from gps.zone_model import ZONE_CONFIG
from noise import NoiseApplicator, NoiseConfig
from rl.dqn_agent import DQNAgent
from rl.emergency_scoring import is_emergency_action_correct, resolved_zone
from rl.environment import TrafficEnv
from rl.state_features import StateFeatureExtractor

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "robustness")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_confusion_episode(
    controller, noise_config, seed, max_steps=30, emergency_prob=BENCHMARK_EMERGENCY_PROBABILITY
):
    """
    Run one episode recording every (true_zone, reported_zone, action) event.
    Returns a list of confusion events.
    """
    np.random.seed(seed)
    stdlib_random.seed(seed)

    env = TrafficEnv()
    env.reset()
    noise = NoiseApplicator(noise_config)
    noise.reset()

    events = []
    emergency_active = False
    emergency_zone = 0
    true_emergency_zone = 0
    emergency_timer = 0
    emergency_eta = 0.0

    for step in range(max_steps):
        metrics = env.simulator.get_metrics()
        queue_lengths = {z: metrics[z].queue_length for z in [1, 2, 3, 4]}

        # Emergency generation (same logic as robustness_runner). Duration
        # is seeded from ZONE_CONFIG's per-zone ETA (5/10/18/30s), matching
        # the flagship benchmark's deliberate timing distribution rather
        # than an arbitrary randint window.
        if not emergency_active:
            if stdlib_random.random() < emergency_prob:
                true_emergency_zone = stdlib_random.choice([1, 2, 3, 4])
                emergency_timer = float(ZONE_CONFIG[true_emergency_zone]["eta_seconds"])

                # GPS false negative check
                if noise.gps.check_false_negative(ambulance_present=True):
                    continue

                # Packet loss check
                if noise.comm.check_packet_loss():
                    continue

                # Zone misclassification
                reported_zone = noise.gps.apply_zone(true_emergency_zone)

                emergency_active = True
                emergency_zone = reported_zone
                emergency_eta = emergency_timer
        else:
            emergency_timer -= 1
            emergency_eta = max(0.0, emergency_eta - 1.0)
            if emergency_timer <= 0:
                emergency_active = False
                emergency_zone = 0
                true_emergency_zone = 0
                emergency_eta = 0.0

        # Apply sensor noise
        noisy_queues = noise.sensor.apply(queue_lengths)

        # Distance is a static per-zone property, keyed by the reported
        # zone (what the controller actually perceives).
        distance = (
            float(ZONE_CONFIG[emergency_zone]["avg_distance"])
            if emergency_active and emergency_zone in ZONE_CONFIG
            else 0.0
        )

        # Controller decision. emergency_eta/distance forwarded so DQN is
        # evaluated on the state distribution it was trained on — see the
        # matching fix/comment in evaluation/robustness_runner.py.
        action = controller.choose_action(
            active_signal=env.active_signal,
            queue_lengths=noisy_queues,
            emergency_active=emergency_active,
            ambulance_zone=emergency_zone,
            emergency_eta=emergency_eta if emergency_active else 0.0,
            distance=distance,
        )

        # Record confusion event
        if emergency_active and true_emergency_zone > 0:
            # resolved_zone() resolves action==4 ("extend current green")
            # to whichever zone is actually being served this tick — using
            # it (rather than raw action) for correct_action/followed_report
            # is what rl/emergency_scoring.py exists to make consistent
            # across every call site instead of nine local reimplementations.
            chosen_zone = resolved_zone(action, env.active_signal)
            events.append(
                {
                    "true_zone": true_emergency_zone,
                    "reported_zone": emergency_zone,
                    "misclassified": emergency_zone != true_emergency_zone,
                    "action": action,
                    "chosen_zone": chosen_zone,
                    "correct_action": is_emergency_action_correct(
                        action, true_emergency_zone, env.active_signal
                    ),
                    "followed_report": is_emergency_action_correct(
                        action, emergency_zone, env.active_signal
                    ),
                }
            )

        # Apply action
        env.active_signal = action + 1 if action < 4 else env.active_signal
        env.simulator.step(
            active_signal=env.active_signal,
            is_green=True,
            current_time=step,
        )

    return events


def build_confusion_matrix(events):
    """Build a 4×4 confusion matrix: true_zone (rows) × chosen_zone (cols)."""
    matrix = np.zeros((4, 4), dtype=int)
    for e in events:
        true_idx = e["true_zone"] - 1
        chosen_idx = e["chosen_zone"] - 1
        if 0 <= true_idx < 4 and 0 <= chosen_idx < 4:
            matrix[true_idx, chosen_idx] += 1
    return matrix


def build_misclass_matrix(events):
    """Build: true_zone (rows) × reported_zone (cols) — GPS error pattern."""
    matrix = np.zeros((4, 4), dtype=int)
    for e in events:
        true_idx = e["true_zone"] - 1
        rep_idx = e["reported_zone"] - 1
        if 0 <= true_idx < 4 and 0 <= rep_idx < 4:
            matrix[true_idx, rep_idx] += 1
    return matrix


def run_analysis(n_seeds=30, misclass_probs=None):
    """Run GPS confusion analysis across multiple misclassification rates."""

    if misclass_probs is None:
        misclass_probs = [0.0, 0.10, 0.20, 0.30, 0.50]

    print("=" * 70)
    print("GPS MISCLASSIFICATION CONFUSION ANALYSIS")
    print(f"Seeds: {n_seeds} | Misclass probs: {misclass_probs}")
    print("=" * 70)

    # Create controllers
    dqn_agent = DQNAgent(epsilon=0.0, inference_mode=True)
    dqn_agent.feature_extractor = StateFeatureExtractor(noise_config=NoiseConfig())
    dqn_agent.load()
    dqn_ctrl = DQNController(agent=dqn_agent)
    rb_ctrl = RuleBasedController()

    controllers = {"dqn": dqn_ctrl, "rule_based": rb_ctrl}
    results = {}

    for prob in misclass_probs:
        print(f"\n  GPS misclassification probability: {prob:.2f}")
        config = NoiseConfig(gps_misclass_prob=prob)
        results[str(prob)] = {}

        for ctrl_name, ctrl in controllers.items():
            all_events = []
            for seed_idx in range(n_seeds):
                seed = BENCHMARK_SEED + seed_idx
                events = run_confusion_episode(ctrl, config, seed)
                all_events.extend(events)

            # Build matrices
            action_matrix = build_confusion_matrix(all_events)
            gps_matrix = build_misclass_matrix(all_events)

            total_events = len(all_events)
            misclassified_events = sum(1 for e in all_events if e["misclassified"])
            correct_actions = sum(1 for e in all_events if e["correct_action"])
            followed_report = sum(1 for e in all_events if e["followed_report"])

            result = {
                "total_emergency_steps": total_events,
                "misclassified_count": misclassified_events,
                "misclassification_rate": round(
                    misclassified_events / max(total_events, 1) * 100, 1
                ),
                "correct_action_count": correct_actions,
                "correct_action_rate": round(correct_actions / max(total_events, 1) * 100, 1),
                "followed_gps_report": followed_report,
                "followed_gps_rate": round(followed_report / max(total_events, 1) * 100, 1),
                "action_confusion_matrix": action_matrix.tolist(),
                "gps_report_matrix": gps_matrix.tolist(),
            }

            results[str(prob)][ctrl_name] = result

            # Print
            print(f"\n    {ctrl_name}:")
            print(f"      Total emergency steps: {total_events}")
            print(
                f"      Misclassified by GPS:  {misclassified_events} "
                f"({result['misclassification_rate']:.1f}%)"
            )
            print(
                f"      Correct actions (vs TRUE zone): {correct_actions} "
                f"({result['correct_action_rate']:.1f}%)"
            )
            print(
                f"      Followed GPS report:   {followed_report} "
                f"({result['followed_gps_rate']:.1f}%)"
            )
            print("      Action confusion matrix (true_zone rows × chosen_zone cols):")
            print("             Z1    Z2    Z3    Z4")
            for row_idx in range(4):
                row_total = max(action_matrix[row_idx].sum(), 1)
                row_pcts = action_matrix[row_idx] / row_total * 100
                vals = "  ".join(f"{action_matrix[row_idx, c]:5d}" for c in range(4))
                pcts = "  ".join(f"{row_pcts[c]:4.0f}%" for c in range(4))
                print(f"        Z{row_idx+1}: {vals}  ({pcts})")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(RESULTS_DIR, f"gps_confusion_{timestamp}.json")
    output = {
        "description": "GPS misclassification confusion analysis",
        "environment": ENV_TAG_SIMPLIFIED,
        "n_seeds": n_seeds,
        "misclass_probs": misclass_probs,
        "controllers": list(controllers.keys()),
        "results": results,
        "matrix_legend": {
            "action_confusion_matrix": "Rows = true ambulance zone (1-4), "
            "Cols = zone the controller chose to green (1-4)",
            "gps_report_matrix": "Rows = true ambulance zone (1-4), "
            "Cols = zone GPS reported (1-4)",
        },
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {filepath}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPS Misclassification Confusion Analysis")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--probs", type=float, nargs="+", default=[0.0, 0.10, 0.20, 0.30, 0.50])
    args = parser.parse_args()
    run_analysis(n_seeds=args.seeds, misclass_probs=args.probs)
