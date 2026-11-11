# evaluation/multi_ambulance_benchmark.py
"""
RescueWay Multi-Ambulance Conflict Resolution Benchmark.

Compares three scheduling strategies for handling multiple
simultaneous ambulance requests:

1. FIFO (First-Come-First-Served) — baseline
2. ETA-based (closest first) — current RescueWay implementation
3. Congestion-weighted — ETA adjusted by zone congestion

Research Question:
  "When multiple ambulances request conflicting green phases,
  what is the best strategy for assigning priority while minimizing
  emergency delay and overall traffic congestion?"

Usage:
    python -m evaluation.multi_ambulance_benchmark
    python -m evaluation.multi_ambulance_benchmark --seeds 30
"""
import argparse
import json
import os
import random as stdlib_random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BENCHMARK_SEED
from gps.zone_model import ZONE_CONFIG
from rl.reward import compute_reward
from traffic.simulator import TrafficSimulator

# ─── Output ──────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "multi_ambulance")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════
# SCHEDULING STRATEGIES
# ══════════════════════════════════════════════════════


class SchedulingStrategy:
    """Base class for ambulance scheduling strategies."""

    def __init__(self, name):
        self.name = name

    def select_next(self, queue, queue_lengths):
        """
        Given a queue of waiting ambulances and current traffic,
        return the index of the ambulance to serve next.
        """
        raise NotImplementedError


class FIFOStrategy(SchedulingStrategy):
    """
    First-Come-First-Served.

    Ambulances are served in the order they arrive.
    No consideration of distance, ETA, or traffic conditions.
    This is the simplest possible policy and the lower-bound baseline.
    """

    def __init__(self):
        super().__init__("FIFO")

    def select_next(self, queue, queue_lengths):
        if not queue:
            return -1
        return 0  # Always pick the first (oldest) request


class ETAStrategy(SchedulingStrategy):
    """
    ETA-Based Priority (Current RescueWay Implementation).

    Serve the closest ambulance first (shortest ETA).
    Rationale: the closest ambulance reaches the intersection soonest,
    so delaying it causes the most harm.
    """

    def __init__(self):
        super().__init__("ETA-Based")

    def select_next(self, queue, queue_lengths):
        if not queue:
            return -1
        # Pick ambulance with shortest ETA
        return min(range(len(queue)), key=lambda i: queue[i]["eta"])


class CongestionWeightedStrategy(SchedulingStrategy):
    """
    Congestion-Weighted Priority.

    score = (1 - eta_normalized) + weight * congestion_normalized

    Uses normalized additive formula so ETA proximity and zone
    congestion compete on equal footing. With weight=1.0, a zone
    at maximum congestion can override a closer ambulance —
    reflecting the real-world tradeoff between emergency urgency
    and traffic impact.
    """

    def __init__(self, congestion_weight=1.0):
        super().__init__("Congestion-Weighted")
        self.congestion_weight = congestion_weight
        self.max_eta = 30.0  # Zone 4's ETA, used for normalization

    def select_next(self, queue, queue_lengths):
        if not queue:
            return -1

        max_queue = max(queue_lengths.values()) if queue_lengths else 1
        if max_queue == 0:
            max_queue = 1

        def score(i):
            amb = queue[i]
            eta_norm = 1.0 - (amb["eta"] / self.max_eta)
            zone_queue = queue_lengths.get(amb["zone"], 0)
            cong_norm = zone_queue / max_queue
            return eta_norm + self.congestion_weight * cong_norm

        return max(range(len(queue)), key=score)


# ══════════════════════════════════════════════════════
# MULTI-AMBULANCE SCENARIO GENERATOR
# ══════════════════════════════════════════════════════


def generate_scenario(scenario_type, seed):
    """
    Generate multi-ambulance scenarios where ambulances arrive
    SIMULTANEOUSLY to force the strategy to make a real choice.
    """
    np.random.seed(seed)
    stdlib_random.seed(seed)

    scenarios = {
        # Far zone listed FIRST so FIFO picks far, ETA picks close
        "two_opposing": [
            {"zone": 3, "arrival_step": 3},  # Far (ETA=18s) — FIFO picks this
            {"zone": 1, "arrival_step": 3},  # Close (ETA=5s) — ETA picks this
        ],
        "two_adjacent": [
            {"zone": 3, "arrival_step": 3},  # Far (ETA=18s)
            {"zone": 2, "arrival_step": 3},  # Closer (ETA=10s)
        ],
        # Three ambulances: farthest first in list
        "triple_rapid": [
            {"zone": 4, "arrival_step": 3},  # Farthest (ETA=30s)
            {"zone": 2, "arrival_step": 3},  # Mid (ETA=10s)
            {"zone": 1, "arrival_step": 4},  # Closest (ETA=5s)
        ],
        # Classic conflict: far listed first
        "close_vs_far": [
            {"zone": 4, "arrival_step": 3},  # Far (500m, ETA=30s)
            {"zone": 1, "arrival_step": 3},  # Close (40m, ETA=5s)
        ],
        # Far arrives first, close arrives one step later
        "far_then_close": [
            {"zone": 4, "arrival_step": 3},  # Far first
            {"zone": 1, "arrival_step": 4},  # Close one step later
        ],
        # All four zones — farthest first, closest last
        "four_simultaneous": [
            {"zone": 4, "arrival_step": 3},  # ETA=30s
            {"zone": 3, "arrival_step": 3},  # ETA=18s
            {"zone": 2, "arrival_step": 3},  # ETA=10s
            {"zone": 1, "arrival_step": 3},  # ETA=5s
        ],
        # Two waves: far zones first in each wave
        "two_waves": [
            {"zone": 3, "arrival_step": 3},
            {"zone": 1, "arrival_step": 3},
            {"zone": 4, "arrival_step": 10},
            {"zone": 2, "arrival_step": 10},
        ],
        # Random
        "random": [
            {
                "zone": stdlib_random.randint(1, 4),
                "arrival_step": stdlib_random.choice([3, 3, 4, 4, 5]),
            }
            for _ in range(stdlib_random.randint(2, 4))
        ],
    }

    return scenarios.get(scenario_type, scenarios["two_opposing"])


# ══════════════════════════════════════════════════════
# EPISODE RUNNER
# ══════════════════════════════════════════════════════


def run_multi_ambulance_episode(
    strategy, scenario_events, seed, max_steps=30, traffic_pattern="medium"
):
    """
    Run one episode with multiple ambulances using a given scheduling strategy.

    Key design: ambulances accumulate in queue BEFORE any are served.
    This forces the strategy to choose between competing requests.
    """
    np.random.seed(seed)
    stdlib_random.seed(seed)

    simulator = TrafficSimulator()
    simulator.set_traffic_pattern(traffic_pattern)
    active_signal = stdlib_random.choice([1, 2, 3, 4])

    # Warm up
    for _ in range(5):
        simulator.step(active_signal=active_signal, is_green=True)

    # State tracking
    pending_queue = []  # Ambulances waiting to be served
    active_emergency = None
    active_green_remaining = 0
    episode_reward = 0
    total_queue_length = 0
    emergency_correct = 0
    emergency_total = 0
    total_emergency_delay = 0
    emergencies_served = 0
    all_arrived = set()

    for step in range(max_steps):
        metrics = simulator.get_metrics()
        queue_lengths = {z: metrics[z].queue_length for z in [1, 2, 3, 4]}

        # Add new ambulances to pending queue at their arrival step
        for idx, event in enumerate(scenario_events):
            if event["arrival_step"] == step and idx not in all_arrived:
                all_arrived.add(idx)
                zone = event["zone"]
                config = ZONE_CONFIG[zone]
                pending_queue.append(
                    {
                        "zone": zone,
                        "eta": config["eta_seconds"],
                        "green_duration": config["emergency_green"],
                        "arrival_step": step,
                        "index": idx,
                    }
                )

        # If active emergency is done, clear it
        if active_emergency is not None:
            active_green_remaining -= 1
            if active_green_remaining <= 0:
                emergencies_served += 1
                active_emergency = None

        # If no active emergency and pending queue has items, select next
        if active_emergency is None and pending_queue:
            idx = strategy.select_next(pending_queue, queue_lengths)
            if idx >= 0:
                selected = pending_queue.pop(idx)
                active_emergency = selected
                active_green_remaining = selected["green_duration"] // 2
                # Track delay
                delay = step - selected["arrival_step"]
                total_emergency_delay += delay

        # Determine action
        if active_emergency:
            action = active_emergency["zone"] - 1
            ambulance_zone = active_emergency["zone"]
        else:
            action = max(queue_lengths, key=queue_lengths.get) - 1
            ambulance_zone = 0

        # Compute reward
        reward = compute_reward(
            action=action,
            amb_zone=ambulance_zone,
            prev_signal=active_signal,
            traffic=traffic_pattern,
            queue_lengths=queue_lengths,
        )

        # Apply action
        new_signal = action + 1 if action < 4 else active_signal
        active_signal = new_signal
        simulator.step(active_signal=active_signal, is_green=True, current_time=step)

        episode_reward += reward
        total_queue_length += sum(queue_lengths.values())

        # Emergency accuracy
        if active_emergency:
            emergency_total += 1
            if active_signal == active_emergency["zone"]:
                emergency_correct += 1

    return {
        "reward": round(float(episode_reward), 2),
        "avg_queue": round(total_queue_length / max(max_steps, 1), 2),
        "emergency_accuracy": round(
            (emergency_correct / emergency_total * 100) if emergency_total > 0 else 100.0, 1
        ),
        "emergencies_served": emergencies_served,
        "emergencies_total": len(scenario_events),
        "emergencies_unserved": len(pending_queue),
        "avg_emergency_delay": round(total_emergency_delay / max(emergencies_served, 1), 2),
    }


# ══════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ══════════════════════════════════════════════════════


def run_benchmark(n_seeds=30):
    """
    Compare all three scheduling strategies across all scenarios.
    """
    strategies = [
        FIFOStrategy(),
        ETAStrategy(),
        CongestionWeightedStrategy(),
    ]

    scenario_types = [
        "two_opposing",
        "two_adjacent",
        "triple_rapid",
        "close_vs_far",
        "far_then_close",
        "four_simultaneous",
        "two_waves",
        "random",
    ]

    print("=" * 70)
    print("MULTI-AMBULANCE CONFLICT RESOLUTION BENCHMARK")
    print(f"Strategies: {[s.name for s in strategies]}")
    print(f"Scenarios: {len(scenario_types)}")
    print(f"Seeds per condition: {n_seeds}")
    print("=" * 70)

    results = {}

    for scenario_type in scenario_types:
        print(f"\n  Scenario: {scenario_type}")
        results[scenario_type] = {}

        for strategy in strategies:
            episode_results = []

            for seed_idx in range(n_seeds):
                seed = BENCHMARK_SEED + seed_idx
                events = generate_scenario(scenario_type, seed)
                result = run_multi_ambulance_episode(
                    strategy,
                    events,
                    seed,
                )
                episode_results.append(result)

            # Aggregate
            def agg(key):
                vals = [r[key] for r in episode_results]
                return {
                    "mean": round(float(np.mean(vals)), 2),
                    "std": round(float(np.std(vals)), 2),
                    "ci_95": round(1.96 * float(np.std(vals)) / np.sqrt(len(vals)), 2),
                }

            agg_result = {
                "reward": agg("reward"),
                "avg_queue": agg("avg_queue"),
                "emergency_accuracy": agg("emergency_accuracy"),
                "avg_emergency_delay": agg("avg_emergency_delay"),
                "emergencies_served": agg("emergencies_served"),
            }

            results[scenario_type][strategy.name] = agg_result

            r = agg_result
            print(
                f"    {strategy.name:22s} | "
                f"reward={r['reward']['mean']:+8.1f} ± {r['reward']['ci_95']:5.1f} | "
                f"emerg={r['emergency_accuracy']['mean']:5.1f}% | "
                f"delay={r['avg_emergency_delay']['mean']:4.1f} steps"
            )

    # Summary: find winner per scenario
    print(f"\n{'-' * 70}")
    print("  SUMMARY: Best Strategy Per Scenario")
    print(f"{'-' * 70}")

    strategy_wins = {s.name: 0 for s in strategies}

    for scenario_type in scenario_types:
        best_name = max(
            results[scenario_type], key=lambda s: results[scenario_type][s]["reward"]["mean"]
        )
        best_reward = results[scenario_type][best_name]["reward"]["mean"]
        strategy_wins[best_name] += 1
        print(f"    {scenario_type:22s} -> {best_name} ({best_reward:+.1f})")

    print(f"\n  Win count: {strategy_wins}")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(RESULTS_DIR, f"multi_amb_{timestamp}.json")
    output = {
        "description": "Multi-ambulance conflict resolution benchmark",
        "strategies": [s.name for s in strategies],
        "scenarios": scenario_types,
        "n_seeds": n_seeds,
        "results": results,
        "strategy_wins": strategy_wins,
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {filepath}")

    return output


# ─── CLI ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Ambulance Conflict Resolution Benchmark")
    parser.add_argument("--seeds", type=int, default=30, help="Seeds per condition")
    args = parser.parse_args()

    run_benchmark(n_seeds=args.seeds)
