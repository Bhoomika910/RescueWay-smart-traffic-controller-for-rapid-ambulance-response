# metrics/calculator.py
import random

import numpy as np

from config import BENCHMARK_SEED
from gps.zone_model import ZONE_CONFIG
from rl.emergency_scoring import is_emergency_action_correct


class PerformanceCalculator:
    """
    Computes detailed performance metrics for a controller
    over a complete evaluation run.
    """

    @staticmethod
    def evaluate_run(controller, episodes=100, max_steps=30, traffic_pattern="medium", seed=None):
        """
        Run a controller through multiple episodes and compute metrics.
        Returns detailed performance report.

        `seed` (default: config.BENCHMARK_SEED) reseeds both `random` and
        `numpy.random` at the start of each episode from `seed + ep`. This
        makes results reproducible and, critically, means calling this
        function once per controller (e.g. compare_across_traffic /
        generate_report_charts.py) gives every controller the identical
        traffic-arrival sequence for the same episode index — previously
        this was never seeded at all, so sequential per-controller calls
        consumed one shared, uncontrolled RNG stream and were not
        comparable to each other.
        """
        from rl.environment import TrafficEnv

        if seed is None:
            seed = BENCHMARK_SEED

        all_rewards = []
        all_queue_lengths = []
        all_wait_times = []
        all_throughputs = []
        emergency_correct = 0
        emergency_total = 0
        emergency_response_steps = []
        total_switches = 0
        zone_green_counts = {1: 0, 2: 0, 3: 0, 4: 0}

        for ep in range(episodes):
            random.seed(seed + ep)
            np.random.seed(seed + ep)

            env = TrafficEnv()
            env.simulator.set_traffic_pattern(traffic_pattern)
            env.reset()

            episode_reward = 0
            prev_action = -1
            emergency_wait = 0
            in_emergency = False
            already_responded = False

            for step in range(max_steps):
                metrics = env.simulator.get_metrics()
                queue_lengths = {z: metrics[z].queue_length for z in [1, 2, 3, 4]}
                system = env.simulator.get_system_metrics()

                # emergency_eta/distance forwarded so DQN/Q-Learning are
                # evaluated on the state distribution they were trained on
                # (see ARCHITECTURE.md Bug #1 and the matching fix in
                # evaluation/comparator.py / robustness_runner.py). env
                # (rl.environment.TrafficEnv) already tracks a real
                # countdown ETA internally — reuse it directly rather than
                # re-deriving it.
                distance = (
                    float(ZONE_CONFIG[env.ambulance_zone]["avg_distance"])
                    if env.ambulance_active and env.ambulance_zone in ZONE_CONFIG
                    else 0.0
                )
                action = controller.choose_action(
                    active_signal=env.active_signal,
                    queue_lengths=queue_lengths,
                    emergency_active=env.ambulance_active,
                    ambulance_zone=env.ambulance_zone,
                    emergency_eta=env.ambulance_eta if env.ambulance_active else 0.0,
                    distance=distance,
                )

                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                # Track queue lengths
                all_queue_lengths.append(sum(queue_lengths.values()))

                # Track wait times
                all_wait_times.append(system.get("avg_wait_time_all", 0))

                # Track zone green time
                if action < 4:
                    zone_green_counts[action + 1] = zone_green_counts.get(action + 1, 0) + 1

                # Track switches
                if action != prev_action and action < 4:
                    total_switches += 1
                prev_action = action

                # Track emergency. emergency_correct/emergency_total remain
                # per-tick (every tick during an active emergency counts
                # toward the denominator, every correctly-served tick
                # counts toward the numerator) — unchanged semantics.
                # emergency_response_steps is per-EVENT: only the first
                # correct tick of each continuous emergency is recorded.
                # Previously `in_emergency` was reset to False on ANY
                # correct action (even mid-ongoing-emergency), so a
                # controller that kept serving the right zone for several
                # more ticks would have each subsequent tick misread as
                # "a new emergency just started", appending a second
                # spurious 1-step "response" for what was really one
                # sustained, already-resolved event.
                if info["ambulance_active"] and info["ambulance_zone"] > 0:
                    emergency_total += 1
                    if not in_emergency:
                        in_emergency = True
                        already_responded = False
                        emergency_wait = 0

                    emergency_wait += 1

                    # env.active_signal reflects the post-step served zone
                    # (env.step() already ran above), so it's the correct
                    # served_zone argument here.
                    if is_emergency_action_correct(
                        action, info["ambulance_zone"], env.active_signal
                    ):
                        emergency_correct += 1
                        if not already_responded:
                            emergency_response_steps.append(emergency_wait)
                            already_responded = True
                else:
                    in_emergency = False
                    already_responded = False

                if done:
                    break

            all_rewards.append(episode_reward)
            all_throughputs.append(env.simulator.get_system_metrics()["total_vehicles_cleared"])

        # Compute Jain's fairness index
        green_values = list(zone_green_counts.values())
        n = len(green_values)
        if sum(green_values) > 0:
            fairness = (sum(green_values) ** 2) / (n * sum(v**2 for v in green_values))
        else:
            fairness = 1.0

        return {
            "name": controller.name,
            "episodes": episodes,
            "max_steps": max_steps,
            "traffic_pattern": traffic_pattern,
            "reward": {
                "mean": round(float(np.mean(all_rewards)), 2),
                "std": round(float(np.std(all_rewards)), 2),
                "min": round(float(np.min(all_rewards)), 2),
                "max": round(float(np.max(all_rewards)), 2),
                "median": round(float(np.median(all_rewards)), 2),
            },
            "queue": {
                "mean": round(float(np.mean(all_queue_lengths)), 2),
                "max": round(float(np.max(all_queue_lengths)), 2),
                "std": round(float(np.std(all_queue_lengths)), 2),
            },
            "wait_time": {
                "mean": round(float(np.mean(all_wait_times)), 2),
                "max": round(float(np.max(all_wait_times)), 2),
            },
            "throughput": {
                "mean": round(float(np.mean(all_throughputs)), 2),
                "total": int(sum(all_throughputs)),
            },
            "emergency": {
                "total_events": emergency_total,
                "correct_responses": emergency_correct,
                "accuracy": round(
                    (emergency_correct / emergency_total * 100) if emergency_total > 0 else 0, 1
                ),
                "avg_response_steps": round(
                    float(np.mean(emergency_response_steps)) if emergency_response_steps else 0, 2
                ),
            },
            "efficiency": {
                "total_switches": total_switches,
                "switches_per_episode": round(total_switches / episodes, 2),
                "fairness_index": round(fairness, 3),
                "zone_green_distribution": zone_green_counts,
            },
        }

    @staticmethod
    def compare_across_traffic(controller, patterns=None, episodes=50, seed=None):
        """Test a controller across different traffic patterns."""
        if patterns is None:
            patterns = ["low", "medium", "high", "rush_hour"]

        results = {}
        for pattern in patterns:
            results[pattern] = PerformanceCalculator.evaluate_run(
                controller, episodes=episodes, traffic_pattern=pattern, seed=seed
            )

        return results
