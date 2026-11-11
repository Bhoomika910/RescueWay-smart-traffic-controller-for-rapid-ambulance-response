"""
RescueWay SUMO Evaluation Script

Loads the trained Q-table from the custom simulator and evaluates
the same RL policy inside SUMO. This validates that the learned
emergency prioritization strategy transfers across simulation
environments.

Usage:
    python evaluate_in_sumo.py
    python evaluate_in_sumo.py --episodes 10 --pattern high --gui
    python evaluate_in_sumo.py --compare-all

Requires:
    - SUMO installed and in PATH
    - intersection.net.xml built (run build_network.bat first)
    - Trained Q-table at ../rl/q_table_v2.pkl
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import traci
from generate_routes import generate_routes
from sumo_adapter import EDGE_TO_ZONE, ZONE_TO_EDGE, SUMOAdapter

from rl.emergency_scoring import is_emergency_action_correct
from rl.q_agent import QAgent
from rl.reward import compute_reward
from rl.state_encoder import StateEncoder

# ─── Configuration ─────────────────────────────────────

SUMO_BINARY = "sumo"  # use "sumo-gui" for visual mode
NET_FILE = os.path.join(os.path.dirname(__file__), "intersection.net.xml")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Timing
GREEN_DURATION = 5  # seconds per green phase
YELLOW_DURATION = 2  # seconds for yellow
ALL_RED_DURATION = 1  # seconds for all-red clearance
EMERGENCY_DURATION = 15  # seconds for emergency hold

# Evaluation
DEFAULT_EPISODES = 5
DEFAULT_DURATION = 300  # seconds per episode
DEFAULT_PATTERN = "medium"


# ─── Controller Implementations ────────────────────────


class SUMORLController:
    """RL controller using trained Q-table."""

    def __init__(self):
        self.name = "Q-Learning RL"
        self.agent = QAgent(epsilon=0.0)  # pure exploitation
        self.agent.load()
        self.encoder = StateEncoder()

    def choose_action(self, active_signal, queue_lengths, emergency_active, ambulance_zone):
        state = self.encoder.encode(
            emergency_active=emergency_active,
            ambulance_zone=ambulance_zone if emergency_active else 0,
            active_signal=active_signal,
            queue_lengths=queue_lengths,
        )
        return self.agent.choose(state)


class SUMOFixedController:
    """Fixed cycle: 1 -> 2 -> 3 -> 4."""

    def __init__(self):
        self.name = "Fixed Timer"
        self.cycle = [1, 2, 3, 4]
        self.index = 0
        self.steps = 0

    def choose_action(self, active_signal, queue_lengths, emergency_active, ambulance_zone):
        self.steps += 1
        if self.steps >= GREEN_DURATION:
            self.steps = 0
            self.index = (self.index + 1) % 4
            return self.cycle[self.index] - 1
        return 4  # extend


class SUMORuleBasedController:
    """Serve most congested zone. Always prioritize ambulance."""

    def __init__(self):
        self.name = "Rule-Based"

    def choose_action(self, active_signal, queue_lengths, emergency_active, ambulance_zone):
        if emergency_active and ambulance_zone > 0:
            return ambulance_zone - 1
        if queue_lengths:
            most = max(queue_lengths, key=queue_lengths.get)
            if queue_lengths[most] > 0:
                return most - 1
        return 4  # extend


class SUMORandomController:
    """Random actions."""

    def __init__(self):
        self.name = "Random"

    def choose_action(self, active_signal, queue_lengths, emergency_active, ambulance_zone):
        return random.randint(0, 5)


# ─── Safe Signal Transition ───────────────────────────


def safe_transition(adapter, from_zone, to_zone):
    """
    Execute safe signal transition in SUMO.
    Same protocol as the custom simulator:
    Current Green -> Yellow -> All Red -> New Green
    """
    if from_zone == to_zone:
        return

    # Phase 1: Yellow
    adapter.set_yellow()
    for _ in range(YELLOW_DURATION):
        adapter.step()

    # Phase 2: All Red
    adapter.set_all_red()
    for _ in range(ALL_RED_DURATION):
        adapter.step()

    # Phase 3: New Green
    adapter.set_signal(to_zone)


# ─── Episode Runner ────────────────────────────────────


def run_episode(
    controller,
    pattern="medium",
    duration=300,
    ambulance_zone=None,
    ambulance_time=None,
    seed=42,
    use_gui=False,
):
    """Run one evaluation episode in SUMO."""
    route_file = os.path.join(os.path.dirname(__file__), "_temp_routes.rou.xml")
    generate_routes(
        pattern=pattern,
        duration=duration,
        ambulance_zone=ambulance_zone,
        ambulance_time=ambulance_time,
        seed=seed,
        output_file=route_file,
    )

    sumo_cmd = [
        SUMO_BINARY if not use_gui else "sumo-gui",
        "-n",
        NET_FILE,
        "-r",
        route_file,
        "--step-length",
        "1",
        "--no-step-log",
        "true",
        "--time-to-teleport",
        "300",
        "--waiting-time-memory",
        "1000",
        "--no-warnings",
        "true",
        # Pins SUMO's own internal RNG to the same seed used for route
        # generation — see the matching comment in sumo_env.py's reset().
        "--seed",
        str(seed),
    ]

    traci.start(sumo_cmd)
    adapter = SUMOAdapter()

    total_reward = 0
    total_wait_time = 0
    emergency_correct = 0
    emergency_total = 0
    emergency_response_times = []
    emergency_detected_step = None
    signal_switches = 0
    queue_history = []
    current_zone = 1
    green_timer = 0

    adapter.set_signal(1)

    try:
        step = 0
        while step < duration:
            adapter.step()
            step += 1
            green_timer += 1

            queue_lengths = adapter.get_queue_lengths()
            queue_history.append(sum(queue_lengths.values()))

            # Check for ambulance EVERY step (not just at decision points)
            amb_present, amb_zone = adapter.detect_ambulance()

            # Also check by vehicle list for ambulances on internal edges
            if not amb_present:
                try:
                    for vid in traci.vehicle.getIDList():
                        if "ambulance" in vid.lower():
                            amb_present = True
                            # Try to get the road
                            road = traci.vehicle.getRoadID(vid)
                            if road in EDGE_TO_ZONE:
                                amb_zone = EDGE_TO_ZONE[road]
                            elif ambulance_zone:
                                amb_zone = ambulance_zone
                            else:
                                amb_zone = 3  # default
                            break
                except Exception:
                    pass

            if amb_present and amb_zone > 0:
                if emergency_detected_step is None:
                    emergency_detected_step = step
                    print(f"      Ambulance detected at step {step} in zone {amb_zone}")

            # Make decision every GREEN_DURATION seconds
            if green_timer >= GREEN_DURATION:
                green_timer = 0

                action = controller.choose_action(
                    active_signal=current_zone,
                    queue_lengths=queue_lengths,
                    emergency_active=amb_present,
                    ambulance_zone=amb_zone if amb_present else 0,
                )

                reward = compute_reward(
                    action=action,
                    amb_zone=amb_zone if amb_present else 0,
                    prev_signal=current_zone,
                    traffic=pattern,
                    queue_lengths=queue_lengths,
                )
                total_reward += reward

                # Track emergency accuracy (count only at decision points).
                # current_zone is still pre-action here, the correct
                # served_zone argument for action==4/"extend".
                if amb_present and amb_zone > 0:
                    emergency_total += 1
                    if is_emergency_action_correct(action, amb_zone, current_zone):
                        emergency_correct += 1
                        if emergency_detected_step is not None:
                            resp_time = step - emergency_detected_step
                            emergency_response_times.append(resp_time)
                            emergency_detected_step = step  # reset for next check

                # Apply action
                if action < 4:
                    new_zone = action + 1
                    if new_zone != current_zone:
                        safe_transition(adapter, current_zone, new_zone)
                        current_zone = new_zone
                        signal_switches += 1
                    else:
                        adapter.set_signal(new_zone)
                else:
                    adapter.set_signal(current_zone)

            # Track wait time
            for zone, edge in ZONE_TO_EDGE.items():
                try:
                    total_wait_time += traci.edge.getWaitingTime(edge)
                except Exception:
                    pass

            # Clear ambulance tracking when it leaves
            if not amb_present and emergency_detected_step is not None:
                emergency_detected_step = None

    except traci.exceptions.FatalTraCIError:
        pass
    finally:
        try:
            traci.close()
        except Exception:
            pass
        try:
            os.remove(route_file)
        except Exception:
            pass

    avg_queue = np.mean(queue_history) if queue_history else 0

    return {
        "total_reward": round(total_reward, 2),
        "avg_queue_length": round(avg_queue, 2),
        "max_queue_length": max(queue_history) if queue_history else 0,
        "avg_wait_time": round(total_wait_time / max(duration, 1), 2),
        "vehicles_cleared": adapter.vehicles_cleared,
        "emergency_accuracy": round(
            (emergency_correct / emergency_total * 100) if emergency_total > 0 else 0, 1
        ),
        "emergency_correct": emergency_correct,
        "emergency_total": emergency_total,
        "avg_emergency_response": round(
            np.mean(emergency_response_times) if emergency_response_times else 0, 2
        ),
        "signal_switches": signal_switches,
        "steps": step,
    }


# ─── Multi-Episode Evaluation ─────────────────────────


def evaluate_controller(
    controller, episodes=5, pattern="medium", duration=300, ambulance_zone=3, use_gui=False
):
    """Run multiple episodes and aggregate metrics."""

    print(f"\n  Evaluating: {controller.name}")
    print(f"  Episodes: {episodes}, Pattern: {pattern}, Duration: {duration}s")

    all_results = []

    for ep in range(episodes):
        seed = 42 + ep  # reproducible but different per episode
        amb_time = random.Random(seed).randint(60, duration - 60)

        result = run_episode(
            controller=controller,
            pattern=pattern,
            duration=duration,
            ambulance_zone=ambulance_zone,
            ambulance_time=amb_time,
            seed=seed,
            use_gui=use_gui and ep == 0,  # GUI only for first episode
        )
        all_results.append(result)

        print(
            f"    Episode {ep+1}: reward={result['total_reward']:+.1f}, "
            f"queue={result['avg_queue_length']:.1f}, "
            f"emerg_acc={result['emergency_accuracy']:.0f}%"
        )

    # Aggregate
    return {
        "name": controller.name,
        "episodes": episodes,
        "pattern": pattern,
        "reward": {
            "mean": round(np.mean([r["total_reward"] for r in all_results]), 2),
            "std": round(np.std([r["total_reward"] for r in all_results]), 2),
        },
        "queue": {
            "mean": round(np.mean([r["avg_queue_length"] for r in all_results]), 2),
        },
        "wait_time": {
            "mean": round(np.mean([r["avg_wait_time"] for r in all_results]), 2),
        },
        "emergency": {
            "accuracy": round(np.mean([r["emergency_accuracy"] for r in all_results]), 1),
            "avg_response": round(np.mean([r["avg_emergency_response"] for r in all_results]), 2),
        },
        "throughput": {
            "mean": round(np.mean([r["vehicles_cleared"] for r in all_results]), 1),
        },
        "switches": {
            "mean": round(np.mean([r["signal_switches"] for r in all_results]), 1),
        },
    }


# ─── Full Comparison ──────────────────────────────────


def run_comparison(episodes=5, pattern="medium", duration=300, use_gui=False):
    """Compare all 4 controllers in SUMO."""

    print("=" * 70)
    print("RescueWay SUMO Evaluation")
    print(f"Pattern: {pattern} | Episodes: {episodes} | Duration: {duration}s")
    print("=" * 70)

    controllers = [
        SUMORLController(),
        SUMOFixedController(),
        SUMORuleBasedController(),
        SUMORandomController(),
    ]

    results = {}
    for ctrl in controllers:
        result = evaluate_controller(
            ctrl,
            episodes=episodes,
            pattern=pattern,
            duration=duration,
            use_gui=use_gui,
        )
        results[ctrl.name] = result

    # Print comparison table
    print("\n" + "=" * 70)
    print("SUMO EVALUATION RESULTS")
    print("=" * 70)
    print(
        f"{'Controller':<20} {'Reward':>10} {'Queue':>10} {'Emergency':>12} "
        f"{'Response':>10} {'Throughput':>12}"
    )
    print("-" * 70)

    for name, r in results.items():
        print(
            f"{name:<20} {r['reward']['mean']:>+10.1f} {r['queue']['mean']:>10.1f} "
            f"{r['emergency']['accuracy']:>11.1f}% "
            f"{r['emergency']['avg_response']:>10.1f}s "
            f"{r['throughput']['mean']:>12.1f}"
        )

    print("=" * 70)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULTS_DIR, f"sumo_comparison_{timestamp}.json")
    with open(result_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "config": {
                    "episodes": episodes,
                    "pattern": pattern,
                    "duration": duration,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {result_file}")

    return results


# ─── Cross-Simulator Comparison ───────────────────────


def cross_simulator_comparison(episodes=5, pattern="medium"):
    """
    Run evaluation in SUMO and compare with custom simulator results.
    This is the key research result — transfer validation.
    """
    print("\n" + "=" * 70)
    print("CROSS-SIMULATOR TRANSFER VALIDATION")
    print("=" * 70)

    # Run in SUMO
    sumo_results = run_comparison(episodes=episodes, pattern=pattern)

    # Run in custom simulator
    print("\n  Running custom simulator comparison...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from evaluation.comparator import ControllerComparator

    comparator = ControllerComparator()
    comparator.run_comparison(episodes=episodes * 10)
    custom_summary = comparator.get_summary()

    # Print cross-comparison
    print("\n" + "=" * 70)
    print("TRANSFER VALIDATION RESULTS")
    print("=" * 70)
    print(f"{'Controller':<20} {'Custom Reward':>15} {'SUMO Reward':>15} {'Transfer %':>12}")
    print("-" * 70)

    for name in ["Q-Learning RL", "Fixed Timer", "Rule-Based", "Random"]:
        custom_key = {
            "Q-Learning RL": "rl",
            "Fixed Timer": "fixed",
            "Rule-Based": "rule_based",
            "Random": "random",
        }[name]

        custom_reward = custom_summary["controllers"].get(custom_key, {}).get("avg_reward", 0)
        sumo_reward = sumo_results.get(name, {}).get("reward", {}).get("mean", 0)

        transfer = (sumo_reward / custom_reward * 100) if custom_reward != 0 else 0

        print(f"{name:<20} {custom_reward:>+15.1f} {sumo_reward:>+15.1f} {transfer:>11.1f}%")

    print("=" * 70)
    print("\nTransfer % = (SUMO reward / Custom reward) × 100")
    print("100% = perfect transfer. >80% = good transfer. <50% = poor transfer.")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULTS_DIR, f"transfer_validation_{timestamp}.json")
    with open(result_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "custom_simulator": custom_summary,
                "sumo_simulator": sumo_results,
            },
            f,
            indent=2,
        )

    print(f"\nFull results saved to: {result_file}")


# ─── Main ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RescueWay SUMO Evaluation")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        choices=["low", "medium", "high", "rush_hour"],
    )
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION)
    parser.add_argument("--gui", action="store_true", help="Show SUMO GUI")
    parser.add_argument("--compare-all", action="store_true", help="Run all controllers comparison")
    parser.add_argument(
        "--transfer", action="store_true", help="Run cross-simulator transfer validation"
    )
    parser.add_argument("--rl-only", action="store_true", help="Evaluate only the RL controller")

    args = parser.parse_args()

    # Check network file exists
    if not os.path.exists(NET_FILE):
        print(f"ERROR: Network file not found: {NET_FILE}")
        print("Run build_network.bat first to generate the SUMO network.")
        sys.exit(1)

    if args.transfer:
        cross_simulator_comparison(
            episodes=args.episodes,
            pattern=args.pattern,
        )
    elif args.compare_all:
        run_comparison(
            episodes=args.episodes,
            pattern=args.pattern,
            duration=args.duration,
            use_gui=args.gui,
        )
    elif args.rl_only:
        ctrl = SUMORLController()
        result = evaluate_controller(
            ctrl,
            episodes=args.episodes,
            pattern=args.pattern,
            duration=args.duration,
            use_gui=args.gui,
        )
        print("\nRL Agent in SUMO:")
        print(json.dumps(result, indent=2))
    else:
        run_comparison(
            episodes=args.episodes,
            pattern=args.pattern,
            duration=args.duration,
            use_gui=args.gui,
        )
