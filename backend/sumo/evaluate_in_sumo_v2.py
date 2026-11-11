"""
RescueWay SUMO Evaluation v2 — NOW INCLUDES DQN

Adds the Double DQN controller to the existing SUMO comparison so that
the paper's "SUMO-trained policy transfers" claim has actual numbers.

Changes from evaluate_in_sumo.py:
  - Adds SUMODQNController (wraps DQNAgent, same as live system)
  - Adds --dqn-only flag
  - run_comparison() now includes DQN in the controller list
  - cross_simulator_comparison() includes DQN on both sides

Usage:
    cd backend/sumo
    python evaluate_in_sumo_v2.py --compare-all --episodes 10
    python evaluate_in_sumo_v2.py --dqn-only --episodes 10
    python evaluate_in_sumo_v2.py --transfer --episodes 10

Requires: SUMO installed and in PATH, intersection.net.xml built.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import traci
from generate_routes import generate_routes
from sumo_adapter import EDGE_TO_ZONE, ZONE_TO_EDGE, SUMOAdapter

from config import ENV_TAG_SUMO
from rl.dqn_agent import DQNAgent
from rl.emergency_scoring import is_emergency_action_correct
from rl.q_agent import QAgent
from rl.reward import compute_reward
from rl.state_encoder import StateEncoder
from rl.state_features import StateFeatureExtractor

# ─── Configuration ─────────────────────────────────────

SUMO_BINARY = "sumo"
NET_FILE = os.path.join(os.path.dirname(__file__), "intersection.net.xml")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GREEN_DURATION = 5
YELLOW_DURATION = 2
ALL_RED_DURATION = 1

DEFAULT_EPISODES = 10
DEFAULT_DURATION = 300
DEFAULT_PATTERN = "medium"


# ═══════════════════════════════════════════════════════
# CONTROLLERS (same interface, now includes DQN)
# ═══════════════════════════════════════════════════════


class SUMODQNController:
    """Double DQN controller using trained neural network."""

    def __init__(self, agent=None):
        """
        agent: an already-constructed, already-trained DQNAgent to
        evaluate directly (its CURRENT in-memory weights, not whatever's
        on disk) — used by train_dqn_sumo.py's post-training
        catastrophic-forgetting check to evaluate the just-trained agent
        without a save/load round-trip. When None (default), constructs
        a fresh inference-mode agent and loads the on-disk checkpoint,
        matching this class's original behavior exactly.
        """
        self.name = "Double DQN"
        if agent is not None:
            self.agent = agent
            self._prior_epsilon = agent.epsilon
            self.agent.epsilon = 0.0  # greedy for evaluation, matching every other benchmark
        else:
            self.agent = DQNAgent(epsilon=0.0, inference_mode=True)
            self.agent.load()
        self.feature_extractor = StateFeatureExtractor()

    def choose_action(
        self,
        active_signal,
        queue_lengths,
        emergency_active,
        ambulance_zone,
        emergency_eta=0.0,
        distance=0.0,
    ):
        state = self.feature_extractor.extract(
            emergency_active=emergency_active,
            ambulance_zone=ambulance_zone if emergency_active else 0,
            active_signal=active_signal,
            queue_lengths=queue_lengths,
            emergency_eta=emergency_eta,
            distance=distance,
        )
        return self.agent.choose(state)


class SUMORLController:
    """Q-Learning controller using trained Q-table."""

    def __init__(self):
        self.name = "Q-Learning RL"
        self.agent = QAgent(epsilon=0.0)
        self.agent.load()
        self.encoder = StateEncoder()

    def choose_action(
        self,
        active_signal,
        queue_lengths,
        emergency_active,
        ambulance_zone,
        emergency_eta=0.0,
        distance=0.0,
    ):
        state = self.encoder.encode(
            emergency_active=emergency_active,
            ambulance_zone=ambulance_zone if emergency_active else 0,
            active_signal=active_signal,
            queue_lengths=queue_lengths,
            emergency_eta=emergency_eta,
        )
        return self.agent.choose(state)


class SUMOFixedController:
    def __init__(self):
        self.name = "Fixed Timer"
        self.cycle = [1, 2, 3, 4]
        self.index = 0
        self.steps = 0

    def choose_action(
        self,
        active_signal,
        queue_lengths,
        emergency_active,
        ambulance_zone,
        emergency_eta=0.0,
        distance=0.0,
    ):
        self.steps += 1
        if self.steps >= GREEN_DURATION:
            self.steps = 0
            self.index = (self.index + 1) % 4
            return self.cycle[self.index] - 1
        return 4


class SUMORuleBasedController:
    def __init__(self):
        self.name = "Rule-Based"

    def choose_action(
        self,
        active_signal,
        queue_lengths,
        emergency_active,
        ambulance_zone,
        emergency_eta=0.0,
        distance=0.0,
    ):
        if emergency_active and ambulance_zone > 0:
            return ambulance_zone - 1
        if queue_lengths:
            most = max(queue_lengths, key=queue_lengths.get)
            if queue_lengths[most] > 0:
                return most - 1
        return 4


class SUMORandomController:
    def __init__(self):
        self.name = "Random"

    def choose_action(
        self,
        active_signal,
        queue_lengths,
        emergency_active,
        ambulance_zone,
        emergency_eta=0.0,
        distance=0.0,
    ):
        return random.randint(0, 5)


class SUMOMaxPressureController:
    """SUMO-side mirror of controllers/max_pressure.py — see that file's
    docstring. Deliberately has NO ambulance-awareness, unlike
    SUMORuleBasedController, so it's a fair non-oracle peer for DQN/RL
    (design-review finding: Rule-Based hardcodes the reward function's own
    priorities and should be reported as an oracle reference, not ranked
    as a competitor)."""

    def __init__(self):
        self.name = "Max-Pressure"

    def choose_action(
        self,
        active_signal,
        queue_lengths,
        emergency_active,
        ambulance_zone,
        emergency_eta=0.0,
        distance=0.0,
    ):
        if not queue_lengths:
            return 5
        best_zone = max(queue_lengths, key=queue_lengths.get)
        if queue_lengths[best_zone] == 0:
            return 5
        if best_zone == active_signal:
            return 4
        return best_zone - 1


# ─── Safe Signal Transition ───────────────────────────


def safe_transition(adapter, from_zone, to_zone):
    if from_zone == to_zone:
        return
    adapter.set_yellow()
    for _ in range(YELLOW_DURATION):
        adapter.step()
    adapter.set_all_red()
    for _ in range(ALL_RED_DURATION):
        adapter.step()
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

    # Real per-vehicle ambulance delay (TraCI-measured), tracked the same
    # way as sumo_env.py: a live snapshot resets to 0 the instant the
    # ambulance is no longer detected, so we track the peak observed
    # during each continuous emergency and flush it into
    # ambulance_delay_events when the emergency ends, rather than reading
    # the live value once at episode end (which would almost always be 0).
    ambulance_waiting_time = 0.0
    current_emergency_peak_wait = 0.0
    ambulance_delay_events = []
    prev_amb_present = False

    adapter.set_signal(1)

    try:
        step = 0
        while step < duration:
            adapter.step()
            step += 1
            green_timer += 1

            queue_lengths = adapter.get_queue_lengths()
            queue_history.append(sum(queue_lengths.values()))

            amb_present, amb_zone = adapter.detect_ambulance()

            # Previously this fallback scan (which finds amb_zone when the
            # primary detect_ambulance() misses it) was the ONLY place
            # emergency_eta/distance were computed — but detect_ambulance()
            # already returns True for essentially the ambulance's entire
            # approach, so this branch (and therefore ETA/distance) almost
            # never ran. ETA/distance are now computed whenever the
            # ambulance is present at all, regardless of which path found
            # it, and distance is tracked alongside ETA (previously not
            # tracked at all in this script).
            if not amb_present:
                try:
                    for vid in traci.vehicle.getIDList():
                        if "ambulance" in vid.lower():
                            amb_present = True
                            road = traci.vehicle.getRoadID(vid)
                            if road in EDGE_TO_ZONE:
                                amb_zone = EDGE_TO_ZONE[road]
                            elif ambulance_zone:
                                amb_zone = ambulance_zone
                            else:
                                amb_zone = 3
                            break
                except Exception:
                    pass

            emergency_eta = 0.0
            emergency_distance = 0.0
            if amb_present:
                try:
                    for vid in traci.vehicle.getIDList():
                        if "ambulance" in vid.lower():
                            spd = traci.vehicle.getSpeed(vid)
                            pos = traci.vehicle.getLanePosition(vid)
                            lane_len = traci.lane.getLength(traci.vehicle.getLaneID(vid))
                            remaining = max(0, lane_len - pos)
                            emergency_eta = remaining / max(spd, 1.0)
                            emergency_distance = remaining
                            break
                except Exception:
                    emergency_eta = 10.0
                    emergency_distance = 0.0

                # Real per-vehicle waiting time (ground truth), same
                # peak-tracking pattern as sumo_env.py's fix.
                ambulance_waiting_time = adapter.get_ambulance_waiting_time()
                current_emergency_peak_wait = max(
                    current_emergency_peak_wait, ambulance_waiting_time
                )
            else:
                if prev_amb_present and current_emergency_peak_wait > 0:
                    ambulance_delay_events.append(current_emergency_peak_wait)
                current_emergency_peak_wait = 0.0
                ambulance_waiting_time = 0.0
            prev_amb_present = amb_present

            if amb_present and amb_zone > 0:
                if emergency_detected_step is None:
                    emergency_detected_step = step

            if green_timer >= GREEN_DURATION:
                green_timer = 0

                action = controller.choose_action(
                    active_signal=current_zone,
                    queue_lengths=queue_lengths,
                    emergency_active=amb_present,
                    ambulance_zone=amb_zone if amb_present else 0,
                    emergency_eta=emergency_eta,
                    distance=emergency_distance,
                )

                # ambulance_delay forwarded so DQN is evaluated on the same
                # reward it was trained to maximize (sumo_env.py's step()
                # passes this same TraCI-measured value) — previously this
                # call omitted it entirely, silently scoring every
                # controller (DQN included) on a reward function that
                # doesn't match what DQN's training actually optimized.
                reward = compute_reward(
                    action=action,
                    amb_zone=amb_zone if amb_present else 0,
                    prev_signal=current_zone,
                    traffic=pattern,
                    queue_lengths=queue_lengths,
                    ambulance_delay=ambulance_waiting_time if amb_present else None,
                )
                total_reward += reward

                if amb_present and amb_zone > 0:
                    emergency_total += 1
                    # current_zone is still pre-action here (the zone about
                    # to be extended), which is the correct served_zone
                    # argument for action==4/"extend".
                    if is_emergency_action_correct(action, amb_zone, current_zone):
                        emergency_correct += 1
                        if emergency_detected_step is not None:
                            resp_time = step - emergency_detected_step
                            emergency_response_times.append(resp_time)
                            emergency_detected_step = step

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

            for zone, edge in ZONE_TO_EDGE.items():
                try:
                    total_wait_time += traci.edge.getWaitingTime(edge)
                except Exception:
                    pass

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

    # Flush a still-in-progress emergency's peak wait (episode ended while
    # an ambulance was still present) — same end-of-episode handling as
    # sumo_env.py's get_metrics().
    if current_emergency_peak_wait > 0:
        ambulance_delay_events.append(current_emergency_peak_wait)
    avg_ambulance_delay = float(np.mean(ambulance_delay_events)) if ambulance_delay_events else 0.0

    return {
        "total_reward": round(total_reward, 2),
        "avg_queue_length": round(avg_queue, 2),
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
        # Real per-vehicle ambulance delay (TraCI ground truth), aggregated
        # across every ambulance that appeared this episode — see the
        # peak-tracking note above. This is what should be cited as "how
        # delayed were ambulances", not a live snapshot.
        "avg_ambulance_delay": round(avg_ambulance_delay, 2),
        "ambulance_count": len(ambulance_delay_events),
    }


def evaluate_controller(
    controller, episodes=10, pattern="medium", duration=300, ambulance_zone=3, use_gui=False
):
    print(f"\n  Evaluating: {controller.name}")
    print(f"  Episodes: {episodes}, Pattern: {pattern}, Duration: {duration}s")

    all_results = []
    for ep in range(episodes):
        seed = 42 + ep
        amb_time = random.Random(seed).randint(60, duration - 60)
        result = run_episode(
            controller=controller,
            pattern=pattern,
            duration=duration,
            ambulance_zone=ambulance_zone,
            ambulance_time=amb_time,
            seed=seed,
            use_gui=use_gui and ep == 0,
        )
        all_results.append(result)
        print(
            f"    Episode {ep+1}: reward={result['total_reward']:+.1f}, "
            f"queue={result['avg_queue_length']:.1f}, "
            f"emerg_acc={result['emergency_accuracy']:.0f}%, "
            f"amb_delay={result['avg_ambulance_delay']:.1f}s"
        )

    rewards = [r["total_reward"] for r in all_results]
    return {
        "name": controller.name,
        "episodes": episodes,
        "pattern": pattern,
        "reward": {
            "mean": round(np.mean(rewards), 2),
            "std": round(np.std(rewards), 2),
            "ci_95": round(1.96 * np.std(rewards) / np.sqrt(len(rewards)), 2),
            "per_episode": [round(r, 2) for r in rewards],
        },
        "ambulance_delay": {
            "mean": round(np.mean([r["avg_ambulance_delay"] for r in all_results]), 2),
            "total_ambulances": int(sum(r["ambulance_count"] for r in all_results)),
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


def run_comparison(episodes=10, pattern="medium", duration=300, use_gui=False):
    """
    Compare all 6 controllers in SUMO — the ONLY evaluation harness in
    this codebase where every controller (including DQN) is evaluated in
    the same environment it either trained in (DQN) or reasons directly
    against (everyone else), so this is the fair, apples-to-apples
    comparison table. The simplified-simulator harness
    (evaluation/comparator.py) is faster but evaluates SUMO-trained DQN
    out-of-distribution — see the "environment" field in this function's
    saved JSON output and backend/ARCHITECTURE.md's domain-gap discussion
    before treating numbers from the two harnesses as interchangeable.
    """
    print("=" * 70)
    print("RescueWay SUMO Evaluation v2 (with DQN)")
    print(f"Pattern: {pattern} | Episodes: {episodes} | Duration: {duration}s")
    print("=" * 70)

    controllers = [
        SUMODQNController(),
        SUMORLController(),
        SUMOFixedController(),
        SUMORuleBasedController(),  # oracle reference — see comparator.py's ORACLE_CONTROLLERS
        SUMOMaxPressureController(),  # fair non-oracle heuristic baseline
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

    print("\n" + "=" * 70)
    print("SUMO EVALUATION RESULTS (v2)")
    print("=" * 70)
    print(
        f"{'Controller':<20} {'Reward':>12} {'±95%CI':>8} {'Emergency':>12} "
        f"{'AmbDelay':>10} {'Queue':>10} {'Throughput':>12}"
    )
    print("-" * 86)

    for name, r in results.items():
        ci = r["reward"].get("ci_95", r["reward"]["std"])
        print(
            f"{name:<20} {r['reward']['mean']:>+12.1f} {ci:>8.1f} "
            f"{r['emergency']['accuracy']:>11.1f}% "
            f"{r['ambulance_delay']['mean']:>9.1f}s "
            f"{r['queue']['mean']:>10.1f} "
            f"{r['throughput']['mean']:>12.1f}"
        )

    print("=" * 74)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULTS_DIR, f"sumo_comparison_v2_{timestamp}.json")
    with open(result_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "version": "v2_with_dqn",
                # Explicit machine-readable provenance: every controller here
                # was trained (if applicable) and evaluated in the SAME
                # environment (SUMO microscopic simulation via TraCI), unlike
                # evaluation/comparator.py's results, which evaluate every
                # controller in a lower-fidelity simplified simulator that
                # SUMO-trained DQN never saw during training. A design-review
                # finding was that tables mixing the two without this tag were
                # easy to misread as one apples-to-apples ranking.
                "environment": ENV_TAG_SUMO,
                "oracle_controllers": ["Rule-Based"],
                "config": {"episodes": episodes, "pattern": pattern, "duration": duration},
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {result_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RescueWay SUMO Evaluation v2 (with DQN)")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        choices=["low", "medium", "high", "rush_hour"],
    )
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--dqn-only", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(NET_FILE):
        print(f"ERROR: Network file not found: {NET_FILE}")
        print("Run build_network.bat first.")
        sys.exit(1)

    if args.dqn_only:
        ctrl = SUMODQNController()
        result = evaluate_controller(
            ctrl,
            episodes=args.episodes,
            pattern=args.pattern,
            duration=args.duration,
            use_gui=args.gui,
        )
        print("\nDQN in SUMO:")
        print(json.dumps(result, indent=2))
    else:
        run_comparison(
            episodes=args.episodes,
            pattern=args.pattern,
            duration=args.duration,
            use_gui=args.gui,
        )
