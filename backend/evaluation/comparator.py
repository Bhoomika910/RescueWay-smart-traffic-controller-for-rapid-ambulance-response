# evaluation/comparator.py
import random

import numpy as np

from config import BENCHMARK_EMERGENCY_PROBABILITY, BENCHMARK_SEED, ENV_TAG_SIMPLIFIED
from controllers.dqn_controller import DQNController, DQNShieldedController
from controllers.fixed_timer import FixedTimerController
from controllers.max_pressure import MaxPressureController
from controllers.random_controller import RandomController
from controllers.rl_controller import RLController
from controllers.rule_based import RuleBasedController
from gps.zone_model import ZONE_CONFIG
from rl.ambulance_delay import estimate_clearance
from rl.emergency_scoring import is_emergency_action_correct
from rl.environment import TrafficEnv
from rl.reward import compute_reward

VALID_PATTERNS = ["low", "medium", "high", "rush_hour"]

# Controllers hardcoded with direct knowledge of this benchmark's reward
# function (Rule-Based: "always serve the ambulance zone first") are
# oracle/upper-bound REFERENCES, not fair competitors to a learned policy
# — DQN/Q-Learning can never legitimately "beat" a 100%-by-construction
# emergency-accuracy ceiling, so ranking them in the same competitive
# leaderboard as best_controller/best_per_metric/best_overall previously
# made every one of those fields a category error (design-review finding).
# Still fully evaluated and reported — just excluded from the ranking
# computations below and surfaced separately as oracle_reference.
ORACLE_CONTROLLERS = {"rule_based"}

# A controller only qualifies for the "fastest response" ranking if it
# resolved at least this fraction of the emergency events it saw. Without
# this floor, a controller that mostly ignores emergencies (e.g. Fixed
# Timer) can still win "fastest response" purely by luck: it only needs to
# stumble onto the correct zone a handful of times, early, to post a small
# average, while its overall emergency_accuracy correctly shows it as the
# worst controller. See ControllerComparator.get_summary().
MIN_RESPONSE_RESOLUTION_RATE = 0.5

# Multi-objective composite score weights. Reward alone rewards a
# hand-coded controller that was authored with direct knowledge of the
# reward function's own priorities (see Rule-Based) at least as much as a
# controller that has to learn a generalizable policy — so "highest
# avg_reward" is not the same question as "best overall controller".
# Weights are documented here so they're auditable/adjustable for research
# reporting. Each metric is min-max normalized across controllers before
# weighting. Direction (higher/lower is better) is stored alongside the
# weight — not in a separate hardcoded list — so adding a new metric here
# can't silently default to the wrong direction if someone forgets to
# update a second, disconnected place.
#
# avg_ambulance_delay carries the largest weight, matching the project's
# primary objective — minimizing ambulance delay, not just eventually
# giving it the correct signal — which is a materially different question
# from emergency_accuracy (a per-tick "was the right zone green" hit
# rate that says nothing about whether the ambulance actually stopped).
# The remaining weights are the previous set, rescaled proportionally
# (×0.7) to make room.
COMPOSITE_WEIGHTS = {
    # field: (weight, higher_is_better)
    "avg_ambulance_delay": (0.30, False),  # primary objective: seconds of predicted/actual delay
    "emergency_accuracy": (0.21, True),  # correctly prioritizing ambulances
    "avg_reward": (0.14, True),  # overall multi-objective reward signal
    "avg_queue_length": (0.11, False),  # fewer vehicles waiting
    "avg_throughput": (0.11, True),  # vehicles cleared
    "avg_wait_time": (0.07, False),  # less time spent waiting
    "fairness_index": (0.06, True),  # Jain's index across zone green time
}

# Short, factual profiles for research-report context — not a substitute
# for the numbers, but an explanation of *why* a controller scores the way
# it does structurally.
CONTROLLER_PROFILES = {
    "dqn": (
        "Learned policy (Double DQN) trained via gradient descent on a "
        "19-dim continuous state. Generalizes to state combinations it "
        "wasn't explicitly programmed for and can be retrained as "
        "conditions change; does not have hand-tuned rules baked in, so "
        "it must approximate rather than guarantee the reward function's "
        "priorities."
    ),
    "rl": (
        "Tabular Q-Learning baseline. Learns from direct experience like "
        "DQN, but with a discretized state space and no function "
        "approximation, so it can't generalize to unseen state "
        "combinations the way DQN can."
    ),
    "fixed": (
        "Blind round-robin cycle with no awareness of queues or "
        "emergencies at all. Included as the traditional-traffic-light "
        "sanity floor every other controller should beat."
    ),
    "rule_based": (
        "ORACLE REFERENCE, not a competitive baseline — see "
        "ORACLE_CONTROLLERS. Hand-coded if/elif priority rules authored "
        "with direct knowledge of this benchmark's reward function "
        "(always serve the ambulance zone, always serve the most "
        "congested zone). This lets it reap near-maximal reward on the "
        "emergency and congestion terms by construction rather than by "
        "learning — but it cannot adapt to situations its author didn't "
        "anticipate (e.g. simultaneous multi-zone emergencies, novel "
        "traffic distributions) without someone rewriting its rules. "
        'Reported as an upper-bound reference ("how close does the '
        'learned policy get to a hand-tuned ceiling"), excluded from '
        "best_controller/best_per_metric/best_overall."
    ),
    "dqn_shielded": (
        "The SAME trained Double DQN policy as 'dqn' above, with the "
        "emergency safety shield (rl/safety_shield.py) applied on top — "
        "the same hard override services/traffic_controller.py has "
        "applied in production since it was added. Compare this against "
        "'dqn' to see exactly what the shield buys: 'dqn' measures the "
        "learned policy's own emergency-handling competence in "
        "isolation; 'dqn_shielded' is what production actually "
        "guarantees a rider experiences."
    ),
    "max_pressure": (
        "Max-Pressure backpressure control (Varaiya 2013) — a genuine "
        "non-oracle heuristic baseline. Optimizes queue pressure only, "
        "with NO ambulance-awareness and no knowledge of this benchmark's "
        "reward function; any emergency performance it shows comes only "
        "from happening to serve a congested zone that an ambulance is "
        "also in. This is the fair peer for DQN/Q-Learning: beating it "
        "means the learned policy actually learned to balance congestion "
        "management against emergency response, not that it was handed "
        "the answer."
    ),
    "random": (
        "Uniform random action every step. Sanity check — every other "
        "controller should significantly outperform this."
    ),
}


class ControllerComparator:
    def __init__(self, seed=None):
        self.controllers = {
            "dqn": DQNController(),  # PRIMARY (Double DQN), RAW policy — no safety shield
            "dqn_shielded": DQNShieldedController(),  # same policy + the emergency safety shield production actually uses
            "rl": RLController(),  # BASELINE (Q-Learning)
            "fixed": FixedTimerController(),
            "rule_based": RuleBasedController(),  # oracle reference — see ORACLE_CONTROLLERS
            "max_pressure": MaxPressureController(),  # fair non-oracle heuristic baseline
            "random": RandomController(),
        }
        self.results = {}
        self.seed = seed if seed is not None else BENCHMARK_SEED

    def run_comparison(
        self,
        episodes=100,
        max_steps=30,
        emergency_probability=BENCHMARK_EMERGENCY_PROBABILITY,
        pattern=None,
    ):
        """
        pattern: one of VALID_PATTERNS to pin every controller to the same
        fixed traffic pattern for the whole run, or None/"random" to let
        each episode randomly (but identically, per the shared seed) pick
        a pattern, as before.
        """
        if pattern not in VALID_PATTERNS:
            pattern = None

        print(f"Running comparison: {episodes} episodes × {max_steps} steps")
        print(
            f"Emergency probability: {emergency_probability} | Pattern: {pattern or 'random (seeded)'}"
        )
        print("=" * 70)

        self.results = {}

        for ctrl_name, controller in self.controllers.items():
            controller.reset_stats()
            metrics = self._evaluate_controller(
                controller, episodes, max_steps, emergency_probability, pattern
            )
            self.results[ctrl_name] = metrics
            print(
                f"  {controller.name:20s} | "
                f"Reward: {metrics['avg_reward']:8.2f} | "
                f"EmergAcc: {metrics['emergency_accuracy']:5.1f}% | "
                f"EmergResp: {metrics['avg_emergency_response_steps']:4.1f} steps | "
                f"Queue: {metrics['avg_queue_length']:6.2f} | "
                f"Throughput: {metrics['avg_throughput']:6.1f}"
            )

        print("=" * 70)

        # Compute DQN improvements (see _reward_gain for why a percentage
        # isn't always reported — reward can be negative, and "% better
        # than a near-zero or negative baseline" is not a meaningful
        # statistic).
        dqn_reward = self.results["dqn"]["avg_reward"]
        for name, metrics in self.results.items():
            if name != "dqn":
                gain = self._reward_gain(dqn_reward, metrics["avg_reward"])
                metrics["dqn_improvement"] = (
                    gain["improvement_pct"] if gain["improvement_pct"] is not None else 0
                )
                metrics["dqn_absolute_gain"] = gain["absolute_gain"]
            else:
                metrics["dqn_improvement"] = 0
                metrics["dqn_absolute_gain"] = 0

        return self.results

    def _evaluate_controller(
        self, controller, episodes, max_steps, emergency_probability, pattern=None
    ):
        all_rewards = []
        all_queue_lengths = []
        all_throughputs = []
        all_wait_times = []
        emergency_correct = 0
        emergency_total = 0
        emergency_events_total = 0
        emergency_events_resolved = 0
        emergency_response_steps_list = []
        total_switches = 0
        zone_green_counts = {1: 0, 2: 0, 3: 0, 4: 0}

        # Ambulance-delay tracking (the primary research objective: did the
        # ambulance actually clear the intersection before arriving, not
        # just "was the right zone eventually green"). One entry per
        # concluded ambulance trip.
        all_ambulance_delays = []  # seconds of predicted delay per trip (0 = no stop)
        ambulance_stop_count = 0
        all_lead_time_available = []  # lead_time_available computed AT SPAWN, using the full ETA

        for ep in range(episodes):
            # Deterministic seed per episode index — same seed = same
            # traffic pattern and vehicle-arrival sequence for every
            # controller, so comparisons are fair. Controllers that draw
            # extra randomness (RL/DQN epsilon-greedy) must skip that draw
            # in inference mode (epsilon=0) or they desync from this
            # stream relative to non-RNG-consuming controllers — see
            # QAgent.choose()/DQNAgent.choose().
            episode_seed = self.seed + ep
            np.random.seed(episode_seed)
            random.seed(episode_seed)

            # The emergency-event schedule gets its OWN RNG, independent
            # of the global `random`/`np.random` streams that controllers'
            # choose_action() may consume from (e.g. RandomController
            # draws random.randint(0, 5) every step from the same global
            # `random` module the block below used to use — that extra
            # draw silently shifted every subsequent random.random()/
            # random.choice() call, giving Random a *different* emergency
            # schedule than Fixed/Rule-Based/DQN/RL under the "same" seed,
            # even though the seeding call above looked identical for
            # everyone). Isolating environment stochasticity from agent
            # stochasticity guarantees every controller sees the exact
            # same sequence of emergencies regardless of how much
            # randomness its own decision logic happens to consume.
            emergency_rng = random.Random(episode_seed)

            # Reset any leftover cycling/timing state from the previous
            # episode (or a previous benchmark run — controllers are
            # long-lived singletons) so every episode starts from the same
            # controller state, not wherever the last episode left off.
            controller.reset_episode_state()

            env = TrafficEnv()
            env.reset()
            if pattern:
                # Pin every controller to the same explicit pattern
                # instead of the per-episode random choice made inside
                # env.reset() (still deterministic/shared via the seed
                # above, but an explicit pattern makes the comparison
                # driven by the user's actual selection).
                env.simulator.set_traffic_pattern(pattern)

            episode_reward = 0
            prev_action = -1

            # Force emergency events during benchmark
            emergency_active = False
            emergency_zone = 0
            emergency_timer = 0
            emergency_wait_steps = 0
            emergency_responded = False

            for step in range(max_steps):
                metrics = env.simulator.get_metrics()
                queue_lengths = {z: metrics[z].queue_length for z in [1, 2, 3, 4]}
                system = env.simulator.get_system_metrics()

                # Generate emergency events with higher probability for fair testing.
                # emergency_timer is seeded from the same per-zone ETA the live
                # system uses (gps/zone_model.ZONE_CONFIG), not an arbitrary
                # small window — this gives the benchmark a realistic mix of
                # "far, plenty of lead time" and "close, almost no lead time"
                # trips, which is what actually differentiates a controller
                # that proactively clears a queue from one that only reacts.
                if not emergency_active:
                    if emergency_rng.random() < emergency_probability:
                        emergency_active = True
                        emergency_zone = emergency_rng.choice([1, 2, 3, 4])
                        emergency_timer = float(ZONE_CONFIG[emergency_zone]["eta_seconds"])
                        emergency_wait_steps = 0
                        emergency_responded = False
                        emergency_events_total += 1
                        spawn_lead_time = estimate_clearance(
                            remaining_eta=emergency_timer,
                            queue_length=queue_lengths.get(emergency_zone, 0),
                            zone_is_served=(env.active_signal == emergency_zone),
                        )["lead_time_available"]
                else:
                    emergency_timer -= 1
                    emergency_wait_steps += 1

                # Bug fix (technical audit): previously only the 4 base
                # args were passed here, so DQN/RL never received ETA or
                # distance during benchmark evaluation even though DQN's
                # SUMO training and the live TrafficController both do —
                # a real train/eval state-distribution mismatch. emergency_timer
                # IS the current remaining ETA (already tracked above);
                # distance comes from the same per-zone ZONE_CONFIG the live
                # system uses.
                action = controller.choose_action(
                    active_signal=env.active_signal,
                    queue_lengths=queue_lengths,
                    emergency_active=emergency_active,
                    ambulance_zone=emergency_zone,
                    emergency_eta=emergency_timer if emergency_active else 0.0,
                    distance=(
                        float(ZONE_CONFIG[emergency_zone]["avg_distance"])
                        if emergency_active
                        else 0.0
                    ),
                )

                ambulance_delay = None
                if emergency_active:
                    ambulance_delay = estimate_clearance(
                        remaining_eta=emergency_timer,
                        queue_length=queue_lengths.get(emergency_zone, 0),
                        zone_is_served=(env.active_signal == emergency_zone),
                    )["deficit"]

                # Compute reward with proper emergency context
                reward = compute_reward(
                    action=action,
                    amb_zone=emergency_zone if emergency_active else 0,
                    prev_signal=env.active_signal,
                    traffic=env.simulator.traffic_pattern,
                    queue_lengths=queue_lengths,
                    ambulance_delay=ambulance_delay,
                )

                # Apply action to env
                env.active_signal = action + 1 if action < 4 else env.active_signal
                env.simulator.step(
                    active_signal=env.active_signal, is_green=True, current_time=step
                )

                episode_reward += reward

                all_queue_lengths.append(sum(queue_lengths.values()))
                all_wait_times.append(system.get("avg_wait_time_all", 0))
                zone_green_counts[env.active_signal] = (
                    zone_green_counts.get(env.active_signal, 0) + 1
                )

                # Track switches
                if action != prev_action and action < 4:
                    total_switches += 1
                prev_action = action

                # Track emergency performance for THIS tick before clearing
                # the event below — the arrival tick (emergency_timer<=0)
                # must still count here, since the controller's action this
                # tick was made while the emergency was active and IS the
                # action that determines whether the ambulance stops.
                #
                # emergency_total/correct are per-step (every step a
                # controller is exposed to an active emergency, was it
                # giving the right zone the green?) — the correct
                # denominator for a hit-rate percentage.
                # emergency_events_total/resolved and
                # emergency_response_steps_list are per-EVENT: how many
                # distinct emergencies occurred, how many were ever
                # correctly served, and — for those — how many steps it
                # took to first respond correctly. Recording only the
                # FIRST correct step per event (via emergency_responded)
                # avoids counting one event's sustained correctness as
                # several independent "responses".
                if emergency_active and emergency_zone > 0:
                    emergency_total += 1
                    # env.active_signal was already updated above (line 336)
                    # to reflect this tick's served zone, so it's the
                    # correct served_zone for both "switch to N" and
                    # "extend" actions. See rl/emergency_scoring.py for why
                    # this shared helper exists instead of nine local copies.
                    if is_emergency_action_correct(action, emergency_zone, env.active_signal):
                        emergency_correct += 1
                        if not emergency_responded:
                            emergency_response_steps_list.append(emergency_wait_steps)
                            emergency_responded = True
                            emergency_events_resolved += 1

                if emergency_active and emergency_timer <= 0:
                    # Trip concludes this step — record the final ambulance
                    # -delay outcome (recomputed against the post-action
                    # signal state) and close out the event.
                    post_step_metrics = env.simulator.get_metrics()
                    final_queue = post_step_metrics[emergency_zone].queue_length
                    final = estimate_clearance(
                        remaining_eta=0.0,
                        queue_length=final_queue,
                        zone_is_served=(env.active_signal == emergency_zone),
                    )
                    all_ambulance_delays.append(final["deficit"])
                    all_lead_time_available.append(spawn_lead_time)
                    if final["will_stop"]:
                        ambulance_stop_count += 1
                    emergency_active = False
                    emergency_zone = 0

            all_rewards.append(episode_reward)
            all_throughputs.append(env.simulator.get_system_metrics()["total_vehicles_cleared"])

        # Jain's fairness index over per-zone green-time distribution.
        green_values = list(zone_green_counts.values())
        if sum(green_values) > 0:
            n = len(green_values)
            fairness_index = (sum(green_values) ** 2) / (n * sum(v**2 for v in green_values))
        else:
            fairness_index = 1.0

        return {
            "name": controller.name,
            "avg_reward": round(float(np.mean(all_rewards)), 2),
            "std_reward": round(float(np.std(all_rewards)), 2),
            "min_reward": round(float(np.min(all_rewards)), 2),
            "max_reward": round(float(np.max(all_rewards)), 2),
            "avg_queue_length": round(float(np.mean(all_queue_lengths)), 2),
            "avg_throughput": round(float(np.mean(all_throughputs)), 2),
            "avg_wait_time": round(float(np.mean(all_wait_times)), 2) if all_wait_times else 0,
            "fairness_index": round(fairness_index, 3),
            "zone_green_distribution": dict(zone_green_counts),
            "emergency_accuracy": round(
                (emergency_correct / emergency_total * 100) if emergency_total > 0 else 0, 1
            ),
            "emergency_correct": emergency_correct,
            "emergency_total": emergency_total,
            "emergency_events_total": emergency_events_total,
            "emergency_events_resolved": emergency_events_resolved,
            "emergency_resolution_rate": round(
                (
                    (emergency_events_resolved / emergency_events_total * 100)
                    if emergency_events_total > 0
                    else 0
                ),
                1,
            ),
            "avg_emergency_response_steps": round(
                (
                    float(np.mean(emergency_response_steps_list))
                    if emergency_response_steps_list
                    else 0
                ),
                2,
            ),
            "total_switches": total_switches,
            "avg_switches_per_episode": round(total_switches / episodes, 2),
            "dqn_improvement": 0,
            # ─── Ambulance-delay metrics (primary research objective) ───
            # avg_ambulance_delay: mean predicted delay in seconds across
            # every concluded ambulance trip (0 = passed through without
            # stopping). This is the headline metric — "was the correct
            # zone eventually green" (emergency_accuracy above) does not
            # imply "did the ambulance avoid stopping"; this does.
            "ambulance_trips_total": len(all_ambulance_delays),
            "avg_ambulance_delay": (
                round(float(np.mean(all_ambulance_delays)), 2) if all_ambulance_delays else 0
            ),
            "ambulance_stops_per_trip": (
                round(ambulance_stop_count / len(all_ambulance_delays), 3)
                if all_ambulance_delays
                else 0
            ),
            "pct_ambulances_no_stop": (
                round((1 - ambulance_stop_count / len(all_ambulance_delays)) * 100, 1)
                if all_ambulance_delays
                else 0
            ),
            # avg_time_to_clear_before_arrival: mean lead_time_available
            # computed AT SPAWN (using the full initial ETA) — positive
            # means the controller had that many seconds of margin to
            # spare on average; negative means it was structurally behind
            # schedule even before it made a single decision for that trip.
            "avg_time_to_clear_before_arrival": (
                round(float(np.mean(all_lead_time_available)), 2) if all_lead_time_available else 0
            ),
            # overall_traffic_efficiency: throughput achieved per unit of
            # standing queue — a controller that clears vehicles quickly
            # without letting queues build up scores higher than one that
            # achieves the same throughput only by keeping big backlogs
            # moving.
            "overall_traffic_efficiency": round(
                float(np.mean(all_throughputs)) / (float(np.mean(all_queue_lengths)) + 1), 3
            ),
        }

    def get_summary(self):
        if not self.results:
            return {"error": "No comparison run yet"}

        summary = {
            # Explicit machine-readable provenance (design-review finding):
            # this harness evaluates every controller — DQN included —
            # against traffic/simulator.py's simplified, non-SUMO physics.
            # DQN trains exclusively in SUMO (sumo/train_dqn_sumo.py), so
            # its numbers here reflect an out-of-distribution evaluation;
            # every other controller reasons directly against whichever
            # environment evaluates it and doesn't have this caveat. For
            # the fair, same-environment comparison, see
            # sumo/evaluate_in_sumo_v2.py's results (environment: "sumo").
            "environment": ENV_TAG_SIMPLIFIED,
            "environment_caveat": (
                "DQN is trained in SUMO and evaluated here in a different, "
                "lower-fidelity simulator — these numbers should not be "
                "read as DQN's real-world quality without also consulting "
                "the SUMO-native comparison (evaluate_in_sumo_v2.py)."
            ),
            "controllers": {},
            "oracle_controllers": sorted(ORACLE_CONTROLLERS & set(self.results)),
            "best_controller": "",
            "best_per_metric": {},
            "oracle_reference": {},
            "rl_improvements": {},
            "controller_profiles": CONTROLLER_PROFILES,
        }

        # Every controller's raw numbers are still reported in full — only
        # the *ranking* computations below exclude oracle controllers.
        for name, metrics in self.results.items():
            summary["controllers"][name] = metrics

        # Oracle controllers (Rule-Based) are surfaced separately as an
        # upper-bound reference rather than folded into the competitive
        # rankings — see ORACLE_CONTROLLERS' module-level docstring.
        for name in ORACLE_CONTROLLERS & set(self.results):
            summary["oracle_reference"][name] = self.results[name]

        # All ranking computations below are scoped to non-oracle
        # controllers only.
        rankable = {k: v for k, v in self.results.items() if k not in ORACLE_CONTROLLERS}

        best_reward = float("-inf")
        for name, metrics in rankable.items():
            if metrics["avg_reward"] > best_reward:
                best_reward = metrics["avg_reward"]
                summary["best_controller"] = metrics["name"]

        # A controller only qualifies for "fastest response" if it
        # resolved a large-enough share of the emergencies it saw — see
        # MIN_RESPONSE_RESOLUTION_RATE. Otherwise a controller that
        # ignores emergencies (e.g. Fixed Timer) could win this card by
        # sheer luck on a handful of early, coincidentally-correct steps
        # while simultaneously scoring worst on "best_emergency".
        qualified_for_response = [
            k
            for k in rankable
            if rankable[k]["emergency_events_resolved"] > 0
            and rankable[k]["emergency_resolution_rate"] >= MIN_RESPONSE_RESOLUTION_RATE * 100
        ]

        # Only rank "lowest ambulance delay" among controllers that actually
        # had at least one concluded ambulance trip recorded — guards
        # against a degenerate run (very few episodes / low emergency
        # probability) where no ambulance ever spawned for a controller,
        # which would otherwise show a misleading "0.00s delay" default.
        qualified_for_delay = [k for k in rankable if rankable[k]["ambulance_trips_total"] > 0]

        # Best per metric (oracle controllers excluded — see rankable above)
        summary["best_per_metric"] = {
            "lowest_ambulance_delay": (
                min(qualified_for_delay, key=lambda k: rankable[k]["avg_ambulance_delay"])
                if qualified_for_delay
                else None
            ),
            "highest_reward": (
                max(rankable, key=lambda k: rankable[k]["avg_reward"]) if rankable else None
            ),
            "lowest_queue": (
                min(rankable, key=lambda k: rankable[k]["avg_queue_length"]) if rankable else None
            ),
            "best_emergency": (
                max(rankable, key=lambda k: rankable[k]["emergency_accuracy"]) if rankable else None
            ),
            "fastest_response": (
                min(
                    qualified_for_response,
                    key=lambda k: rankable[k]["avg_emergency_response_steps"],
                )
                if qualified_for_response
                else None
            ),
            "highest_throughput": (
                max(rankable, key=lambda k: rankable[k]["avg_throughput"]) if rankable else None
            ),
            "best_overall": (
                max(rankable, key=lambda k: self._composite_score(rankable[k]))
                if rankable
                else None
            ),
        }

        # Composite multi-objective score, exposed per-controller (oracle
        # controllers included here since it's informative context, just
        # excluded from best_overall above) so the basis for "best_overall"
        # is visible rather than a black box.
        for name, metrics in self.results.items():
            metrics["composite_score"] = round(self._composite_score(metrics), 4)

        summary["composite_score_sensitivity"] = self.weight_sensitivity_sweep()

        # DQN improvements over each baseline (Rule-Based kept here
        # deliberately, labeled as oracle context, not a ranking claim —
        # "how much reward would DQN need to gain to match a hand-tuned
        # ceiling" is a legitimate thing to report, just not a competitive
        # "DQN beat/lost to Rule-Based" framing).
        dqn_reward = self.results["dqn"]["avg_reward"]
        for name in ["rl", "fixed", "rule_based", "max_pressure", "random"]:
            if name in self.results:
                baseline = self.results[name]["avg_reward"]
                gain = self._reward_gain(dqn_reward, baseline)
                summary["rl_improvements"][name] = {
                    "dqn_reward": dqn_reward,
                    "baseline_reward": baseline,
                    "absolute_gain": gain["absolute_gain"],
                    "improvement_pct": gain["improvement_pct"],
                    "is_oracle": name in ORACLE_CONTROLLERS,
                }

        return summary

    @staticmethod
    def _reward_gain(dqn_reward: float, baseline: float) -> dict:
        """
        Report DQN's reward advantage over a baseline as an
        always-well-defined absolute point gain, plus a percentage that is
        only computed when it's actually meaningful.

        "% improvement" = (new - old) / |old| * 100 blows up or becomes
        nonsensical when the baseline is at or near zero (division by a
        tiny number) or negative (going from -15 to +464 isn't cleanly
        "3099% better" — the sign flip breaks the usual growth-rate
        reading of a percentage). This reward function routinely produces
        negative averages for weak controllers (Fixed Timer, Random), so
        the percentage is only reported when the baseline is positive and
        far enough from zero for the ratio to be stable; otherwise
        improvement_pct is None and callers should fall back to
        absolute_gain, which is always valid.
        """
        absolute_gain = round(dqn_reward - baseline, 2)
        if baseline > 10:
            improvement_pct = round((dqn_reward - baseline) / baseline * 100, 1)
        else:
            improvement_pct = None
        return {"absolute_gain": absolute_gain, "improvement_pct": improvement_pct}

    def weight_sensitivity_sweep(self, perturbation=0.5):
        """
        Sensitivity analysis on the composite score's weights
        (design-review finding): COMPOSITE_WEIGHTS' specific values were
        previously justified only by prose, with no check on whether
        best_overall is an artifact of that one particular choice. This
        perturbs each weight ±`perturbation` (default ±50%) one at a time,
        holding the others fixed, and reports whether best_overall
        changes under any of those perturbations — the composite score
        should be presented as "one linear scalarization, sensitivity-
        tested" (per-metric numbers remain the primary evidence), not as
        an unweighted, unquestioned ranking.
        """
        if not self.results:
            return {"error": "No comparison run yet"}

        rankable = {k: v for k, v in self.results.items() if k not in ORACLE_CONTROLLERS}
        if not rankable:
            return {"error": "No non-oracle controllers to rank"}

        def score_with_weights(metrics, weights):
            score = 0.0
            for field, (weight, higher_is_better) in weights.items():
                values = [m[field] for m in rankable.values()]
                lo, hi = min(values), max(values)
                v = metrics[field]
                if hi == lo:
                    normalized = 1.0
                elif higher_is_better:
                    normalized = (v - lo) / (hi - lo)
                else:
                    normalized = (hi - v) / (hi - lo)
                score += weight * normalized
            return score

        baseline_best = max(
            rankable, key=lambda k: score_with_weights(rankable[k], COMPOSITE_WEIGHTS)
        )

        perturbations = []
        for field in COMPOSITE_WEIGHTS:
            for direction, factor in [
                ("higher", 1 + perturbation),
                ("lower", max(0.0, 1 - perturbation)),
            ]:
                perturbed_weights = dict(COMPOSITE_WEIGHTS)
                old_weight, higher_is_better = perturbed_weights[field]
                perturbed_weights[field] = (old_weight * factor, higher_is_better)

                best_under_perturbation = max(
                    rankable, key=lambda k: score_with_weights(rankable[k], perturbed_weights)
                )
                perturbations.append(
                    {
                        "weight_perturbed": field,
                        "direction": direction,
                        "original_weight": old_weight,
                        "perturbed_weight": round(old_weight * factor, 4),
                        "best_overall_under_perturbation": best_under_perturbation,
                        "ranking_changed": best_under_perturbation != baseline_best,
                    }
                )

        n_changed = sum(1 for p in perturbations if p["ranking_changed"])
        return {
            "baseline_best_overall": baseline_best,
            "perturbation_magnitude": perturbation,
            "n_perturbations_tested": len(perturbations),
            "n_perturbations_that_changed_ranking": n_changed,
            "ranking_is_robust": n_changed == 0,
            "details": perturbations,
        }

    def _composite_score(self, metrics: dict) -> float:
        """
        Multi-objective composite score (0-1) combining reward, queue,
        throughput, wait time, emergency accuracy, and fairness — see
        COMPOSITE_WEIGHTS. Reward alone favors a hand-coded controller
        written with direct knowledge of the reward function's structure
        (see Rule-Based); this gives a rounder picture of "best overall
        controller" for research reporting, without hiding the raw
        per-metric numbers (still reported alongside it).
        """
        score = 0.0
        for field, (weight, higher_is_better) in COMPOSITE_WEIGHTS.items():
            values = [m[field] for m in self.results.values()]
            lo, hi = min(values), max(values)
            v = metrics[field]
            if hi == lo:
                normalized = 1.0
            elif higher_is_better:
                normalized = (v - lo) / (hi - lo)
            else:
                normalized = (hi - v) / (hi - lo)
            score += weight * normalized
        return score
