# evaluation/reward_sensitivity.py
"""
Reward-weight sensitivity analysis.

Design-review finding (category 8 "Reward design"): "weights are all
hand-tuned magic numbers ... with no principled derivation, sensitivity
analysis" — mirrors the composite-score weight-sensitivity sweep already
implemented for the benchmark's evaluation weights
(evaluation/comparator.py's weight_sensitivity_sweep()), applied here to
rl/reward.py's own weights instead.

What this checks, and what it deliberately does NOT check: perturbing a
reward weight and then asking "does the trained policy change" would
require a full retrain per perturbed weight (17 weights x +/-50% = 34
retrains) — far too expensive to run as a matter of course, and the
result would be confounded by training-run seed variance on top of the
weight change itself (see ARCHITECTURE.md's DQN seed-variance findings —
16.2pp std at this project's training budget, which would swamp a modest
reward-shaping effect). Instead, this asks the CHEAPER, still-meaningful
question: does the reward function's own QUALITATIVE ORDERING across a
battery of representative (state, action) scenarios stay stable under
each individual weight's plausible range of disagreement? If "correct
zone beats wrong zone" or "serving the congested zone beats serving an
empty one" flips sign under a +/-50% perturbation of some other, seemingly
unrelated weight, that is a real design fragility worth knowing about
even before touching a training run. If every qualitative ordering the
reward function is supposed to encode survives every single-weight
perturbation, that is real (if more modest) evidence the reward design is
not fragile to the exact magic-number choices — the honest scope of what
a no-retrain sensitivity analysis can and can't tell you.

Usage:
    python -m evaluation.reward_sensitivity
Output:
    results/reward_sensitivity/reward_sensitivity_<timestamp>.json
"""
import json
import os
import time

from rl.reward import DEFAULT_WEIGHTS, compute_reward

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "reward_sensitivity")
os.makedirs(RESULTS_DIR, exist_ok=True)

PERTURBATION_FACTORS = [0.5, 1.5]  # -50% / +50%, matching comparator.py's own sweep magnitude


# Each scenario is (name, kwargs) plus an `ordering` list of (label, kwargs)
# pairs whose compute_reward() outputs must stay in DESCENDING order for
# the scenario to "pass" — this is the qualitative property being checked,
# not an exact numeric target.
def _scenarios():
    congested_queues = {1: 25, 2: 3, 3: 2, 4: 5}
    balanced_queues = {1: 8, 2: 7, 3: 8, 4: 7}
    return [
        {
            "name": "emergency_zone_correctness",
            "description": "Correct emergency zone > passive normal-cycle > actively wrong zone",
            "ordering": [
                ("correct_zone", dict(action=0, amb_zone=1, prev_signal=3, traffic="medium")),
                ("normal_cycle", dict(action=5, amb_zone=1, prev_signal=3, traffic="medium")),
                (
                    "actively_wrong_zone",
                    dict(action=2, amb_zone=1, prev_signal=3, traffic="medium"),
                ),
            ],
        },
        {
            "name": "emergency_no_stop_vs_stop",
            "description": "No-stop outcome (delay<=0.5s) beats any nonzero measured delay",
            "ordering": [
                (
                    "no_stop",
                    dict(
                        action=0, amb_zone=1, prev_signal=3, traffic="medium", ambulance_delay=0.0
                    ),
                ),
                (
                    "delayed_5s",
                    dict(
                        action=0, amb_zone=1, prev_signal=3, traffic="medium", ambulance_delay=5.0
                    ),
                ),
                (
                    "delayed_15s",
                    dict(
                        action=0, amb_zone=1, prev_signal=3, traffic="medium", ambulance_delay=15.0
                    ),
                ),
            ],
        },
        {
            "name": "congestion_serve_most_congested",
            "description": "Serving the most congested zone beats serving an empty one",
            "ordering": [
                (
                    "serve_congested",
                    dict(
                        action=0,
                        amb_zone=0,
                        prev_signal=2,
                        traffic="high",
                        queue_lengths=congested_queues,
                    ),
                ),
                (
                    "serve_empty",
                    dict(
                        action=2,
                        amb_zone=0,
                        prev_signal=2,
                        traffic="high",
                        queue_lengths=congested_queues,
                    ),
                ),
            ],
        },
        {
            "name": "fairness_balanced_vs_starved",
            "description": "Serving a starved zone under severe imbalance beats ignoring it",
            "ordering": [
                (
                    "serve_starved",
                    dict(
                        action=0,
                        amb_zone=0,
                        prev_signal=2,
                        traffic="high",
                        queue_lengths={1: 40, 2: 1, 3: 1, 4: 1},
                    ),
                ),
                (
                    "ignore_starved",
                    dict(
                        action=1,
                        amb_zone=0,
                        prev_signal=2,
                        traffic="high",
                        queue_lengths={1: 40, 2: 1, 3: 1, 4: 1},
                    ),
                ),
            ],
        },
        {
            "name": "switching_cost_vs_maintain",
            "description": "Maintaining the current (already-fine) green beats an unnecessary switch",
            "ordering": [
                (
                    "maintain",
                    dict(
                        action=1,
                        amb_zone=0,
                        prev_signal=2,
                        traffic="medium",
                        queue_lengths=balanced_queues,
                    ),
                ),
                (
                    "switch",
                    dict(
                        action=3,
                        amb_zone=0,
                        prev_signal=2,
                        traffic="medium",
                        queue_lengths=balanced_queues,
                    ),
                ),
            ],
        },
        {
            "name": "extend_normal_parity",
            "description": "Extend and Normal-cycle score identically to an explicit re-selection "
            "of the same zone (non-emergency) — the reward-hacking loophole fix",
            "ordering": None,  # equality check, not an ordering — handled specially below
            "equality_group": [
                (
                    "extend",
                    dict(
                        action=4,
                        amb_zone=0,
                        prev_signal=1,
                        traffic="high",
                        queue_lengths=congested_queues,
                    ),
                ),
                (
                    "normal",
                    dict(
                        action=5,
                        amb_zone=0,
                        prev_signal=1,
                        traffic="high",
                        queue_lengths=congested_queues,
                    ),
                ),
                (
                    "explicit_reselect",
                    dict(
                        action=0,
                        amb_zone=0,
                        prev_signal=1,
                        traffic="high",
                        queue_lengths=congested_queues,
                    ),
                ),
            ],
        },
    ]


def _check_scenario(scenario, weights):
    if scenario.get("equality_group"):
        values = [compute_reward(**kw, weights=weights) for _, kw in scenario["equality_group"]]
        return len(set(values)) == 1
    values = [compute_reward(**kw, weights=weights) for _, kw in scenario["ordering"]]
    return all(a > b for a, b in zip(values, values[1:]))


def run_sensitivity_analysis():
    scenarios = _scenarios()
    weight_names = [f for f in DEFAULT_WEIGHTS.as_dict()]

    print("=" * 70)
    print("REWARD-WEIGHT SENSITIVITY ANALYSIS")
    print(
        f"Weights: {len(weight_names)} | Scenarios: {len(scenarios)} | "
        f"Perturbations per weight: {PERTURBATION_FACTORS}"
    )
    print("=" * 70)

    baseline_results = {s["name"]: _check_scenario(s, DEFAULT_WEIGHTS) for s in scenarios}
    print("\nBaseline (unperturbed) scenario checks:")
    for name, passed in baseline_results.items():
        print(f"  [{'OK' if passed else 'FAIL'}] {name}")
    if not all(baseline_results.values()):
        print(
            "\n  WARNING: at least one scenario fails even at baseline weights — "
            "that is a reward-function bug, not a sensitivity finding. Investigate "
            "before trusting the perturbation results below."
        )

    flips = []  # (weight_name, factor, scenario_name)
    per_weight_results = {}

    for weight_name in weight_names:
        per_weight_results[weight_name] = {}
        for factor in PERTURBATION_FACTORS:
            perturbed_weights = DEFAULT_WEIGHTS.perturbed(weight_name, factor)
            results = {s["name"]: _check_scenario(s, perturbed_weights) for s in scenarios}
            per_weight_results[weight_name][str(factor)] = results
            for scenario_name, passed in results.items():
                if baseline_results[scenario_name] and not passed:
                    flips.append((weight_name, factor, scenario_name))

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    total_checks = len(weight_names) * len(PERTURBATION_FACTORS) * len(scenarios)
    print(f"  Total (weight x factor x scenario) checks: {total_checks}")
    print(f"  Orderings that flipped under some perturbation: {len(flips)}")
    if flips:
        print("\n  Flips found:")
        for weight_name, factor, scenario_name in flips:
            print(f"    {weight_name} x{factor} -> '{scenario_name}' ordering flipped")
    else:
        print(
            "\n  No qualitative ordering flipped under any single +/-50% weight "
            "perturbation — the reward function's encoded design choices are "
            "robust to this class of disagreement about the exact magic-number "
            "values, within the scope this no-retrain analysis can check (see "
            "this module's docstring for what it does NOT check)."
        )

    output = {
        "environment": "reward_function_analytic",  # not a simulator/env — pure function evaluation
        "method": "one-weight-at-a-time perturbation (+/-50%), qualitative ordering check across "
        "representative scenarios — no retraining performed (see module docstring for why)",
        "perturbation_factors": PERTURBATION_FACTORS,
        "scenarios": [
            {
                "name": s["name"],
                "description": s["description"],
                "baseline_passed": baseline_results[s["name"]],
            }
            for s in scenarios
        ],
        "per_weight_results": per_weight_results,
        "flips": [{"weight": w, "factor": f, "scenario": s} for w, f, s in flips],
        "n_flips": len(flips),
        "n_total_checks": total_checks,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"reward_sensitivity_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    return output


if __name__ == "__main__":
    run_sensitivity_analysis()
