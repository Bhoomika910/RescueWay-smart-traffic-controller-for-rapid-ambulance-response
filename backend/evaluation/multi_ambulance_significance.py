"""
RescueWay Multi-Ambulance Benchmark — Paired Statistical Significance.

Extends multi_ambulance_benchmark.py to run paired significance tests
(paired t-test, Wilcoxon signed-rank, Cohen's d) between scheduling
strategies on both reward and emergency delay, per scenario.

Key design: every strategy is evaluated on the SAME seed sequence within
each scenario, making this a matched/paired design — same as the
robustness studies.

Usage:
    python -m evaluation.multi_ambulance_significance --seeds 30

Output:
    results/multi_ambulance/significance_<timestamp>.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BENCHMARK_SEED, ENV_TAG_SIMPLIFIED
from evaluation.multi_ambulance_benchmark import (
    CongestionWeightedStrategy,
    ETAStrategy,
    FIFOStrategy,
    generate_scenario,
    run_multi_ambulance_episode,
)
from evaluation.statistical_analysis import (
    apply_multiple_comparison_correction,
    paired_comparison,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "multi_ambulance")
os.makedirs(RESULTS_DIR, exist_ok=True)

SCENARIO_TYPES = [
    "two_opposing",
    "two_adjacent",
    "triple_rapid",
    "close_vs_far",
    "far_then_close",
    "four_simultaneous",
    "two_waves",
    "random",
]


def paired_test(name_a, values_a, name_b, values_b, metric):
    """
    Thin wrapper around the shared evaluation.statistical_analysis.paired_comparison.

    Previously this was a full, independent re-implementation of the same
    paired t-test/Wilcoxon/Cohen's d logic (a real, previously-uncaught
    case of the same duplicated-constants problem the design-review audit
    already flagged elsewhere: two copies of the same statistical test
    that could silently drift apart). It also lacked the normality-gated
    primary-test selection and power analysis added to the shared
    function — this module gets both for free now. Field name aliases
    below (strategy_a/strategy_b/mean_diff/paired_ttest.significant) keep
    this module's own print statements working unchanged.
    """
    c = paired_comparison(name_a, values_a, name_b, values_b, metric)
    c["strategy_a"] = c["controller_a"]
    c["strategy_b"] = c["controller_b"]
    c["mean_diff"] = c["mean_difference"]
    c["paired_ttest"]["significant"] = c["paired_ttest"]["significant_at_05"]
    return c


def run_significance_benchmark(n_seeds=30):
    """Run all strategies with per-episode raw data, then paired tests."""

    strategies = [
        FIFOStrategy(),
        ETAStrategy(),
        CongestionWeightedStrategy(),
    ]

    print("=" * 80)
    print("MULTI-AMBULANCE CONFLICT RESOLUTION — SIGNIFICANCE TESTING")
    print(f"Strategies: {[s.name for s in strategies]}")
    print(f"Scenarios: {len(SCENARIO_TYPES)} | Seeds: {n_seeds}")
    print("=" * 80)

    all_comparisons = []
    raw_data = {}

    for scenario_type in SCENARIO_TYPES:
        print(f"\n  Scenario: {scenario_type}")
        raw_data[scenario_type] = {}

        # Collect per-seed raw results for each strategy
        for strategy in strategies:
            rewards = []
            delays = []
            for seed_idx in range(n_seeds):
                seed = BENCHMARK_SEED + seed_idx
                events = generate_scenario(scenario_type, seed)
                result = run_multi_ambulance_episode(strategy, events, seed)
                rewards.append(result["reward"])
                delays.append(result["avg_emergency_delay"])
            raw_data[scenario_type][strategy.name] = {
                "reward": rewards,
                "delay": delays,
            }

        # Pairwise comparisons: FIFO vs ETA, FIFO vs Congestion, ETA vs Congestion
        pairs = [
            ("FIFO", "ETA-Based"),
            ("FIFO", "Congestion-Weighted"),
            ("ETA-Based", "Congestion-Weighted"),
        ]

        for name_a, name_b in pairs:
            for metric in ["reward", "delay"]:
                vals_a = raw_data[scenario_type][name_a][metric]
                vals_b = raw_data[scenario_type][name_b][metric]

                comp = paired_test(name_a, vals_a, name_b, vals_b, metric)
                comp["scenario"] = scenario_type
                all_comparisons.append(comp)

    # Multiple-comparison correction across the whole run's family of
    # comparisons (3 strategy-pairs x 2 metrics x 8 scenarios = 48 tests)
    # — previously reported at a raw, uncorrected alpha=0.05 with no
    # correction at all, same issue as evaluation/statistical_analysis.py.
    # This is an exploratory sweep across many scenarios (characterizing
    # WHERE strategies differ, not one confirmatory claim), so
    # Benjamini-Hochberg (FDR control) is used, matching Study 3's same
    # reasoning. Printing is deferred to after correction (rather than
    # inline as each comparison was computed) so the printed markers can
    # reflect it, and now reads each comparison's normality-gated PRIMARY
    # test rather than unconditionally the t-test.
    apply_multiple_comparison_correction(all_comparisons, method="bh")
    for comp in all_comparisons:
        primary = comp["primary"]
        if primary["significant_at_05"]:
            corrected = primary.get("significant_at_05_corrected")
            corr_note = "also survives correction" if corrected else "does NOT survive correction"
            print(
                f"    [{comp['scenario']:18s}] {comp['strategy_a']} vs {comp['strategy_b']} on {comp['metric']}: "
                f"Δ={comp['mean_diff']:+.2f} "
                f"{primary['test']} p={primary['p_value']:.4f} "
                f"d={comp['effect_size']['cohens_d']:+.2f} "
                f"({comp['effect_size']['magnitude']}) "
                f"favors {comp['effect_size']['favors']} [{corr_note}]"
            )

    # Summary
    sig = [c for c in all_comparisons if c["primary"]["significant_at_05"]]
    sig_corrected = [c for c in all_comparisons if c["primary"].get("significant_at_05_corrected")]
    print(f"\n{'='*80}")
    print(
        f"SUMMARY: {len(sig)}/{len(all_comparisons)} comparisons significant at UNCORRECTED p<0.05 (primary test)"
    )
    print(
        f"         {len(sig_corrected)}/{len(all_comparisons)} remain significant after "
        f"Benjamini-Hochberg correction — cite THIS number, not the uncorrected one above."
    )
    print(f"{'='*80}")

    if sig:
        for c in sig:
            primary = c["primary"]
            corrected = primary.get("significant_at_05_corrected")
            print(
                f"  [{c['scenario']:18s}] {c['strategy_a']} vs {c['strategy_b']} "
                f"on {c['metric']:6s}: {primary['test']} p={primary['p_value']:.4f}, "
                f"d={c['effect_size']['cohens_d']:+.2f} "
                f"({c['effect_size']['magnitude']}) "
                f"[{'survives' if corrected else 'does NOT survive'} correction]"
            )
    else:
        print("  No significant differences found.")
        print("  (This is expected for scenarios where strategies produce identical behavior.)")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(RESULTS_DIR, f"significance_{timestamp}.json")
    output = {
        "description": "Multi-ambulance paired significance testing",
        "environment": ENV_TAG_SIMPLIFIED,
        "n_seeds": n_seeds,
        "scenarios": SCENARIO_TYPES,
        "strategies": [s.name for s in strategies],
        "comparisons": all_comparisons,
        "significant_count": len(sig),
        "total_comparisons": len(all_comparisons),
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {filepath}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Ambulance Significance Testing")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    run_significance_benchmark(n_seeds=args.seeds)
