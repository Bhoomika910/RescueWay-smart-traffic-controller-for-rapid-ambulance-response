# evaluation/statistical_analysis.py
"""
RescueWay Statistical Significance Analysis.

Runs paired significance tests on robustness study results.

Why paired tests: every controller in a given study is evaluated on the
SAME sequence of seeds (base_seed, base_seed+1, ...). That means episode i
for DQN and episode i for Rule-Based used the identical traffic arrival
sequence and identical noise draws. This is a matched/paired design, so a
paired t-test (which looks at per-episode DIFFERENCES) is both statistically
correct and more powerful than an unpaired test — it removes episode-to-episode
variance that has nothing to do with which controller is better.

Tests performed, for each comparison:
  1. Paired t-test (scipy.stats.ttest_rel) — is the mean difference significant?
  2. Wilcoxon signed-rank test — non-parametric backup, robust to non-normality
  3. Cohen's d for paired samples — effect size (how big is the difference,
     independent of sample size)
  4. Shapiro-Wilk normality check on the differences — sanity check for
     whether the t-test's normality assumption is reasonable here

Usage:
    python -m evaluation.statistical_analysis --study 1
    python -m evaluation.statistical_analysis --study 3
    python -m evaluation.statistical_analysis --rerun --seeds 30
"""
import argparse
import json
import os
import sys
import time

import math

import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestPower

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.robustness_runner import (
    evaluate_condition,
    get_all_controllers,
)
from noise import NOISE_PRESETS, NoiseConfig

STATS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "statistics")
os.makedirs(STATS_DIR, exist_ok=True)

REFERENCE_CONTROLLER = "dqn"
METRICS_TO_TEST = ["reward", "emergency_accuracy"]


# ══════════════════════════════════════════════════════
# CORE STATISTICAL TESTS
# ══════════════════════════════════════════════════════


def cohens_d_paired(a, b):
    """
    Cohen's d for paired samples: mean difference / std of differences.

    Interpretation (Cohen 1988):
        |d| < 0.2   negligible
        |d| < 0.5   small
        |d| < 0.8   medium
        |d| >= 0.8  large
    """
    diffs = np.array(a) - np.array(b)
    std_diff = np.std(diffs, ddof=1)
    mean_diff = np.mean(diffs)
    # Guards a real numerical bug this project has hit in practice: when
    # every seed produces an (almost) identical difference — e.g. a
    # near-deterministic metric like a fixed-cycle controller's delay —
    # floating-point noise leaves std_diff at something like 1e-14
    # instead of exactly 0, and mean/std_diff then explodes to values
    # like 7.4e15, which is meaningless as an effect size (and previously
    # got printed/saved as-is). Anything below this epsilon is treated the
    # same as exactly-zero variance: the two conditions are consistently,
    # deterministically different (or identical), and no further precision
    # about "how many standard deviations apart" is meaningful.
    if std_diff < 1e-6:
        return 0.0 if abs(mean_diff) < 1e-6 else float(np.sign(mean_diff) * 10.0)
    return float(mean_diff / std_diff)


def significance_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _sanitise_p_values(p_values):
    """
    Replace non-finite p-values with 1.0 before any correction.

    A degenerate comparison — most commonly one where every paired
    difference is exactly zero, so the t-statistic is 0/0 — yields NaN
    from scipy. NaN then sorts unpredictably, and `min(1.0, nan * k)`
    returns nan, which compares False against alpha in some code paths
    and True in others. In this project's own output that produced an
    adjusted p-value of 0.0 and a "significant" verdict for a pair of
    perfectly identical 100%-accuracy vectors: the strongest possible
    claim, from the weakest possible evidence. A degenerate test is not
    evidence of an effect, so it is treated as p = 1.0.
    """
    out = []
    for p in p_values:
        try:
            fp = float(p)
        except (TypeError, ValueError):
            fp = float("nan")
        out.append(1.0 if not math.isfinite(fp) else fp)
    return out


def holm_bonferroni_correction(p_values, alpha=0.05):
    """
    Holm-Bonferroni step-down correction for a family of hypothesis tests.

    Why this is needed: analyze_study_1/3/5 each run many pairwise
    comparisons (Study 3 alone: ~5 presets x 4 baselines x 2 metrics = 40
    tests; combined across studies 1/3/5, ~58+). Reporting each at a raw,
    uncorrected alpha=0.05 inflates the family-wise false-positive rate
    well past 5% — with enough comparisons, some will look "significant"
    by chance alone. Holm-Bonferroni controls the family-wise error rate
    (guarantees P(at least one false positive) <= alpha) while being
    uniformly more powerful than a plain Bonferroni correction.

    Returns (adjusted_p_values, reject_flags), both in the SAME ORDER as
    the input `p_values` (not sorted).
    """
    p_values = _sanitise_p_values(p_values)
    n = len(p_values)
    if n == 0:
        return [], []

    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = n - rank  # largest for the smallest p-value, per Holm's step-down
        adj = min(1.0, p_values[idx] * factor)
        # Holm-Bonferroni requires the adjusted p-values to be
        # non-decreasing when read in ascending order of the raw p-value.
        running_max = max(running_max, adj)
        adjusted[idx] = running_max

    reject = [adj < alpha for adj in adjusted]
    return adjusted, reject


def benjamini_hochberg_correction(p_values, alpha=0.05):
    """
    Benjamini-Hochberg step-up procedure controlling the False Discovery
    Rate (Benjamini & Hochberg, "Controlling the False Discovery Rate: A
    Practical and Powerful Approach to Multiple Testing", J. Royal
    Statistical Society B, 1995).

    Why this exists alongside Holm-Bonferroni: Holm-Bonferroni controls
    the family-wise error rate (P(any false positive) <= alpha) — the
    right, conservative choice for a single confirmatory claim where one
    false positive anywhere is costly (e.g. Study 1's baseline calibration,
    Study 5's DQN-vs-Q-Learning head-to-head). Study 3 is different in
    kind: it deliberately sweeps dozens of noise-factor x value x
    controller combinations specifically to characterize WHERE
    degradation happens, not to certify one binary claim — FDR control
    (expected proportion of false positives AMONG significant results) is
    the standard, higher-power choice for that kind of exploratory,
    many-comparisons sweep, per Benjamini & Hochberg 1995.

    Returns (adjusted_p_values, reject_flags), both in the SAME ORDER as
    the input `p_values` (not sorted) — same contract as
    holm_bonferroni_correction.
    """
    p_values = _sanitise_p_values(p_values)
    n = len(p_values)
    if n == 0:
        return [], []

    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_min = 1.0
    # Step up from the LARGEST p-value to the smallest, taking a running
    # minimum, so adjusted p-values are non-decreasing in rank (BH's own
    # monotonicity requirement) when read in ascending order of raw p.
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        bh_factor = n / (rank + 1)
        adj = min(1.0, p_values[idx] * bh_factor)
        running_min = min(running_min, adj)
        adjusted[idx] = running_min

    reject = [adj < alpha for adj in adjusted]
    return adjusted, reject


def apply_multiple_comparison_correction(comparisons, alpha=0.05, method="holm"):
    """
    Attach multiple-comparison-corrected significance to every comparison
    in one study (one "family" of tests — see holm_bonferroni_correction's
    docstring for why correcting within a study, not globally across
    unrelated studies, is the right scope: that's how results are
    reported and is the natural unit a reader would scan for "how many
    of these N comparisons are real").

    Corrects the PRIMARY test's p-value (paired_comparison's normality-
    gated choice between the t-test and Wilcoxon), not unconditionally
    the t-test's — previously this always read paired_ttest's p-value
    regardless of what the normality check said, so the Shapiro-Wilk
    check was computed but never actually consulted.

    method: "holm" (family-wise error rate control — the conservative
    default, appropriate for a confirmatory head-to-head claim) or "bh"
    (Benjamini-Hochberg false-discovery-rate control — appropriate for an
    exploratory sweep across many conditions; see Study 3's usage).
    """
    if not comparisons:
        return comparisons
    p_values = [c["primary"]["p_value"] for c in comparisons]
    if method == "bh":
        adjusted, reject = benjamini_hochberg_correction(p_values, alpha=alpha)
        key_prefix = "bh"
    else:
        adjusted, reject = holm_bonferroni_correction(p_values, alpha=alpha)
        key_prefix = "holm"
    for c, adj_p, sig in zip(comparisons, adjusted, reject):
        c["primary"][f"p_value_{key_prefix}_adjusted"] = round(adj_p, 6)
        c["primary"]["significant_at_05_corrected"] = bool(sig)
        c["primary"]["correction_method"] = key_prefix
        # Kept for backward compatibility with anything still reading the
        # old location directly.
        c["paired_ttest"]["significant_at_05_corrected"] = bool(sig)
    return comparisons


def effect_label(d):
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def paired_comparison(name_a, values_a, name_b, values_b, metric_name):
    """
    Run the full battery of paired tests between two controllers on one metric.
    values_a and values_b must be the SAME LENGTH and index-aligned by seed.
    """
    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(
            f"Paired test requires equal-length arrays, got {len(a)} vs {len(b)}. "
            f"Both controllers must be evaluated on the same seed sequence."
        )

    diffs = a - b
    mean_diff = float(np.mean(diffs))

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(a, b)

    # Wilcoxon signed-rank (non-parametric backup)
    # Handle the all-zero-differences edge case (identical arrays)
    if np.allclose(diffs, 0):
        w_stat, w_p = 0.0, 1.0
    else:
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = None, None

    # Normality check on differences (informs which test to trust more).
    # Only the p-value is used downstream — Shapiro-Wilk's W statistic
    # itself isn't part of this project's reporting.
    if len(diffs) >= 3 and not np.allclose(diffs, diffs[0]):
        _, sw_p = stats.shapiro(diffs)
    else:
        sw_p = None

    d = cohens_d_paired(a, b)
    differences_look_normal = bool(sw_p > 0.05) if sw_p is not None else None

    # Normality-gated primary test (design-review finding): Shapiro-Wilk
    # was previously computed and stored but never actually consulted —
    # apply_multiple_comparison_correction and print_summary always read
    # paired_ttest's p-value regardless of what the normality check said.
    # Wilcoxon is the correct fallback when normality is violated (or
    # unknown/inconclusive at small n); default to the t-test only when
    # differences do look normal.
    # A degenerate comparison — every paired difference exactly zero, so
    # the t-statistic is 0/0 and scipy returns NaN — is not evidence of
    # an effect, and must not be allowed to reach the correction step as
    # a NaN. It arises legitimately here whenever two configurations are
    # identical by construction (e.g. any two shielded controllers on
    # emergency accuracy, which the shield pins to 100% for both), so it
    # is a normal outcome to report, not an error to raise.
    degenerate = bool(np.allclose(diffs, 0))

    if degenerate:
        primary_test = "degenerate"
        primary_p = 1.0
    elif differences_look_normal is False and w_p is not None:
        primary_test = "wilcoxon"
        primary_p = float(w_p)
    else:
        primary_test = "t_test"
        primary_p = float(t_p)
    if not math.isfinite(primary_p):
        primary_test = "degenerate"
        primary_p = 1.0

    # Achieved statistical power for this comparison's observed effect
    # size and sample size (statsmodels.stats.power.TTestPower — paired
    # t-test power via the one-sample/paired formulation, matching
    # ttest_rel above). Reported, not gated on — a comparison isn't
    # discarded for being underpowered, but "not significant" should
    # always be read alongside how much power there was to detect an
    # effect in the first place (design-review finding: no power
    # analysis existed anywhere in this codebase).
    try:
        achieved_power = float(TTestPower().power(effect_size=abs(d), nobs=len(a), alpha=0.05))
    except Exception:
        achieved_power = None

    return {
        "metric": metric_name,
        "controller_a": name_a,
        "controller_b": name_b,
        "mean_a": round(float(np.mean(a)), 3),
        "mean_b": round(float(np.mean(b)), 3),
        "mean_difference": round(mean_diff, 3),
        "n_pairs": len(a),
        "paired_ttest": {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(t_p), 6),
            "significant_at_05": bool(t_p < 0.05),
            "stars": significance_stars(t_p),
        },
        "wilcoxon": {
            "statistic": round(float(w_stat), 4) if w_stat is not None else None,
            "p_value": round(float(w_p), 6) if w_p is not None else None,
            "significant_at_05": bool(w_p < 0.05) if w_p is not None else None,
        },
        "normality_check": {
            "shapiro_p": round(float(sw_p), 4) if sw_p is not None else None,
            "differences_look_normal": differences_look_normal,
        },
        # The test this comparison's headline significance claim should
        # actually be read from, chosen by the normality check above
        # rather than always defaulting to the t-test.
        "primary": {
            "test": primary_test,
            "p_value": round(primary_p, 6),
            "significant_at_05": bool(primary_p < 0.05),
        },
        "effect_size": {
            "cohens_d": round(d, 3),
            "magnitude": effect_label(d),
            "favors": name_a if mean_diff > 0 else name_b,
        },
        "power_analysis": {
            "achieved_power": round(achieved_power, 3) if achieved_power is not None else None,
            "adequately_powered": (
                bool(achieved_power >= 0.8) if achieved_power is not None else None
            ),
            "note": (
                "Post hoc power for the observed effect size/n — a 'not "
                "significant' result with low achieved_power should be "
                "read as inconclusive, not as evidence of no effect."
            ),
        },
    }


# ══════════════════════════════════════════════════════
# STUDY-LEVEL ANALYSIS
# ══════════════════════════════════════════════════════


def run_and_collect_raw(noise_config, n_seeds, controllers=None):
    """Run all controllers under one noise condition, return raw per-seed data."""
    if controllers is None:
        controllers = get_all_controllers()

    raw = {}
    for name, ctrl in controllers.items():
        ctrl.reset_stats()
        result = evaluate_condition(ctrl, noise_config, n_seeds=n_seeds)
        raw[name] = result
    return raw


def analyze_condition(condition_name, raw_results):
    """
    Run all pairwise comparisons (DQN vs every other controller) for
    every metric in METRICS_TO_TEST, for one noise condition.
    """
    comparisons = []
    ref = raw_results.get(REFERENCE_CONTROLLER)
    if ref is None:
        return comparisons

    for other_name, other_result in raw_results.items():
        if other_name == REFERENCE_CONTROLLER:
            continue
        for metric in METRICS_TO_TEST:
            ref_values = ref["raw"][metric]
            other_values = other_result["raw"][metric]
            comparison = paired_comparison(
                REFERENCE_CONTROLLER,
                ref_values,
                other_name,
                other_values,
                metric,
            )
            comparison["condition"] = condition_name
            comparisons.append(comparison)

    return comparisons


def print_comparison(c):
    arrow = "beats" if c["effect_size"]["favors"] == c["controller_a"] else "loses to"
    primary = c["primary"]
    # "corrected" reflects the study's chosen multiple-comparison method
    # (Holm-Bonferroni or Benjamini-Hochberg — see
    # apply_multiple_comparison_correction) applied to the PRIMARY test's
    # p-value (normality-gated between t-test/Wilcoxon), not
    # unconditionally the t-test's — this is the number that should be
    # trusted over any single raw, uncorrected p-value.
    method = primary.get("correction_method", "holm")
    adj_key = f"p_value_{method}_adjusted"
    corrected = primary.get("significant_at_05_corrected")
    corrected_str = (
        f"corr_p={primary[adj_key]:.4f} ({method}, {'sig' if corrected else 'ns'})"
        if corrected is not None
        else ""
    )
    power = c["power_analysis"]["achieved_power"]
    power_str = f"power={power:.2f}" if power is not None else ""
    print(
        f"    [{c['condition']:12s}] {c['metric']:20s} | "
        f"{c['controller_a']}={c['mean_a']:>8.2f} vs {c['controller_b']}={c['mean_b']:>8.2f} | "
        f"Δ={c['mean_difference']:>+8.2f} | "
        f"primary={primary['test']:8s} p={primary['p_value']:.4f} | "
        f"{corrected_str} | "
        f"d={c['effect_size']['cohens_d']:>+.2f} ({c['effect_size']['magnitude']}) | "
        f"{power_str} | "
        f"{c['controller_a']} {arrow} {c['controller_b']}"
    )


# ══════════════════════════════════════════════════════
# STUDY 1: BASELINE SIGNIFICANCE
# ══════════════════════════════════════════════════════


def analyze_study_1(n_seeds=30):
    print("=" * 100)
    print("SIGNIFICANCE ANALYSIS — Study 1: Baseline Calibration (Clean Conditions)")
    print(f"Seeds: {n_seeds} | Test: paired t-test + Wilcoxon signed-rank + Cohen's d (paired)")
    print("=" * 100)

    raw = run_and_collect_raw(NoiseConfig(), n_seeds)
    comparisons = analyze_condition("clean", raw)
    apply_multiple_comparison_correction(comparisons)

    for c in comparisons:
        print_comparison(c)

    return {"study": "1_significance", "comparisons": comparisons}


# ══════════════════════════════════════════════════════
# STUDY 3: STRESS-CONDITION SIGNIFICANCE
# ══════════════════════════════════════════════════════


def analyze_study_3(n_seeds=30, presets=None):
    print("=" * 100)
    print("SIGNIFICANCE ANALYSIS — Study 3: Combined Stress Conditions")
    print(f"Seeds: {n_seeds}")
    print("=" * 100)

    if presets is None:
        presets = list(NOISE_PRESETS.keys())

    all_comparisons = []
    for preset_name in presets:
        print(f"\n  Condition: {preset_name}")
        config = NOISE_PRESETS[preset_name]
        raw = run_and_collect_raw(config, n_seeds)
        comparisons = analyze_condition(preset_name, raw)
        all_comparisons.extend(comparisons)

    # Correction is applied across the WHOLE study (all presets/metrics/
    # controllers) — that's the actual family of hypotheses a reader would
    # scan when asking "how many of these are real" — so printing happens
    # after correction, not per-preset inside the loop above. Study 3
    # deliberately sweeps many conditions to characterize WHERE
    # degradation happens rather than certifying one binary claim, so
    # Benjamini-Hochberg (FDR control, higher power for exploratory
    # sweeps) is used here instead of Study 1/5's Holm-Bonferroni (FWER
    # control, appropriate for a single confirmatory head-to-head claim).
    apply_multiple_comparison_correction(all_comparisons, method="bh")
    for c in all_comparisons:
        print_comparison(c)

    return {"study": "3_significance", "comparisons": all_comparisons}


# ══════════════════════════════════════════════════════
# STUDY 5: DQN vs Q-LEARNING SIGNIFICANCE
# ══════════════════════════════════════════════════════


def analyze_study_5(n_seeds=30):
    print("=" * 100)
    print("SIGNIFICANCE ANALYSIS — Study 5: DQN vs Q-Learning Head-to-Head")
    print(f"Seeds: {n_seeds}")
    print("=" * 100)

    from controllers.dqn_controller import DQNController
    from controllers.rl_controller import RLController

    all_comparisons = []
    for preset_name, config in NOISE_PRESETS.items():
        print(f"\n  Condition: {preset_name}")
        controllers = {"dqn": DQNController(), "q_learning": RLController()}
        raw = run_and_collect_raw(config, n_seeds, controllers=controllers)

        for metric in METRICS_TO_TEST:
            c = paired_comparison(
                "dqn",
                raw["dqn"]["raw"][metric],
                "q_learning",
                raw["q_learning"]["raw"][metric],
                metric,
            )
            c["condition"] = preset_name
            all_comparisons.append(c)

    # See analyze_study_3's comment: correction scope is the whole study.
    apply_multiple_comparison_correction(all_comparisons)
    for c in all_comparisons:
        print_comparison(c)

    return {"study": "5_significance", "comparisons": all_comparisons}


# ══════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════


def print_summary(all_comparisons):
    print("\n" + "=" * 100)
    print("SUMMARY: Significant Findings (p < 0.05, using each comparison's PRIMARY test)")
    print("=" * 100)

    # "Significant" below is judged on each comparison's PRIMARY test — the
    # t-test, unless Shapiro-Wilk found the paired differences non-normal,
    # in which case Wilcoxon is used instead (see paired_comparison). This
    # previously always read the t-test's p-value regardless of the
    # normality check.
    sig = [c for c in all_comparisons if c["primary"]["significant_at_05"]]
    not_sig = [c for c in all_comparisons if not c["primary"]["significant_at_05"]]
    # Multiple-comparison-corrected count — each comparison's flag was
    # computed against its OWN study's family (see
    # apply_multiple_comparison_correction), not against this combined
    # cross-study list; when --study all runs all three studies, this
    # combined summary is a convenience view over three separately-
    # corrected families, not a fourth, larger correction.
    sig_corrected = [c for c in all_comparisons if c["primary"].get("significant_at_05_corrected")]
    underpowered_not_sig = [
        c
        for c in not_sig
        if c["power_analysis"]["achieved_power"] is not None
        and c["power_analysis"]["achieved_power"] < 0.8
    ]

    print(
        f"\n  {len(sig)}/{len(all_comparisons)} comparisons statistically significant at "
        f"UNCORRECTED p<0.05 (primary test)"
    )
    print(
        f"  {len(sig_corrected)}/{len(all_comparisons)} comparisons remain significant after "
        f"multiple-comparison correction (per-study family) — THIS is the number that should "
        f"be cited, not the uncorrected count above, since {len(all_comparisons)} simultaneous "
        f"comparisons at an uncorrected alpha=0.05 would be expected to produce "
        f"~{len(all_comparisons) * 0.05:.1f} false positives by chance alone."
    )
    if underpowered_not_sig:
        print(
            f"  {len(underpowered_not_sig)}/{len(not_sig)} of the NOT-significant comparisons "
            f"had achieved power < 0.80 — read those as inconclusive (not enough seeds to "
            f"detect an effect this size), not as evidence of no difference.\n"
        )
    else:
        print()

    for c in sig:
        primary = c["primary"]
        corrected = primary.get("significant_at_05_corrected")
        corr_note = "also survives correction" if corrected else "DOES NOT survive correction"
        print(
            f"    [{c['condition']:12s}] {c['controller_a']} vs {c['controller_b']} "
            f"on {c['metric']}: {primary['test']} p={primary['p_value']:.4f}, "
            f"d={c['effect_size']['cohens_d']:+.2f} ({c['effect_size']['magnitude']}), "
            f"favors {c['effect_size']['favors']} [{corr_note}]"
        )

    if not_sig:
        print(f"\n  Not significant ({len(not_sig)}):")
        for c in not_sig:
            power = c["power_analysis"]["achieved_power"]
            power_note = f", power={power:.2f}" if power is not None else ""
            print(
                f"    [{c['condition']:12s}] {c['controller_a']} vs {c['controller_b']} "
                f"on {c['metric']}: {c['primary']['test']} p={c['primary']['p_value']:.4f}{power_note}"
            )


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════


def save_results(data, name):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(STATS_DIR, f"{name}_{timestamp}.json")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RescueWay Statistical Significance Analysis")
    parser.add_argument(
        "--study",
        type=str,
        default="1",
        choices=["1", "3", "5", "all"],
        help="Which study's significance to analyze",
    )
    parser.add_argument(
        "--seeds", type=int, default=30, help="Number of seeds (should match the original study)"
    )

    args = parser.parse_args()

    start = time.time()
    all_results = []

    if args.study in ("1", "all"):
        r = analyze_study_1(n_seeds=args.seeds)
        save_results(r, "study1_significance")
        all_results.extend(r["comparisons"])

    if args.study in ("3", "all"):
        r = analyze_study_3(n_seeds=args.seeds)
        save_results(r, "study3_significance")
        all_results.extend(r["comparisons"])

    if args.study in ("5", "all"):
        r = analyze_study_5(n_seeds=args.seeds)
        save_results(r, "study5_significance")
        all_results.extend(r["comparisons"])

    print_summary(all_results)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
