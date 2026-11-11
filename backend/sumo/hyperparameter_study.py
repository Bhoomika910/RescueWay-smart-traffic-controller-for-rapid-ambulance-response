"""
RescueWay DQN Hyperparameter Study.

Design-review finding (category "RL Improvements"): "hyperparameter studies"
were absent — every prior training run used one fixed (lr, hidden_dim,
target_update_freq) triple chosen once, early, and never revisited with
evidence. This script runs a one-factor-at-a-time sweep around that
baseline and reports whether any alternative actually helps, with the same
honesty-over-n standard already applied to the Dueling+PER comparison in
ARCHITECTURE.md Part 10.1: 3 seeds per configuration is enough to see a
gross difference, not enough to certify a small one.

Usage:
    cd backend/sumo
    python hyperparameter_study.py

Output:
    sumo/training_logs/hyperparameter_study_<timestamp>.json
"""

import json
import os
import time

import numpy as np
from train_multi_seed import run_multi_seed

LOG_DIR = os.path.join(os.path.dirname(__file__), "training_logs")

BASELINE = {"lr": 5e-4, "hidden_dim": 128, "target_update": 200}

# One-factor-at-a-time: hold the other two at BASELINE, vary the third.
GRID = [
    {"label": "baseline", **BASELINE},
    {"label": "lr_low", "lr": 1e-4, "hidden_dim": 128, "target_update": 200},
    {"label": "lr_high", "lr": 2e-3, "hidden_dim": 128, "target_update": 200},
    {"label": "hidden_small", "lr": 5e-4, "hidden_dim": 64, "target_update": 200},
    {"label": "hidden_large", "lr": 5e-4, "hidden_dim": 256, "target_update": 200},
    {"label": "target_update_fast", "lr": 5e-4, "hidden_dim": 128, "target_update": 50},
    {"label": "target_update_slow", "lr": 5e-4, "hidden_dim": 128, "target_update": 500},
]

N_SEEDS = 3
EPISODES = 100
PATTERN = "medium"


def run_study():
    start = time.time()
    results = {}
    for cfg in GRID:
        label = cfg["label"]
        print(
            f"\n{'#'*70}\n# CONFIG: {label}  (lr={cfg['lr']}, hidden_dim={cfg['hidden_dim']}, "
            f"target_update={cfg['target_update']})\n{'#'*70}"
        )
        out = run_multi_seed(
            n_runs=N_SEEDS,
            episodes=EPISODES,
            pattern=PATTERN,
            lr=cfg["lr"],
            hidden_dim=cfg["hidden_dim"],
            target_update=cfg["target_update"],
            variant_label=f"hp_{label}",
        )
        results[label] = out

    total_time = time.time() - start

    # Pairwise comparison: each variant vs. baseline, using the per-seed
    # evaluation reward/accuracy (same paired-seed structure as Part 10.1,
    # since GRID's seed list is identical across configs — see
    # train_multi_seed.SEED_OFFSETS, which is deterministic and unparameterized).
    baseline_rewards = np.array([e["reward_mean"] for e in results["baseline"]["evaluations"]])
    baseline_acc = np.array([e["emergency_accuracy"] for e in results["baseline"]["evaluations"]])

    comparison = {}
    for cfg in GRID:
        label = cfg["label"]
        if label == "baseline":
            continue
        evals = results[label]["evaluations"]
        if len(evals) != len(baseline_rewards):
            comparison[label] = {"error": "unequal run counts, skipping paired test"}
            continue
        rewards = np.array([e["reward_mean"] for e in evals])
        acc = np.array([e["emergency_accuracy"] for e in evals])

        reward_diff = rewards - baseline_rewards
        acc_diff = acc - baseline_acc
        comparison[label] = {
            "config": cfg,
            "reward_mean": round(float(rewards.mean()), 2),
            "reward_vs_baseline_mean_diff": round(float(reward_diff.mean()), 2),
            "emergency_accuracy_mean": round(float(acc.mean()), 2),
            "emergency_accuracy_vs_baseline_mean_diff": round(float(acc_diff.mean()), 2),
        }
        # n=3 is too small for a meaningful p-value from scipy's paired
        # t-test (2 degrees of freedom) — reported as a descriptive mean
        # difference only, not accompanied by a p-value that would imply
        # more statistical power than 3 seeds can actually provide.

    output = {
        "grid": GRID,
        "n_seeds": N_SEEDS,
        "episodes": EPISODES,
        "pattern": PATTERN,
        "results_by_label": {
            label: {"config": next(c for c in GRID if c["label"] == label), "summary": r["summary"]}
            for label, r in results.items()
        },
        "comparison_vs_baseline": comparison,
        "total_time_seconds": round(total_time, 1),
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"hyperparameter_study_{timestamp}.json")
    with open(log_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}\nHYPERPARAMETER STUDY COMPLETE\n{'='*70}")
    print(f"{'Config':<22} {'Reward (mean)':>15} {'Emerg Acc %':>12}")
    for label, r in output["results_by_label"].items():
        s = r["summary"]
        if "reward_mean_of_means" in s:
            print(
                f"{label:<22} {s['reward_mean_of_means']:>15.1f} {s['emergency_accuracy_mean']:>12.1f}"
            )
    print(f"\nSaved to: {log_file}")
    print(f"Total time: {total_time/60:.1f} minutes")
    return output


if __name__ == "__main__":
    run_study()
