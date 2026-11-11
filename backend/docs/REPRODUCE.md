# Reproducing the research results

Design-review finding (category 26 "Documentation"): "add experiment reproduction guide."
Every command below is exactly what produced the result files cited throughout
`backend/ARCHITECTURE.md` — none of this is paraphrased or simplified from what was
actually run. Run `python -m evaluation.experiment_registry` afterward (from `backend/`) to
get an indexed list of every result file this produces, with a one-line summary of each.

All commands assume `cd backend` first unless stated otherwise. Every training script
seeds `random`/`np.random`/`torch.manual_seed` (and, for SUMO, `traci.start()`'s own
`--seed` flag) — see `backend/ARCHITECTURE.md` Part 8.15 for the reproducibility
verification (re-running the same seed twice produces byte-identical reward trajectories).

## Prerequisites

```bash
pip install -r requirements-dev.txt
```

SUMO-based commands additionally require a real SUMO install with `SUMO_HOME` set — see
`backend/sumo/README.md`. Everything else (simplified-simulator training/evaluation) has no
external dependency beyond the pip packages above.

## Tabular Q-Learning

```bash
# Canonical production checkpoint (rl/q_table_v2.pkl) — seed=42, rush_hour held out
python train_offline.py --episodes 25000 --seed 42 --held-out-pattern rush_hour --eval-episodes 300

# 5-seed variance study (matches ARCHITECTURE.md Part 8.11's table)
for seed in 42 1337 7777 2024 9999; do
  python train_offline.py --episodes 25000 --seed $seed --held-out-pattern rush_hour --eval-episodes 300
done
python aggregate_qlearning_variance.py
# IMPORTANT: the sweep above leaves whichever seed ran LAST as the file on disk
# (rl/q_table_v2.pkl always saves to the same path) — re-run seed=42 once more
# afterward to restore the documented canonical checkpoint:
python train_offline.py --episodes 25000 --seed 42 --held-out-pattern rush_hour --eval-episodes 300

# Native SUMO training (closes the domain-gap asymmetry — see Part 8's SUMO section)
cd sumo
python train_qlearning_sumo.py --episodes 1000 --pattern medium --seed 42
```

## DQN

```bash
cd sumo

# Multi-pattern curriculum with replay-buffer carryover + forgetting check
# (produces the currently-deployed rl/dqn_model.pt)
python train_dqn_sumo.py --pattern all --episodes-per-pattern 250 --duration 300 --ambulance-prob 0.7

# 5-seed variance study, baseline architecture
python train_multi_seed.py --runs 5 --episodes 200 --pattern medium

# 5-seed variance study, Dueling DQN + Prioritized Experience Replay
python train_multi_seed.py --runs 5 --episodes 200 --pattern medium \
    --dueling --prioritized-replay --variant-label dueling_per
```

## Evaluation / benchmark pipeline

Run from `backend/` in this order (none depend on each other's *output*, but all depend on
the checkpoints above already existing):

```bash
python run_comparison.py                                  # flagship 7-controller benchmark
python -m evaluation.robustness_runner --study all --seeds 30
python -m evaluation.ablation_studies --study all           # Study C uses 3 seeds by default (10k episodes each)
python -m evaluation.statistical_analysis --study all --seeds 30
python -m evaluation.multi_ambulance_significance --seeds 30
python -m evaluation.gps_confusion_analysis --seeds 30
python -m evaluation.reward_sensitivity                      # no seeds — analytic, not simulated
```

## Rebuilding the experiment index

```bash
python -m evaluation.experiment_registry
```

Produces `results/experiment_registry.json` — every result file above (and every prior run
still on disk), indexed with a one-line summary. This is not itself a research result; it's
the map to all the others.

## What "reproduce" means here, precisely

- **Bit-identical**: reward trajectories for a fixed seed on the simplified simulator
  (verified in `backend/ARCHITECTURE.md` Part 8.15).
- **Statistically equivalent, not bit-identical**: anything involving SUMO's own internal
  stochasticity beyond what `--seed` controls, and anything run on different hardware
  (wall-clock timings will differ; the *learned behavior* should not).
- **Not reproducible by design**: the live Arduino hardware interaction (falls back to
  simulation automatically if no real hardware is attached) and the one 5-6-zone
  hypothetical-parametrization test (`tests/test_zone_parametrization.py`) — that test
  exists to prove the encoding *scheme* generalizes, not to produce a citable result.
