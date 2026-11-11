# RescueWay SUMO Validation Module

## Purpose

This module validates the RL controllers (both the tabular Q-Learning
policy and the DQN policy trained here in SUMO) inside SUMO — an
industry-standard microscopic traffic simulator.

This proves the learned emergency prioritization strategy is not
simulator-specific and can transfer to higher-fidelity environments.

## Architecture

```
Mode 1: Live Demo (existing system)
  Dashboard <-> FastAPI <-> Custom Simulator <-> RL Agent <-> Arduino

Mode 2: Research Validation (this module)
  SUMO <-> TraCI <-> evaluate_in_sumo_v2.py <-> DQN (dqn_model.pt) and
                                                 Q-Learning (q_table_v2.pkl)
```

DQN is trained directly in SUMO (`train_dqn_sumo.py` / `sumo_env.py`), so
`dqn_model.pt` is SUMO-native. Q-Learning is trained on the custom
simulator (`../train_offline.py`) and evaluated here to check transfer.

## Setup

### 1. Install SUMO

Download from: https://eclipse.dev/sumo/
Windows installer: Eclipse SUMO (includes netconvert, sumo, sumo-gui)

After installation, set environment variables:
```
SUMO_HOME = C:\Program Files (x86)\Eclipse\Sumo
PATH += C:\Program Files (x86)\Eclipse\Sumo\bin
```

### 2. Install Python TraCI

```bash
pip install traci sumolib
```

### 3. Build the Network

```bash
cd sumo
build_network.bat
```

This creates `intersection.net.xml` from the node and edge definitions.

### 4. Generate Routes

```bash
python generate_routes.py
```

This creates route files for each traffic pattern (used for standalone/
manual runs only — training and evaluation both regenerate routes on the
fly per episode with a deterministic seed, so these static files aren't
consumed by any pipeline).

### 5. Train DQN in SUMO (optional — a checkpoint is already included)

```bash
python train_dqn_sumo.py --episodes 250 --pattern all --seed 42
```

Saves to `../rl/dqn_model.pt` (same file the live system and the custom-
simulator benchmark load) and a training log to `training_logs/`.

### 6. Run Evaluation

**`evaluate_in_sumo_v2.py` is the current, recommended comparison script —
it is the only one of the two evaluation scripts that includes DQN.**
`evaluate_in_sumo.py` (v1) predates DQN's SUMO training and only compares
Q-Learning against the non-learned baselines; it's kept for its
`--transfer` cross-simulator check (see below), not as the primary
comparison tool.

```bash
# Compare all 5 controllers (DQN, Q-Learning, Fixed, Rule-Based, Random) in SUMO
python evaluate_in_sumo_v2.py --episodes 10

# DQN only
python evaluate_in_sumo_v2.py --dqn-only --episodes 10

# Run with GUI (visual)
python evaluate_in_sumo_v2.py --episodes 1 --gui

# Different traffic pattern
python evaluate_in_sumo_v2.py --pattern rush_hour --episodes 10
```

`evaluate_in_sumo.py` (v1, Q-Learning + baselines only) additionally
supports:
```bash
# Cross-simulator transfer validation: same Q-table, custom simulator vs
# SUMO — there is no DQN equivalent of this specific check yet, since DQN
# is trained directly in SUMO rather than on the custom simulator.
python evaluate_in_sumo.py --transfer --episodes 5

# Q-Learning only
python evaluate_in_sumo.py --rl-only --episodes 10
```

## Output

Results are saved as JSON in `sumo/results/` (v1: `sumo_comparison_*.json`
/ `transfer_validation_*.json`; v2: `sumo_comparison_v2_*.json`).

## Files

```
sumo/
├── intersection.nod.xml       Node definitions
├── intersection.edg.xml       Edge definitions
├── intersection.tll.xml       Traffic light logic (source — see note below)
├── intersection.net.xml       Compiled network (netconvert output; this is
│                              what SUMO actually runs against)
├── build_network.bat          Network builder script
├── rescueway.sumocfg          SUMO configuration (manual/GUI use only)
├── generate_routes.py         Route generator (called per-episode by every
│                              training/eval script; static .rou.xml files
│                              below are for manual use only)
├── routes_low/medium/high/rush_hour.rou.xml   Static route files (manual use)
├── sumo_adapter.py             SUMO <-> RescueWay bridge (zone/edge/phase mapping)
├── sumo_env.py                 Gym-like DQN training environment
├── train_dqn_sumo.py           DQN training loop (single or multi-pattern)
├── train_multi_seed.py         Multi-seed variance study for DQN training
├── evaluate_in_sumo_v2.py      Current 5-controller comparison (includes DQN)
├── evaluate_in_sumo.py         Legacy Q-Learning-only comparison + cross-
│                              simulator transfer check (--transfer)
├── checkpoints/                DQN training checkpoints
├── experiments/                Named experiment runs (config + summary)
├── training_logs/              Per-run JSON training logs
├── README.md                   This file
└── results/                    Evaluation output
```

### Note on `intersection.tll.xml`

`netconvert` (via `build_network.bat`) compiles its own traffic-light
program into `intersection.net.xml` (12-movement phase strings matching
`sumo_adapter.py`'s `ZONE_PHASES`), which does not match the simpler
8-character phase strings in the source `intersection.tll.xml`. SUMO runs
against the compiled `.net.xml`, so `.tll.xml` is not itself consulted at
simulation time — treat it as historical/reference input to `netconvert`,
not a file to hand-edit expecting it to change signal behavior.
