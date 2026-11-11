"""
RescueWay Multi-Seed Training Variance Estimation.

Trains the Double DQN from scratch N times with different random seeds,
then reports mean±std of final performance across training runs.

This addresses the reviewer concern: "your results come from a single
training run — is this seed-dependent?"

Usage:
    cd backend/sumo
    python train_multi_seed.py --runs 5 --episodes 200
    python train_multi_seed.py --runs 3 --episodes 200 --pattern all
    python train_multi_seed.py --runs 5 --episodes 200 --dueling --prioritized-replay --variant-label dueling_per

Expected time: this docstring previously estimated "5 runs x 200 episodes
x ~1 min/episode ~= 16-17 hours total" — off by roughly 50-60x against
observed throughput. Actually measured (2026-07-21, this machine): ~1.2s
per training episode at duration=300s, so 5 runs x 200 episodes ~= 20
minutes total, not 16-17 hours. Kept as a corrected number rather than
silently deleting the old (wrong) one, since the original estimate is
partly why this script existed but had never actually been run at
meaningful scale before that date — see ARCHITECTURE.md Part 8.11.

Output:
    sumo/training_logs/multi_seed_variance_<variant_label>_<timestamp>.json
    sumo/checkpoints/<variant_label>_seed_<N>_best.pt  (one per run)
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sumo_env import SumoTrafficEnv

from rl.dqn_agent import DQNAgent

# Patterns generate_routes.py actually knows how to build a route file for.
# "all" is NOT one of them — passing it here previously fell through to
# generate_routes.py's PATTERNS.get(pattern, PATTERNS["medium"]) default
# and silently trained on "medium" traffic the whole run despite the
# docstring/CLI advertising `--pattern all`. train_dqn_sumo.py's
# train_multi_pattern() is the actual multi-pattern cycling implementation;
# this script studies seed variance for a single pattern, so we fail loud
# instead of silently substituting a different pattern than requested.
SINGLE_PATTERNS = ["low", "medium", "high", "rush_hour"]

# ─── Output ───────────────────────────────────────────

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(__file__), "training_logs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def get_epsilon(episode, total, start=1.0, end=0.05, frac=0.7):
    decay = int(total * frac)
    if episode >= decay:
        return end
    return start - (episode / decay) * (start - end)


def train_single_run(
    run_id,
    base_seed,
    episodes=200,
    pattern="medium",
    duration=300,
    lr=5e-4,
    gamma=0.95,
    batch_size=64,
    buffer_size=100_000,
    target_update=200,
    hidden_dim=128,
    ambulance_prob=0.7,
    dueling=False,
    prioritized_replay=False,
    variant_label="baseline",
):
    """Train one DQN from scratch with a specific seed.

    `target_update` defaults to 200, matching train_dqn_sumo.py's
    corrected default (see DQNAgent's docstring for why 1000 — the old
    default here — is too infrequent at this training scale).

    `dueling`/`prioritized_replay`: forwarded to DQNAgent — see
    rl/dqn_network.py / rl/replay_buffer.py for what each does. Both
    default to False, matching this script's original (and still default)
    behavior exactly.

    `variant_label`: included in the saved checkpoint's filename so a
    dueling/PER run and a baseline run using the SAME seeds (as
    ARCHITECTURE.md's comparison deliberately does, for a fair paired
    comparison) don't silently overwrite each other's checkpoints.
    """
    if pattern not in SINGLE_PATTERNS:
        raise ValueError(
            f"train_single_run does not support pattern={pattern!r}. "
            f"Choose one of {SINGLE_PATTERNS}, or use "
            f"train_dqn_sumo.py --pattern all (train_multi_pattern) for "
            f"actual multi-pattern cycling."
        )

    print(f"\n{'='*60}")
    print(f"  TRAINING RUN {run_id} | seed={base_seed} | variant={variant_label}")
    print(f"{'='*60}")

    # Seed every RNG this run depends on, including torch's global RNG
    # (network weight init) — previously unseeded anywhere in the
    # codebase, which meant this script's whole purpose (isolating
    # variance attributable to `base_seed`) was undermined by an
    # uncontrolled extra source of run-to-run variance.
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    agent = DQNAgent(
        alpha=lr,
        gamma=gamma,
        epsilon=1.0,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_update_freq=target_update,
        hidden_dim=hidden_dim,
        inference_mode=False,
        dueling=dueling,
        prioritized_replay=prioritized_replay,
    )

    env = SumoTrafficEnv(
        pattern=pattern,
        duration=duration,
        ambulance_prob=ambulance_prob,
    )

    episode_rewards = []
    best_avg = -float("inf")
    best_path = os.path.join(CHECKPOINT_DIR, f"{variant_label}_seed_{base_seed}_best.pt")
    run_start = time.time()

    try:
        for ep in range(episodes):
            agent.epsilon = get_epsilon(ep, episodes)
            state = env.reset(seed=base_seed + ep)
            ep_reward = 0
            done = False

            while not done:
                action = agent.choose(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward

            env.close()
            episode_rewards.append(ep_reward)

            avg20 = np.mean(episode_rewards[-20:])
            if len(episode_rewards) >= 20 and avg20 > best_avg:
                best_avg = avg20
                agent.save(best_path)

            if (ep + 1) % 20 == 0:
                stats = agent.get_training_stats()
                elapsed = time.time() - run_start
                print(
                    f"    Ep {ep+1:4d}/{episodes} | "
                    f"Reward: {ep_reward:+8.1f} | Avg20: {avg20:+8.1f} | "
                    f"eps: {agent.epsilon:.3f} | Loss: {stats['avg_loss_100']:.4f} | "
                    f"Time: {elapsed:.0f}s"
                )

    except KeyboardInterrupt:
        print(f"\n  Run {run_id} interrupted.")
    finally:
        env.close()

    total_time = time.time() - run_start

    return {
        "run_id": run_id,
        "base_seed": base_seed,
        "episodes_completed": len(episode_rewards),
        "total_time_seconds": round(total_time, 1),
        "best_avg_reward_20": round(best_avg, 2),
        "final_avg_reward_20": (
            round(np.mean(episode_rewards[-20:]), 2) if len(episode_rewards) >= 20 else None
        ),
        "final_avg_reward_all": round(np.mean(episode_rewards), 2),
        "best_model_path": best_path,
        "reward_history": [round(r, 2) for r in episode_rewards],
    }


def evaluate_checkpoint(model_path, n_seeds=30):
    """
    Evaluate a saved checkpoint on the custom simulator (no SUMO needed)
    using the same protocol as robustness_runner Study 1.
    """
    from controllers.dqn_controller import DQNController
    from evaluation.robustness_runner import evaluate_condition
    from noise import NoiseConfig
    from rl.dqn_agent import DQNAgent as _DQNAgent
    from rl.state_features import StateFeatureExtractor as _SFE

    agent = _DQNAgent(epsilon=0.0, inference_mode=True)
    agent.feature_extractor = _SFE(noise_config=NoiseConfig())
    agent.load(model_path)

    ctrl = DQNController(agent=agent)
    result = evaluate_condition(ctrl, NoiseConfig(), n_seeds=n_seeds)
    return {
        "reward_mean": result["reward"]["mean"],
        "reward_std": result["reward"]["std"],
        "emergency_accuracy": result["emergency_accuracy"]["mean"],
    }


def run_multi_seed(
    n_runs=5,
    episodes=200,
    pattern="medium",
    dueling=False,
    prioritized_replay=False,
    variant_label="baseline",
    **kwargs,
):
    """Train N independent runs and report variance."""

    SEED_OFFSETS = [42, 1337, 7777, 2024, 9999, 314, 271, 161]
    seeds = SEED_OFFSETS[:n_runs]

    print("=" * 60)
    print("MULTI-SEED DQN TRAINING — VARIANCE ESTIMATION")
    print(f"Runs: {n_runs} | Episodes: {episodes} | Pattern: {pattern}")
    print(f"Variant: {variant_label} (dueling={dueling}, prioritized_replay={prioritized_replay})")
    print(f"Seeds: {seeds}")
    print("=" * 60)

    all_runs = []
    start = time.time()

    for i, seed in enumerate(seeds):
        run_result = train_single_run(
            run_id=i + 1,
            base_seed=seed,
            episodes=episodes,
            pattern=pattern,
            dueling=dueling,
            prioritized_replay=prioritized_replay,
            variant_label=variant_label,
            **kwargs,
        )
        all_runs.append(run_result)

    # Evaluate each checkpoint on clean custom-sim
    print("\n" + "=" * 60)
    print("EVALUATING EACH CHECKPOINT (30 seeds, clean, custom simulator)")
    print("=" * 60)

    eval_results = []
    for run in all_runs:
        path = run["best_model_path"]
        if os.path.exists(path):
            print(f"  Evaluating {path}...", end=" ", flush=True)
            ev = evaluate_checkpoint(path)
            eval_results.append(ev)
            print(f"reward={ev['reward_mean']:+.1f} emerg={ev['emergency_accuracy']:.1f}%")
        else:
            print(f"  WARNING: {path} not found, skipping eval")

    # Summary
    total_time = time.time() - start
    if eval_results:
        rewards = [e["reward_mean"] for e in eval_results]
        accuracies = [e["emergency_accuracy"] for e in eval_results]
        summary = {
            "n_runs": len(eval_results),
            "reward_mean_of_means": round(np.mean(rewards), 2),
            "reward_std_of_means": round(np.std(rewards), 2),
            "reward_range": [round(min(rewards), 2), round(max(rewards), 2)],
            "emergency_accuracy_mean": round(np.mean(accuracies), 2),
            "emergency_accuracy_std": round(np.std(accuracies), 2),
        }
        print(f"\n{'='*60}")
        print("MULTI-SEED VARIANCE SUMMARY")
        print(f"{'='*60}")
        print(f"  Runs evaluated:     {summary['n_runs']}")
        print(
            f"  Reward (mean±std):  {summary['reward_mean_of_means']:+.1f} "
            f"± {summary['reward_std_of_means']:.1f}"
        )
        print(
            f"  Reward range:       [{summary['reward_range'][0]:+.1f}, "
            f"{summary['reward_range'][1]:+.1f}]"
        )
        print(
            f"  Emergency acc:      {summary['emergency_accuracy_mean']:.1f}% "
            f"± {summary['emergency_accuracy_std']:.1f}%"
        )
        print(f"  Total time:         {total_time/3600:.1f} hours")
    else:
        summary = {"error": "no checkpoints evaluated"}

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"multi_seed_variance_{variant_label}_{timestamp}.json")
    output = {
        "config": {
            "n_runs": n_runs,
            "episodes": episodes,
            "pattern": pattern,
            "seeds": seeds,
            "dueling": dueling,
            "prioritized_replay": prioritized_replay,
            "variant_label": variant_label,
        },
        "runs": all_runs,
        "evaluations": eval_results,
        "summary": summary,
        "total_time_seconds": round(total_time, 1),
    }
    with open(log_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Full results saved to: {log_file}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed DQN training for variance estimation")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent training runs")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per run")
    parser.add_argument(
        "--pattern",
        type=str,
        default="medium",
        choices=["low", "medium", "high", "rush_hour", "all"],
    )
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--target-update",
        type=int,
        default=200,
        help="Steps between target network updates "
        "(matches train_dqn_sumo.py's corrected default)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden layer width for DQNetwork/DuelingDQNetwork.",
    )
    parser.add_argument(
        "--dueling",
        action="store_true",
        help="Use the Dueling DQN architecture (Wang et al., 2016) instead of the "
        "standard network. Default off — see rl/dqn_network.py.",
    )
    parser.add_argument(
        "--prioritized-replay",
        action="store_true",
        help="Use Prioritized Experience Replay (Schaul et al., 2016) instead of "
        "uniform sampling. Default off — see rl/replay_buffer.py.",
    )
    parser.add_argument(
        "--variant-label",
        type=str,
        default="baseline",
        help="Tag included in checkpoint/log filenames so runs with different "
        "architectures using the same seeds don't overwrite each other.",
    )

    args = parser.parse_args()
    run_multi_seed(
        n_runs=args.runs,
        episodes=args.episodes,
        pattern=args.pattern,
        duration=args.duration,
        lr=args.lr,
        target_update=args.target_update,
        hidden_dim=args.hidden_dim,
        dueling=args.dueling,
        prioritized_replay=args.prioritized_replay,
        variant_label=args.variant_label,
    )
