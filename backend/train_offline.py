# LEGACY: Q-Learning utility script. Superseded by DQN training pipeline.
# train_offline.py
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")  # headless — never blocks on plt.show()
import matplotlib.pyplot as plt
import numpy as np

from rl.emergency_scoring import is_emergency_action_correct
from rl.environment import ALL_PATTERNS, TrafficEnv
from rl.q_agent import QAgent
from rl.state_encoder import N_ACTIONS, N_STATES

parser = argparse.ArgumentParser()
parser.add_argument(
    "--episodes",
    type=int,
    default=25000,
    help="Training episodes (default 25000, the original full-scale run)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Base random seed. Previously this script seeded "
    "nothing at all — two 'identical' runs would "
    "silently produce different Q-tables every time, "
    "unlike sumo/train_dqn_sumo.py which already seeds "
    "random/np.random/torch before training starts.",
)
parser.add_argument(
    "--held-out-pattern",
    type=str,
    default="rush_hour",
    choices=ALL_PATTERNS + ["none"],
    help="Traffic pattern EXCLUDED from training and used "
    "for a genuine held-out generalization check "
    "afterward (design-review finding: this codebase "
    "previously had zero train/test separation — "
    "every evaluation used patterns statistically "
    "identical to training. 'rush_hour' was already "
    "silently excluded from TrafficEnv's default "
    "rotation by accident; this makes that choice "
    "deliberate and measures the actual generalization "
    "gap it implies). Pass 'none' to train on all 4 "
    "patterns, matching the pre-fix behavior for "
    "comparison.",
)
parser.add_argument(
    "--eval-episodes",
    type=int,
    default=300,
    help="Episodes used for the post-training held-out " "generalization evaluation",
)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

held_out = None if args.held_out_pattern == "none" else args.held_out_pattern
training_patterns = [p for p in ALL_PATTERNS if p != held_out] if held_out else list(ALL_PATTERNS)

print("=" * 60)
print("RescueWay Q-Learning Training v2")
print(f"State space: {N_STATES} states | Action space: {N_ACTIONS} actions")
print(f"Seed: {args.seed}")
print(f"Training patterns: {training_patterns}")
print(f"Held-out pattern (generalization test): {held_out or 'NONE — trained on all 4'}")
print("=" * 60)

# Training settings
EPISODES = args.episodes
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = EPSILON_START / (EPISODES * 0.7)

agent = QAgent(alpha=0.1, gamma=0.95, epsilon=EPSILON_START)
env = TrafficEnv(training_patterns=training_patterns)

episode_rewards = []
episode_lengths = []
correct_actions = 0
total_actions = 0
emergency_correct = 0
emergency_total = 0
# Real per-episode ambulance-delay outcomes (rl.ambulance_delay proxy via
# TrafficEnv), tracked separately from reward so the proactive objective
# can be verified directly — reward can rise from congestion/throughput
# terms alone, average ambulance delay trending down cannot be faked.
episode_ambulance_delays = []
episode_ambulance_stop_rates = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done:
        action = agent.choose(state)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state, done)

        total_reward += reward
        steps += 1
        total_actions += 1

        # Track emergency accuracy. action==4 ("extend current green") also
        # env.active_signal reflects the post-step served zone (env.step()
        # already ran above), so it's the correct served_zone argument.
        if info["ambulance_active"] and info["ambulance_zone"] > 0:
            emergency_total += 1
            if is_emergency_action_correct(action, info["ambulance_zone"], env.active_signal):
                emergency_correct += 1
                correct_actions += 1

        state = next_state

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    if env.episode_ambulance_trips > 0:
        episode_ambulance_delays.append(float(np.mean(env.episode_ambulance_delays)))
        episode_ambulance_stop_rates.append(
            env.episode_ambulance_stops / env.episode_ambulance_trips
        )

    # Decay epsilon
    agent.epsilon = max(EPSILON_END, agent.epsilon - EPSILON_DECAY)

    # Print progress
    if (episode + 1) % 500 == 0:
        avg_reward = np.mean(episode_rewards[-500:])
        avg_length = np.mean(episode_lengths[-500:])
        emerg_acc = (emergency_correct / emergency_total * 100) if emergency_total > 0 else 0
        avg_delay = np.mean(episode_ambulance_delays[-500:]) if episode_ambulance_delays else 0
        print(
            f"Episode {episode + 1:5d} | "
            f"Avg Reward: {avg_reward:8.2f} | "
            f"Avg Steps: {avg_length:5.1f} | "
            f"Emergency Acc: {emerg_acc:5.1f}% | "
            f"Avg Ambulance Delay: {avg_delay:5.2f}s | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

# Save
agent.save()

# Final stats
final_avg = np.mean(episode_rewards[-200:])
emerg_acc = (emergency_correct / emergency_total * 100) if emergency_total > 0 else 0
final_avg_delay = np.mean(episode_ambulance_delays[-200:]) if episode_ambulance_delays else 0
final_stop_rate = (
    np.mean(episode_ambulance_stop_rates[-200:]) if episode_ambulance_stop_rates else 0
)

print("\n" + "=" * 60)
print("Training Complete")
print(f"States: {N_STATES} | Actions: {N_ACTIONS}")
print(f"Final avg reward (last 200): {final_avg:.2f}")
print(f"Emergency accuracy: {emerg_acc:.1f}%")
print(f"Final avg ambulance delay (last 200 ep): {final_avg_delay:.2f}s")
print(f"Final ambulance stop rate (last 200 ep): {final_stop_rate*100:.1f}%")
print(f"Total episodes: {EPISODES}")
print("=" * 60)


def evaluate_on_pattern(pattern, n_episodes, eval_seed):
    """Greedy (epsilon=0) evaluation restricted to ONE fixed pattern —
    used both for the held-out generalization check and for the matched
    in-distribution comparison below, so the two numbers are measured the
    same way and differ only in which pattern was used."""
    prior_epsilon = agent.epsilon
    agent.epsilon = 0.0
    try:
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        eval_env = TrafficEnv(training_patterns=[pattern])
        rewards, delays, stops, trips = [], [], 0, 0
        emerg_correct, emerg_total = 0, 0
        for _ in range(n_episodes):
            state = eval_env.reset()
            eval_env.simulator.set_traffic_pattern(
                pattern
            )  # reset() already restricts to [pattern], explicit for clarity
            done = False
            ep_reward = 0
            while not done:
                action = agent.choose(state)
                state, reward, done, info = eval_env.step(action)
                ep_reward += reward
                if info["ambulance_active"] and info["ambulance_zone"] > 0:
                    emerg_total += 1
                    if is_emergency_action_correct(
                        action, info["ambulance_zone"], eval_env.active_signal
                    ):
                        emerg_correct += 1
            rewards.append(ep_reward)
            if eval_env.episode_ambulance_trips > 0:
                delays.append(float(np.mean(eval_env.episode_ambulance_delays)))
                stops += eval_env.episode_ambulance_stops
                trips += eval_env.episode_ambulance_trips
        return {
            "pattern": pattern,
            "avg_reward": round(float(np.mean(rewards)), 2),
            "emergency_accuracy_pct": (
                round(emerg_correct / emerg_total * 100, 1) if emerg_total else 0.0
            ),
            "avg_ambulance_delay": round(float(np.mean(delays)), 3) if delays else 0.0,
            "ambulance_stop_rate_pct": round(stops / trips * 100, 1) if trips else 0.0,
        }
    finally:
        agent.epsilon = prior_epsilon


generalization_check = None
if held_out:
    print(
        f"\nEvaluating generalization: held-out pattern ('{held_out}') "
        f"vs. in-distribution training patterns..."
    )
    held_out_result = evaluate_on_pattern(
        held_out, args.eval_episodes, eval_seed=args.seed + 90_000
    )
    in_distribution_results = [
        evaluate_on_pattern(p, args.eval_episodes, eval_seed=args.seed + 90_000)
        for p in training_patterns
    ]
    in_distribution_avg_reward = float(np.mean([r["avg_reward"] for r in in_distribution_results]))
    in_distribution_avg_acc = float(
        np.mean([r["emergency_accuracy_pct"] for r in in_distribution_results])
    )

    generalization_check = {
        "held_out_pattern": held_out,
        "training_patterns": training_patterns,
        "held_out_result": held_out_result,
        "in_distribution_results": in_distribution_results,
        "in_distribution_avg_reward": round(in_distribution_avg_reward, 2),
        "in_distribution_avg_emergency_accuracy_pct": round(in_distribution_avg_acc, 1),
        "reward_generalization_gap": round(
            held_out_result["avg_reward"] - in_distribution_avg_reward, 2
        ),
        "emergency_accuracy_generalization_gap_pct": round(
            held_out_result["emergency_accuracy_pct"] - in_distribution_avg_acc, 1
        ),
    }

    print(
        f"  In-distribution (trained patterns) avg reward: {in_distribution_avg_reward:+.2f}, "
        f"emergency accuracy: {in_distribution_avg_acc:.1f}%"
    )
    print(
        f"  Held-out ('{held_out}')            avg reward: {held_out_result['avg_reward']:+.2f}, "
        f"emergency accuracy: {held_out_result['emergency_accuracy_pct']:.1f}%"
    )
    print(
        f"  Generalization gap: {generalization_check['reward_generalization_gap']:+.2f} reward, "
        f"{generalization_check['emergency_accuracy_generalization_gap_pct']:+.1f}pp emergency accuracy"
    )
else:
    print(
        "\nNo held-out pattern configured (--held-out-pattern none) — skipping generalization check."
    )

# Save a JSON results log alongside the checkpoint/plot — previously this
# script printed and plotted but never saved a machine-readable record,
# unlike every other training/evaluation script in this codebase.
timestamp = time.strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(os.path.dirname(__file__), "results", "qlearning_training")
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, f"train_offline_{timestamp}.json")
with open(results_path, "w") as f:
    json.dump(
        {
            "timestamp": timestamp,
            "episodes": EPISODES,
            "seed": args.seed,
            "training_patterns": training_patterns,
            "held_out_pattern": held_out,
            "final_avg_reward_last200": round(float(final_avg), 2),
            "final_emergency_accuracy_pct": round(float(emerg_acc), 1),
            "final_avg_ambulance_delay": round(float(final_avg_delay), 3),
            "final_ambulance_stop_rate_pct": round(float(final_stop_rate) * 100, 1),
            "generalization_check": generalization_check,
        },
        f,
        indent=2,
    )
print(f"\nResults saved to: {results_path}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("RescueWay Q-Learning v2 — Training Results", fontsize=14)

# Reward curve
ax1 = axes[0][0]
ax1.plot(episode_rewards, alpha=0.2, color="steelblue", linewidth=0.5)
if len(episode_rewards) >= 100:
    moving_avg = np.convolve(episode_rewards, np.ones(100) / 100, mode="valid")
    ax1.plot(moving_avg, color="steelblue", linewidth=2, label="Moving avg (100 ep)")
    ax1.legend()
ax1.set_title("Reward per Episode")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.grid(True, alpha=0.3)

# Ambulance delay curve — the direct evidence that the proactive
# objective (not just reward) is being learned.
ax2 = axes[0][1]
if episode_ambulance_delays:
    ax2.plot(episode_ambulance_delays, alpha=0.3, color="coral", linewidth=0.5)
    window = min(50, len(episode_ambulance_delays))
    if len(episode_ambulance_delays) >= window and window > 1:
        delay_avg = np.convolve(episode_ambulance_delays, np.ones(window) / window, mode="valid")
        ax2.plot(delay_avg, color="coral", linewidth=2, label=f"Moving avg ({window} ep)")
        ax2.legend()
ax2.set_title("Ambulance Delay per Episode (lower is better)")
ax2.set_xlabel("Episode (with an ambulance trip)")
ax2.set_ylabel("Seconds")
ax2.grid(True, alpha=0.3)

# Episode length
ax3 = axes[1][0]
ax3.plot(episode_lengths, alpha=0.2, color="green", linewidth=0.5)
if len(episode_lengths) >= 100:
    length_avg = np.convolve(episode_lengths, np.ones(100) / 100, mode="valid")
    ax3.plot(length_avg, color="green", linewidth=2, label="Moving avg (100 ep)")
    ax3.legend()
ax3.set_title("Episode Length")
ax3.set_xlabel("Episode")
ax3.set_ylabel("Steps")
ax3.grid(True, alpha=0.3)

# Reward distribution
ax4 = axes[1][1]
ax4.hist(episode_rewards[-500:], bins=30, color="steelblue", alpha=0.7, edgecolor="white")
ax4.set_title("Reward Distribution (Last 500 Episodes)")
ax4.set_xlabel("Total Reward")
ax4.set_ylabel("Frequency")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curve_v2.png", dpi=150)
print("\nTraining curves saved as training_curve_v2.png")
