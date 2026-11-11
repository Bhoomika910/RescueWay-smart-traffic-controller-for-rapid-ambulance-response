# sumo/train_qlearning_sumo.py
"""
Train tabular Q-Learning natively in SUMO — closes the other half of the
documented SUMO-vs-simplified-environment domain gap (design-review
finding, category 20/21 "SUMO integration"/"Traffic simulator realism":
"reduce the SUMO vs simplified-environment domain gap ... train Q-Learning
natively in SUMO, compare vs simplified-env checkpoint").

Before this script existed: DQN trained EXCLUSIVELY in SUMO
(train_dqn_sumo.py) with no simplified-environment-only training path,
while Q-Learning trained EXCLUSIVELY in the simplified custom simulator
(train_offline.py) with no SUMO training path. That asymmetry — not just
"the two environments differ physically," which is expected and
unavoidable — is what made a genuine like-for-like domain-gap comparison
impossible: any performance difference between the two agents could
always be attributed to "of course, they were never even trained the same
way" rather than to anything about the agents/algorithms themselves. This
script, combined with sumo_env.SumoTrafficEnv's new use_discrete_encoder
flag, gives Q-Learning a real SUMO training path so both agents can now be
compared honestly on: (a) SUMO-trained vs SUMO-trained, and (b) each
agent's own SUMO-trained-vs-simplified-trained gap.

Scope/budget note: unlike train_offline.py's 25,000-episode simplified-
simulator run (cheap — no real per-vehicle simulation), SUMO training is
~1.2s/episode on this project's hardware (see sumo/train_multi_seed.py's
corrected docstring), making 25,000 SUMO episodes a ~8-hour run. This
script defaults to a budget comparable to DQN's own SUMO training scale
(sumo/train_dqn_sumo.py's train_multi_pattern default is 250
episodes/pattern; sumo/train_multi_seed.py's variance study uses 200) so
the two agents are compared at comparable SUMO training budgets, not an
apples-to-oranges "25,000 vs 1,000" episode count. This means the
resulting Q-table is likely LESS converged than the 25,000-episode
simplified-environment one — that is an expected, honestly-reported
consequence of the budget-matching choice, not a hidden flaw.

Usage:
    cd backend/sumo
    python train_qlearning_sumo.py --episodes 1000 --pattern medium --seed 42

Output:
    sumo/training_logs/qlearning_sumo_<timestamp>.json
    rl/q_table_sumo.pkl (a SEPARATE file from the production
        rl/q_table_v2.pkl — this script never overwrites the production
        simplified-environment checkpoint; see --output-path to change).
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sumo_env import SumoTrafficEnv

from config import ENV_TAG_SUMO
from rl.emergency_scoring import is_emergency_action_correct
from rl.q_agent import QAgent

LOG_DIR = os.path.join(os.path.dirname(__file__), "training_logs")
os.makedirs(LOG_DIR, exist_ok=True)
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "rl", "q_table_sumo.pkl")


def get_epsilon(episode, total, start=1.0, end=0.05, frac=0.7):
    decay = int(total * frac)
    if episode >= decay:
        return end
    return start - (episode / decay) * (start - end)


def train(
    episodes=1000, pattern="medium", duration=300, seed=42, ambulance_prob=0.7, output_path=None
):
    random.seed(seed)
    np.random.seed(seed)

    agent = QAgent(epsilon=1.0)
    env = SumoTrafficEnv(
        pattern=pattern, duration=duration, ambulance_prob=ambulance_prob, use_discrete_encoder=True
    )

    episode_rewards = []
    episode_ambulance_delays = []
    emergency_total = 0
    emergency_correct = 0
    start_time = time.time()

    print("=" * 70)
    print("Q-LEARNING TRAINING IN SUMO (native, not the simplified simulator)")
    print(f"Episodes: {episodes} | Pattern: {pattern} | Seed: {seed}")
    print(f"Environment tag: {ENV_TAG_SUMO}")
    print("=" * 70)

    try:
        for ep in range(episodes):
            agent.epsilon = get_epsilon(ep, episodes)
            state = env.reset(seed=seed + ep)
            ep_reward = 0.0
            done = False

            while not done:
                action = agent.choose(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)

                if env.emergency_active and env.ambulance_zone > 0:
                    emergency_total += 1
                    if is_emergency_action_correct(action, env.ambulance_zone, env.current_zone):
                        emergency_correct += 1

                state = next_state
                ep_reward += reward

            env.close()
            episode_rewards.append(ep_reward)
            metrics = env.get_metrics()
            if metrics.get("avg_ambulance_delay") is not None:
                episode_ambulance_delays.append(metrics["avg_ambulance_delay"])

            if (ep + 1) % 20 == 0:
                avg20 = np.mean(episode_rewards[-20:])
                elapsed = time.time() - start_time
                acc = (emergency_correct / emergency_total * 100) if emergency_total > 0 else 0.0
                print(
                    f"  Ep {ep + 1:4d}/{episodes} | Reward: {ep_reward:+8.1f} | "
                    f"Avg20: {avg20:+8.1f} | EmergAcc: {acc:5.1f}% | "
                    f"eps: {agent.epsilon:.3f} | Time: {elapsed:.0f}s"
                )
    except KeyboardInterrupt:
        print("\n  Training interrupted.")
    finally:
        env.close()

    total_time = time.time() - start_time
    save_path = output_path or DEFAULT_OUTPUT_PATH
    agent.save(path=save_path)

    final_acc = (emergency_correct / emergency_total * 100) if emergency_total > 0 else 0.0
    final_avg_reward = float(np.mean(episode_rewards[-200:])) if episode_rewards else 0.0
    final_avg_delay = (
        float(np.mean(episode_ambulance_delays[-50:])) if episode_ambulance_delays else None
    )

    print("\n" + "=" * 70)
    print("Training Complete")
    print(f"Final avg reward (last 200): {final_avg_reward:.2f}")
    print(f"Emergency accuracy: {final_acc:.1f}%")
    if final_avg_delay is not None:
        print(
            f"Final avg ambulance delay (last 50 ep, real TraCI measurement): {final_avg_delay:.2f}s"
        )
    print(f"Total time: {total_time / 60:.1f} minutes")
    print("=" * 70)

    result = {
        "environment": ENV_TAG_SUMO,
        "agent": "tabular_q_learning",
        "episodes": episodes,
        "pattern": pattern,
        "seed": seed,
        "final_avg_reward_last200": round(final_avg_reward, 2),
        "final_emergency_accuracy_pct": round(final_acc, 2),
        "final_avg_ambulance_delay_s": (
            round(final_avg_delay, 3) if final_avg_delay is not None else None
        ),
        "total_time_seconds": round(total_time, 1),
        "q_table_path": save_path,
        "reward_history": [round(r, 2) for r in episode_rewards],
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"qlearning_sumo_{timestamp}.json")
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {log_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tabular Q-Learning natively in SUMO")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument(
        "--pattern", type=str, default="medium", choices=["low", "medium", "high", "rush_hour"]
    )
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ambulance-prob", type=float, default=0.7)
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Where to save the trained Q-table. Defaults to rl/q_table_sumo.pkl — "
        "NEVER the production rl/q_table_v2.pkl (that file is exclusively the "
        "simplified-environment-trained checkpoint; see this module's docstring).",
    )
    args = parser.parse_args()
    train(
        episodes=args.episodes,
        pattern=args.pattern,
        duration=args.duration,
        seed=args.seed,
        ambulance_prob=args.ambulance_prob,
        output_path=args.output_path,
    )
