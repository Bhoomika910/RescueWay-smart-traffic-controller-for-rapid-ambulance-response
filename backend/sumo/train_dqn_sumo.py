# sumo/train_dqn_sumo.py
"""
RescueWay DQN Training in SUMO.

Trains a Double DQN controller using the SUMO microscopic simulator.
The trained model is saved to rl/dqn_model.pt for deployment in the
live system (Option B2 architecture).

Usage:
    python train_dqn_sumo.py
    python train_dqn_sumo.py --episodes 500 --pattern medium
    python train_dqn_sumo.py --episodes 1000 --pattern high --lr 0.001
    python train_dqn_sumo.py --resume   # continue from saved checkpoint

Training pipeline:
    SUMO + TraCI → SumoTrafficEnv → DQN Agent → dqn_model.pt

Expected training time on CPU:
    100 episodes × 300s each ≈ 30-60 minutes
    500 episodes × 300s each ≈ 3-5 hours

The trained model can then be loaded by the live system:
    from rl.dqn_agent import DQNAgent
    agent = DQNAgent(inference_mode=True)
    agent.load()
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sumo_env import SumoTrafficEnv

from config import BENCHMARK_SEED
from rl.dqn_agent import DQNAgent
from rl.training_diagnostics import TrainingDiagnostics

# ─── Output Paths ─────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "rl", "dqn_model.pt")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(__file__), "training_logs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ─── Epsilon Schedule ─────────────────────────────────


def get_epsilon(episode, total_episodes, eps_start=1.0, eps_end=0.05, decay_fraction=0.7):
    """
    Linear epsilon decay.

    Decays from eps_start to eps_end over the first decay_fraction
    of total episodes. Remains at eps_end for the rest.
    """
    decay_episodes = int(total_episodes * decay_fraction)
    if episode >= decay_episodes:
        return eps_end
    progress = episode / decay_episodes
    return eps_start - progress * (eps_start - eps_end)


# ─── Training Loop ────────────────────────────────────


def train(
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
    save_every=50,
    resume=False,
    seed=None,
    epsilon_schedule_total=None,
    epsilon_schedule_offset=0,
    diagnostics=None,
    q_snapshot_every=10,
    start_episode=None,
    agent=None,
):
    """
    Train DQN agent in SUMO.

    Args:
        episodes: Number of training episodes
        pattern: Traffic pattern
        duration: SUMO seconds per episode
        lr: Learning rate
        gamma: Discount factor
        batch_size: Replay buffer sample size
        buffer_size: Maximum replay buffer capacity
        target_update: Steps between target network updates
        hidden_dim: Hidden layer width
        ambulance_prob: Probability of ambulance per episode
        save_every: Save checkpoint every N episodes
        resume: Whether to resume from last checkpoint
        seed: Base random seed
        epsilon_schedule_total: total episode count the epsilon decay is
            computed against. Defaults to `episodes` (this call's own
            count), preserving single-pattern behavior exactly. For
            multi-pattern training, train_multi_pattern() passes the
            GRAND TOTAL across all patterns here so epsilon decays
            smoothly across the whole run instead of resetting to 1.0 at
            the start of every pattern (bug found in technical audit —
            see backend/ARCHITECTURE.md).
        epsilon_schedule_offset: how many episodes' worth of decay to
            start this call at, so it continues where a previous
            pattern's decay left off. 0 for single-pattern training.
        diagnostics: optional rl.training_diagnostics.TrainingDiagnostics.
            If None, a fresh one is created for this call (single-pattern
            use). For multi-pattern training, train_multi_pattern() passes
            ONE shared instance across all 4 calls so action/reward/Q-value
            evidence accumulates over the whole run, not just one pattern.
        q_snapshot_every: how often (in episodes) to record probe-state
            Q-values for the policy-collapse diagnostic.
        start_episode: if given, overrides the episode index this call
            starts counting from (used for both `ep_seed = base_seed +
            episode` and, together with epsilon_schedule_offset, the
            epsilon schedule). When None (default), it's inferred from
            the highest-numbered checkpoint filename in CHECKPOINT_DIR —
            which is fragile for train_multi_pattern(): that inference
            only lands exactly on the intended pattern boundary
            (i * episodes_per_pattern) if `save_every` evenly divides
            episodes_per_pattern, since otherwise no checkpoint exists at
            exactly that episode count. train_multi_pattern() passes its
            own known-correct value explicitly instead of relying on the
            inference. Weight loading (agent.load(...)) is unaffected
            either way — it always loads whatever the actual latest
            checkpoint file is, regardless of this value.
        agent: an existing DQNAgent to keep training, instead of
            constructing (and, if resume=True, checkpoint-loading into) a
            fresh one. train_multi_pattern() uses this to carry the SAME
            agent — weights AND replay buffer — across all pattern
            stages. Previously each pattern stage got a brand-new
            DQNAgent (a fresh, empty ReplayBuffer) with only the network
            WEIGHTS transferred via checkpoint load; that meant training
            data from earlier patterns was completely discarded the
            moment a new pattern started, a textbook catastrophic-
            forgetting setup (design-review finding) since nothing in the
            replay buffer could remind the network what low/medium-
            pattern states looked like while it trained exclusively on
            high/rush_hour. Passing the same agent through fixes the root
            cause directly. When None (default), behavior is unchanged
            from before this parameter existed.
    """
    if diagnostics is None:
        diagnostics = TrainingDiagnostics()
    base_seed = seed if seed is not None else BENCHMARK_SEED

    # Seed EVERY RNG the training run depends on. random/np.random were
    # already seeded per-episode inside SumoTrafficEnv.reset(), but
    # torch's global RNG was never seeded anywhere in the codebase —
    # DQNetwork's Xavier-uniform weight initialization (rl/dqn_network.py)
    # draws from it, so two "identical seed" runs previously started from
    # different random initial weights every time, silently breaking
    # reproducibility (this is what train_multi_seed.py exists to measure
    # correctly). Seeding here, before the agent (and its network) is
    # constructed, fixes that.
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    print("=" * 70)
    print("RescueWay DQN Training in SUMO")
    print("=" * 70)
    print(f"Episodes:       {episodes}")
    print(f"Pattern:        {pattern}")
    print(f"Duration:       {duration}s per episode")
    print(f"Learning rate:  {lr}")
    print(f"Gamma:          {gamma}")
    print(f"Batch size:     {batch_size}")
    print(f"Buffer size:    {buffer_size:,}")
    print(f"Target update:  every {target_update} steps")
    print(f"Hidden dim:     {hidden_dim}")
    print(f"Ambulance prob: {ambulance_prob}")
    print(f"Base seed:      {base_seed}")
    print("=" * 70)

    # Create agent (unless the caller is carrying one across pattern
    # stages — see the `agent` parameter's docstring).
    owns_agent = agent is None
    if owns_agent:
        agent = DQNAgent(
            alpha=lr,
            gamma=gamma,
            epsilon=1.0,
            batch_size=batch_size,
            buffer_size=buffer_size,
            target_update_freq=target_update,
            hidden_dim=hidden_dim,
            inference_mode=False,
        )

    # Resume from checkpoint if requested. `detected_start_episode` is
    # only used as a fallback for the episode/seed-indexing start point
    # when the caller didn't pass an explicit `start_episode` — see that
    # parameter's docstring for why inferring it from the checkpoint
    # filename is fragile for train_multi_pattern(). Weight loading below
    # is unaffected either way. Skipped entirely when the caller passed
    # an existing `agent` — its weights (and buffer) are already exactly
    # what should continue training; re-loading from a checkpoint file
    # would be redundant at best and could load a stale/unrelated
    # checkpoint at worst.
    detected_start_episode = 0
    if resume and owns_agent:
        checkpoint_files = []
        for f in os.listdir(CHECKPOINT_DIR):
            if f.startswith("dqn_ep") and f.endswith(".pt"):
                try:
                    ep_num = int(f.split("_ep")[1].split(".")[0])
                except (ValueError, IndexError):
                    continue
                checkpoint_files.append((ep_num, f))
        # Sort numerically by episode count, not lexicographically by
        # filename — "dqn_ep450.pt" < "dqn_ep60.pt" as strings (since '4'
        # < '6'), so a plain sorted(list_of_names) picks the wrong "latest"
        # checkpoint whenever episode counts have different digit lengths.
        checkpoint_files.sort(key=lambda pair: pair[0])
        if checkpoint_files:
            detected_start_episode, latest_name = checkpoint_files[-1]
            latest = os.path.join(CHECKPOINT_DIR, latest_name)
            agent.load(latest)
            print(f"Resumed from checkpoint: {latest} (episode {detected_start_episode})")

    if start_episode is None:
        start_episode = detected_start_episode
    elif resume and start_episode != detected_start_episode:
        print(
            f"Using caller-supplied start_episode={start_episode} for episode/seed "
            f"indexing (checkpoint filename implied {detected_start_episode} — "
            f"expected when save_every doesn't evenly divide episodes_per_pattern)."
        )

    # Create environment
    env = SumoTrafficEnv(
        pattern=pattern,
        duration=duration,
        ambulance_prob=ambulance_prob,
    )

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_switches = []
    # Real per-episode ambulance delay (TraCI-measured peak waiting time
    # per emergency, see sumo_env.SumoTrafficEnv.get_metrics's
    # avg_ambulance_delay) — tracked separately from reward so the
    # proactive-delay objective can be verified directly (reward can rise
    # from congestion/throughput terms alone; this cannot). NOTE: prior to
    # the fix in sumo_env.py, this read the instantaneous
    # `ambulance_waiting_time` snapshot instead, which is ~always 0 at
    # episode end because it resets the moment the ambulance clears —
    # every historical training log's `final_avg_ambulance_delay_20` is
    # 0.0 as a result. `avg_ambulance_delay` aggregates the peak wait
    # recorded per emergency instead, so it reflects real experienced delay.
    episode_ambulance_delays = []
    best_avg_reward = -float("inf")
    training_start = time.time()

    epsilon_total = epsilon_schedule_total if epsilon_schedule_total is not None else episodes

    try:
        for episode in range(start_episode, start_episode + episodes):
            # Set epsilon for this episode. Computed against
            # epsilon_total/epsilon_schedule_offset (not `episode -
            # start_episode` alone) so that in multi-pattern training the
            # decay continues smoothly across pattern boundaries instead
            # of resetting to 1.0 every time train() is called for a new
            # pattern — see epsilon_schedule_total's docstring above.
            agent.epsilon = get_epsilon(
                epsilon_schedule_offset + (episode - start_episode), epsilon_total
            )

            # Episode seed (deterministic per episode index)
            ep_seed = base_seed + episode

            # Reset environment
            state = env.reset(seed=ep_seed)
            episode_reward = 0
            episode_steps = 0
            done = False

            # Run episode
            while not done:
                # Agent selects action
                action = agent.choose(state)

                # Environment executes
                next_state, reward, done = env.step(action)

                # Agent learns
                agent.update(state, action, reward, next_state, done, diagnostics=diagnostics)

                diagnostics.record_step(action, reward, env.emergency_active, env.ambulance_zone)

                state = next_state
                episode_reward += reward
                episode_steps += 1

            # Close SUMO for this episode
            metrics = env.get_metrics()
            env.close()

            if episode % q_snapshot_every == 0:
                diagnostics.snapshot_q_values(episode, agent)

            # Track metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            episode_switches.append(metrics.get("switches", 0))
            episode_ambulance_delays.append(metrics.get("avg_ambulance_delay", 0))

            # Logging
            avg_reward_20 = np.mean(episode_rewards[-20:])
            avg_delay_20 = np.mean(episode_ambulance_delays[-20:])
            elapsed = time.time() - training_start
            stats = agent.get_training_stats()

            print(
                f"  Ep {episode+1:4d}/{start_episode+episodes} | "
                f"Reward: {episode_reward:+8.1f} | "
                f"Avg(20): {avg_reward_20:+8.1f} | "
                f"AmbDelay(20): {avg_delay_20:5.1f}s | "
                f"Steps: {episode_steps:3d} | "
                f"eps: {agent.epsilon:.3f} | "
                f"Loss: {stats['avg_loss_100']:.4f} | "
                f"Buffer: {stats['buffer_size']:,} | "
                f"Time: {elapsed:.0f}s"
            )

            # Save checkpoint periodically
            if (episode + 1) % save_every == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"dqn_ep{episode+1}.pt")
                agent.save(ckpt_path)

            # Save best model
            if len(episode_rewards) >= 20:
                if avg_reward_20 > best_avg_reward:
                    best_avg_reward = avg_reward_20
                    agent.save(MODEL_PATH)
                    print(f"New best model saved (avg reward: {avg_reward_20:+.1f})")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    finally:
        env.close()

    # Final save
    agent.epsilon = 0.05  # Set to inference epsilon
    agent.save(MODEL_PATH)

    # Save training log
    total_time = time.time() - training_start
    log = {
        "config": {
            "episodes": episodes,
            "pattern": pattern,
            "duration": duration,
            "lr": lr,
            "gamma": gamma,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "target_update": target_update,
            "hidden_dim": hidden_dim,
            "ambulance_prob": ambulance_prob,
            "base_seed": base_seed,
        },
        "results": {
            "total_episodes": len(episode_rewards),
            "total_time_seconds": round(total_time, 1),
            "final_avg_reward_20": (
                round(np.mean(episode_rewards[-20:]), 2) if episode_rewards else 0
            ),
            "best_avg_reward_20": round(best_avg_reward, 2),
            "final_avg_ambulance_delay_20": (
                round(np.mean(episode_ambulance_delays[-20:]), 2) if episode_ambulance_delays else 0
            ),
            "final_epsilon": round(agent.epsilon, 4),
            "train_steps": agent.train_step_count,
            "buffer_final_size": len(agent.replay_buffer),
        },
        "reward_history": [round(r, 2) for r in episode_rewards],
        "ambulance_delay_history": [round(d, 2) for d in episode_ambulance_delays],
        "diagnostics": diagnostics.summary(),
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"training_{timestamp}.json")
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)

    diagnostics.print_report()

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total episodes:     {len(episode_rewards)}")
    print(f"Total time:         {total_time/60:.1f} minutes")
    print(f"Best avg reward:    {best_avg_reward:+.1f}")
    print(
        f"Final avg reward:   {np.mean(episode_rewards[-20:]):+.1f}" if episode_rewards else "N/A"
    )
    print(
        f"Final avg ambulance delay (last 20 ep): "
        f"{np.mean(episode_ambulance_delays[-20:]):.1f}s"
        if episode_ambulance_delays
        else "N/A"
    )
    print(f"Train steps:        {agent.train_step_count:,}")
    print(f"Buffer size:        {len(agent.replay_buffer):,}")
    print(f"Model saved to:     {MODEL_PATH}")
    print(f"Log saved to:       {log_file}")
    print("=" * 70)

    return log


# ─── Multi-Pattern Training ──────────────────────────


def train_multi_pattern(
    episodes_per_pattern=100,
    held_out_pattern="rush_hour",
    run_forgetting_check=True,
    forgetting_check_episodes=10,
    **kwargs,
):
    """
    Train across traffic patterns for generalization.

    Cycles through the training patterns (all 4 minus held_out_pattern),
    running episodes_per_pattern for each, using ONE shared DQNAgent
    across every stage — its replay buffer carries forward between
    patterns instead of resetting (see train()'s `agent` parameter
    docstring for why a fresh buffer per stage was a real, previously-
    unaddressed catastrophic-forgetting risk).

    held_out_pattern: excluded from training entirely and used for a
    genuine generalization check afterward, the same held-out-pattern
    design used by train_offline.py's Q-Learning training (design-review
    finding: this codebase had zero train/test separation anywhere).
    Pass None to train on all 4 patterns, matching pre-fix behavior.

    run_forgetting_check: if True, after training completes, evaluates
    the FINAL checkpoint separately on every pattern it trained on (not
    just the last one) plus the held-out pattern, at
    forgetting_check_episodes episodes each — directly answering "did
    performance on earlier patterns regress while training on later
    ones," which this codebase previously never measured at all
    (absence of forgetting evidence was being read as absence of
    forgetting, which isn't valid). Uses real SUMO episodes, so this
    adds forgetting_check_episodes * (n_patterns) SUMO episodes of
    runtime — kept modest by default since it's a diagnostic check, not
    the headline benchmark (see sumo/evaluate_in_sumo_v2.py for that).
    """
    all_patterns = ["low", "medium", "high", "rush_hour"]
    patterns = (
        [p for p in all_patterns if p != held_out_pattern]
        if held_out_pattern
        else list(all_patterns)
    )
    total_episodes = episodes_per_pattern * len(patterns)

    print("=" * 70)
    print("MULTI-PATTERN DQN TRAINING")
    print(f"Training patterns: {patterns}")
    print(
        f"Held-out pattern (generalization test): {held_out_pattern or 'NONE — trained on all 4'}"
    )
    print(f"Episodes per pattern: {episodes_per_pattern}")
    print("=" * 70)

    # One shared instance across all patterns (see train()'s docstring)
    # so action/reward/Q-value evidence accumulates over the whole run
    # instead of resetting — without this, each train() call below would
    # default to a fresh TrainingDiagnostics() and only the last pattern's
    # evidence would survive.
    diagnostics = TrainingDiagnostics()

    # One shared DQNAgent across all patterns — see train()'s `agent`
    # docstring. Constructed with the same defaults train() itself would
    # use, pulling any override kwargs the caller passed (lr/gamma/
    # batch_size/etc.) so this matches what train() would have built.
    shared_agent = DQNAgent(
        alpha=kwargs.get("lr", 5e-4),
        gamma=kwargs.get("gamma", 0.95),
        epsilon=1.0,
        batch_size=kwargs.get("batch_size", 64),
        buffer_size=kwargs.get("buffer_size", 100_000),
        target_update_freq=kwargs.get("target_update", 200),
        hidden_dim=kwargs.get("hidden_dim", 128),
        inference_mode=False,
    )

    all_logs = []
    for i, pattern in enumerate(patterns):
        print(f"\n{'-' * 50}")
        print(f"Training on pattern: {pattern}")
        print(f"{'-' * 50}")
        log = train(
            episodes=episodes_per_pattern,
            pattern=pattern,
            agent=shared_agent,
            # resume is now irrelevant to weight/buffer continuity (the
            # shared agent object already carries both) — kept False so
            # train() never attempts a checkpoint-file load, which would
            # be redundant at best and load a stale unrelated checkpoint
            # at worst (see train()'s resume-skip-when-agent-passed logic).
            resume=False,
            # Epsilon decays smoothly across the FULL multi-pattern run
            # (bug fix — see train()'s docstring) instead of resetting to
            # 1.0 at the start of every pattern.
            epsilon_schedule_total=total_episodes,
            epsilon_schedule_offset=i * episodes_per_pattern,
            start_episode=i * episodes_per_pattern,
            diagnostics=diagnostics,
            **kwargs,
        )
        all_logs.append({"pattern": pattern, "log": log})

    # Final save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"multi_pattern_{timestamp}.json")

    forgetting_check = None
    if run_forgetting_check:
        print(f"\n{'=' * 70}")
        print("CATASTROPHIC-FORGETTING CHECK")
        print(
            f"Evaluating final checkpoint on every trained pattern + held-out "
            f"({forgetting_check_episodes} episodes each)"
        )
        print(f"{'=' * 70}")
        forgetting_check = _run_per_pattern_check(
            shared_agent,
            all_patterns,
            held_out_pattern,
            n_episodes=forgetting_check_episodes,
            duration=kwargs.get("duration", 300),
        )

    with open(log_file, "w") as f:
        json.dump({"pattern_logs": all_logs, "forgetting_check": forgetting_check}, f, indent=2)

    print(f"\nMulti-pattern training complete. Log: {log_file}")
    return {"pattern_logs": all_logs, "forgetting_check": forgetting_check}


def _run_per_pattern_check(agent, all_patterns, held_out_pattern, n_episodes, duration):
    """
    Runs sumo/evaluate_in_sumo_v2.py's SUMODQNController evaluation
    against every pattern in all_patterns (trained-on and held-out alike)
    using the given agent's CURRENT weights, so a caller can directly see
    whether earlier-trained patterns regressed. Returns a dict keyed by
    pattern with each pattern's accuracy/delay, plus which patterns were
    actually trained on vs held out for easy comparison.
    """
    # Imported lazily to avoid a hard import-time dependency between
    # train_dqn_sumo.py and evaluate_in_sumo_v2.py for callers that only
    # ever use one of the two.
    from evaluate_in_sumo_v2 import SUMODQNController, evaluate_controller

    ctrl = SUMODQNController(agent)
    results = {}
    for pattern in all_patterns:
        print(
            f"\n  Pattern: {pattern} ({'HELD OUT' if pattern == held_out_pattern else 'trained on'})"
        )
        result = evaluate_controller(ctrl, episodes=n_episodes, pattern=pattern, duration=duration)
        results[pattern] = {
            "role": "held_out" if pattern == held_out_pattern else "trained_on",
            "reward_mean": result["reward"]["mean"],
            "emergency_accuracy_mean": result["emergency"]["accuracy"],
            "avg_ambulance_delay_mean": result["ambulance_delay"]["mean"],
        }
        print(
            f"    reward={result['reward']['mean']:+.1f} "
            f"emerg_acc={result['emergency']['accuracy']:.1f}% "
            f"amb_delay={result['ambulance_delay']['mean']:.1f}s"
        )

    return results


# ─── Main ────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent in SUMO for RescueWay")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    parser.add_argument(
        "--pattern",
        type=str,
        default="medium",
        choices=["low", "medium", "high", "rush_hour", "all"],
        help="Traffic pattern (use 'all' for multi-pattern)",
    )
    parser.add_argument("--duration", type=int, default=300, help="SUMO seconds per episode")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="Replay buffer batch size")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer capacity")
    parser.add_argument(
        "--target-update",
        type=int,
        default=200,
        help="Steps between target network updates (lowered "
        "from 1000 — see DQNAgent's docstring)",
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer width")
    parser.add_argument(
        "--ambulance-prob", type=float, default=0.7, help="Ambulance probability per episode"
    )
    parser.add_argument(
        "--save-every", type=int, default=50, help="Save checkpoint every N episodes"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")

    args = parser.parse_args()

    # Check SUMO network file
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "intersection.net.xml")):
        print("ERROR: intersection.net.xml not found!")
        print("Run build_network.bat first.")
        sys.exit(1)

    if args.pattern == "all":
        train_multi_pattern(
            episodes_per_pattern=args.episodes // 4,
            duration=args.duration,
            lr=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            target_update=args.target_update,
            hidden_dim=args.hidden_dim,
            ambulance_prob=args.ambulance_prob,
            save_every=args.save_every,
            seed=args.seed,
        )
    else:
        train(
            episodes=args.episodes,
            pattern=args.pattern,
            duration=args.duration,
            lr=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            target_update=args.target_update,
            hidden_dim=args.hidden_dim,
            ambulance_prob=args.ambulance_prob,
            save_every=args.save_every,
            resume=args.resume,
            seed=args.seed,
        )
