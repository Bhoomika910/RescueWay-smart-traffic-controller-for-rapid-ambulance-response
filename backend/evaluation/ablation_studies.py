"""
RescueWay Reward/State Sensitivity Studies + one genuine retrained ablation.

Naming note (design-review finding): Studies A and B below were previously
both labeled "ablation studies," but neither one retrains anything — they
re-score an already-trained, frozen policy under a different reward scale
(A) or compare two already-trained agents' encodings (B). "Ablation" in
the standard ML sense means removing/disabling a component and
RETRAINING to isolate its causal effect on the learned policy — what A/B
actually measure is real and useful, just a different, narrower question
("how sensitive is a fixed policy's *score* to a scale/representation
change", not "what would the agent have learned without this component").
They are renamed below to describe what they actually do, and a third,
genuine retrained ablation (Study C) was added specifically to make a
causal claim possible.

A) REWARD-SCALE EVALUATION SENSITIVITY (renamed from "ablation")
   Sweeps the emergency reward magnitude (currently +25/−25) from
   0.5× to 3× baseline, RE-SCORING the same already-trained DQN
   checkpoint at each level. Tests how sensitive the *reported reward
   metric* is to the reward function's scale — not whether the agent
   would learn differently under a different scale.

B) STATE-REPRESENTATION COMPARISON (renamed from "ablation")
   Compares two independently-trained agents (DQN's continuous 19-dim
   encoder vs. Q-Learning's discrete 4320-state encoder) side by side.
   Informative about the representational difference between the two
   already-existing checkpoints, but conflates state representation with
   the DQN-vs-QL algorithm difference (also covered in Study 5) — it is
   not a controlled single-factor ablation of representation alone.

C) REWARD-COMPONENT ABLATION (NEW — genuinely retrained)
   Trains fresh tabular Q-Learning agents from scratch, from identical
   seeds, with the ambulance-delay reward term either present (baseline)
   or entirely removed via TrafficEnv(disable_ambulance_delay_reward=True)
   (ablated) — see rl/environment.py. Both conditions are evaluated on
   the SAME held-out seeds, always scored by the real (non-ablated)
   reward/ambulance-delay outcome, so any difference in the resulting
   avg_ambulance_delay/stop-rate is genuine causal evidence about what
   that reward term contributes to the learned policy — not a post-hoc
   rescoring of one frozen policy.

Usage:
    python -m evaluation.ablation_studies --study A --seeds 30
    python -m evaluation.ablation_studies --study B --seeds 30
    python -m evaluation.ablation_studies --study C --seeds 3 --episodes 10000
    python -m evaluation.ablation_studies --study all

Output:
    results/ablation/ablation_<study>_<timestamp>.json
"""

import argparse
import json
import os
import random as stdlib_random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BENCHMARK_SEED, ENV_TAG_SIMPLIFIED
from controllers.dqn_controller import DQNController
from controllers.rl_controller import RLController
from evaluation.robustness_runner import evaluate_condition
from noise import NoiseConfig
from rl import reward as reward_module
from rl.dqn_agent import DQNAgent
from rl.environment import TrafficEnv
from rl.q_agent import QAgent
from rl.state_encoder import StateEncoder
from rl.state_features import StateFeatureExtractor

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "ablation")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════
# STUDY A: REWARD WEIGHT SENSITIVITY
# ═══════════════════════════════════════════════════════

# Save originals so we can restore after each sweep point
_ORIGINAL_EMERGENCY = {
    "correct_zone": 25.0,
    "extend_correct": 10.0,
    "extend_wrong": -8.0,
    "normal_during_emg": -15.0,
    "wrong_zone": -25.0,
}


def _patch_emergency_reward(scale):
    """
    Monkey-patch the emergency reward constants in reward_module.
    This is ugly but avoids modifying the core reward.py file.
    We restore after each evaluation.

    Bug fixed here (first occurrence): `patched` previously only accepted
    3 positional arguments (action, amb_zone, prev_signal), but
    rl.reward.compute_reward always calls the real _emergency_reward with
    a 4th positional argument (ambulance_delay) — so this monkeypatch
    crashed with `TypeError: patched() takes 3 positional arguments but 4
    were given` on every single call while it was installed, meaning
    run_study_a could never complete. It also duplicated
    _ORIGINAL_EMERGENCY's values as separate hardcoded literals instead of
    referencing that dict — now sourced from it directly, so there's one
    copy to keep in sync with rl/reward.py's real constants instead of
    three.

    Bug fixed here (second occurrence, same class): rl/reward.py's later
    reward-weight-sensitivity refactor (RewardWeights) added a 5th
    positional parameter (`weights`) to the real _emergency_reward, which
    would have reintroduced the identical TypeError this docstring already
    describes fixing once. `patched` accepts (and ignores) `weights` for
    that reason — this monkeypatch tests reward-SCALE sensitivity on a
    frozen, already-trained policy (Study A), an orthogonal question to
    the RewardWeights per-weight sensitivity sweep
    (evaluation/reward_sensitivity.py), so it deliberately keeps sourcing
    its own scaled constants from _ORIGINAL_EMERGENCY rather than reading
    `weights` — see run_study_a's docstring for why these are separate
    studies, not a redundancy.
    """
    original_fn = reward_module._emergency_reward

    def patched(action, amb_zone, prev_signal, ambulance_delay=None, weights=None):
        if amb_zone == 0:
            return 0.0
        target = amb_zone - 1
        if action == target:
            reward = _ORIGINAL_EMERGENCY["correct_zone"] * scale
        elif action == 4 and prev_signal == amb_zone:
            reward = _ORIGINAL_EMERGENCY["extend_correct"] * scale
        elif action == 4 and prev_signal != amb_zone:
            reward = _ORIGINAL_EMERGENCY["extend_wrong"] * scale
        elif action == 5:
            reward = _ORIGINAL_EMERGENCY["normal_during_emg"] * scale
        else:
            reward = _ORIGINAL_EMERGENCY["wrong_zone"] * scale

        # Ambulance-delay term is NOT part of what this study sweeps
        # (only the base zone-correctness magnitude is scaled) — mirrored
        # unscaled from the real _emergency_reward for consistency on the
        # rare path that does pass a real delay value.
        if ambulance_delay is not None:
            if ambulance_delay <= 0.5:
                reward += 40.0
            else:
                capped_delay = min(ambulance_delay, reward_module.MAX_DELAY_FOR_PENALTY)
                reward -= 4.0 * capped_delay

        return reward

    reward_module._emergency_reward = patched
    return original_fn


def run_study_a(n_seeds=30):
    """
    Sweep emergency reward weight multiplier.
    Evaluates the EXISTING trained DQN (trained at scale=1.0)
    under modified reward weights to see how sensitive the
    reward metric is to the reward-function constants.

    NOT an ablation (see module docstring): this re-scores one frozen,
    already-trained policy — the policy itself never changes across
    scales, only the yardstick used to score its fixed behavior.
    """
    print("=" * 70)
    print("STUDY A: REWARD-SCALE EVALUATION SENSITIVITY (not a retrained ablation)")
    print(f"Seeds: {n_seeds}")
    print("=" * 70)

    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

    # Create clean DQN controller
    agent = DQNAgent(epsilon=0.0, inference_mode=True)
    agent.feature_extractor = StateFeatureExtractor(noise_config=NoiseConfig())
    agent.load()
    ctrl = DQNController(agent=agent)

    results = {}
    for scale in scales:
        print(f"\n  Scale: {scale:.2f}×", end=" ", flush=True)

        # Patch reward
        original_fn = _patch_emergency_reward(scale)

        try:
            result = evaluate_condition(ctrl, NoiseConfig(), n_seeds=n_seeds)
            results[str(scale)] = {
                "reward": result["reward"],
                "emergency_accuracy": result["emergency_accuracy"],
                "avg_queue": result["avg_queue"],
            }
            print(
                f"reward={result['reward']['mean']:+.1f} "
                f"emerg={result['emergency_accuracy']['mean']:.1f}%"
            )
        finally:
            # Restore original
            reward_module._emergency_reward = original_fn

    print(f"\n{'-'*70}")
    print("  Summary: Emergency Reward Scale vs Performance")
    print(f"{'-'*70}")
    for scale in scales:
        r = results[str(scale)]
        print(
            f"    {scale:5.2f}× | reward={r['reward']['mean']:>+8.1f} ± {r['reward']['ci_95']:<5.1f} | "
            f"emerg={r['emergency_accuracy']['mean']:>5.1f}%"
        )

    return {
        "study": "A",
        "description": "Reward-scale evaluation sensitivity (re-scores one frozen "
        "trained policy at different reward scales — NOT a retrained "
        "ablation; see Study C for a genuine reward-component ablation)",
        "results": results,
    }


# ═══════════════════════════════════════════════════════
# STUDY B: STATE REPRESENTATION INFORMATION ANALYSIS
# ═══════════════════════════════════════════════════════


def run_study_b(n_seeds=30):
    """
    Compare DQN (continuous 19-dim) vs Q-Learning (discrete 4320-state)
    state representations — information retention analysis.

    Also runs both controllers side-by-side under clean + noisy conditions
    to quantify the state-representation contribution separately from
    the algorithm contribution.

    NOT a controlled single-factor ablation (see module docstring): this
    compares two independently-trained agents that differ in BOTH state
    representation AND algorithm (DQN vs tabular Q-Learning) — it cannot
    cleanly isolate the representation variable from the algorithm
    variable, which Study 5 (robustness_runner.py) already covers.
    """
    print("=" * 70)
    print("STUDY B: STATE-REPRESENTATION COMPARISON (not a controlled ablation)")
    print(f"Seeds: {n_seeds}")
    print("=" * 70)

    from noise import NOISE_PRESETS

    # Create both controllers with clean noise
    dqn_agent = DQNAgent(epsilon=0.0, inference_mode=True)
    dqn_agent.feature_extractor = StateFeatureExtractor(noise_config=NoiseConfig())
    dqn_agent.load()
    dqn_ctrl = DQNController(agent=dqn_agent)

    from rl.q_agent import QAgent

    ql_agent = QAgent(epsilon=0.0)
    ql_agent.encoder = StateEncoder(noise_config=NoiseConfig())
    ql_agent.load()
    ql_ctrl = RLController(agent=ql_agent)

    # Information analysis: how much variance does each encoding preserve?
    print("\n  A. Information retention analysis (100 random states):")
    continuous_encoder = StateFeatureExtractor(noise_config=NoiseConfig())
    discrete_encoder = StateEncoder(noise_config=NoiseConfig())

    np.random.seed(42)
    continuous_states = set()
    discrete_states = set()

    for _ in range(1000):
        queues = {z: np.random.randint(0, 50) for z in [1, 2, 3, 4]}
        emg = np.random.random() < 0.2
        amb_z = np.random.randint(1, 5) if emg else 0
        sig = np.random.randint(1, 5)

        c_state = tuple(continuous_encoder.extract(emg, amb_z, sig, queues).tolist())
        d_state = discrete_encoder.encode(emg, amb_z, sig, queues)

        continuous_states.add(c_state)
        discrete_states.add(d_state)

    print("    From 1000 random traffic states:")
    print(
        f"    Continuous (19-dim): {len(continuous_states)} unique states "
        f"({len(continuous_states)/1000*100:.1f}% unique)"
    )
    print(
        f"    Discrete ({discrete_encoder.total_states}):    {len(discrete_states)} unique states "
        f"({len(discrete_states)/1000*100:.1f}% unique)"
    )
    print(f"    Discrete capacity:  {discrete_encoder.total_states} possible states")
    print(
        f"    Compression ratio:  {len(continuous_states)/max(len(discrete_states),1):.1f}× more distinct states in continuous"
    )

    # Side-by-side evaluation across noise presets
    print("\n  B. Performance comparison across noise presets:")

    presets_to_test = ["clean", "suburban", "urban_dense", "degraded"]
    results = {
        "information_analysis": {
            "n_samples": 1000,
            "continuous_unique": len(continuous_states),
            "discrete_unique": len(discrete_states),
            "discrete_capacity": discrete_encoder.total_states,
        },
        "performance": {},
    }

    for preset_name in presets_to_test:
        config = NOISE_PRESETS[preset_name]
        print(f"\n    Preset: {preset_name}")

        dqn_result = evaluate_condition(dqn_ctrl, config, n_seeds=n_seeds)
        ql_result = evaluate_condition(ql_ctrl, config, n_seeds=n_seeds)

        results["performance"][preset_name] = {
            "dqn_continuous": {
                "reward": dqn_result["reward"],
                "emergency_accuracy": dqn_result["emergency_accuracy"],
            },
            "ql_discrete": {
                "reward": ql_result["reward"],
                "emergency_accuracy": ql_result["emergency_accuracy"],
            },
        }

        dqn_r = dqn_result["reward"]["mean"]
        ql_r = ql_result["reward"]["mean"]
        dqn_e = dqn_result["emergency_accuracy"]["mean"]
        ql_e = ql_result["emergency_accuracy"]["mean"]
        print(f"      DQN (continuous): reward={dqn_r:+.1f} emerg={dqn_e:.1f}%")
        print(f"      QL  (discrete):   reward={ql_r:+.1f} emerg={ql_e:.1f}%")

    return {
        "study": "B",
        "description": "State-representation comparison between two independently "
        "trained agents (confounds representation with algorithm — "
        "NOT a controlled single-factor ablation; see Study C for a "
        "genuine retrained ablation)",
        "results": results,
    }


# ═══════════════════════════════════════════════════════
# STUDY C: REWARD-COMPONENT ABLATION (genuinely retrained)
# ═══════════════════════════════════════════════════════


def _train_qagent(episodes, seed, disable_ambulance_delay_reward):
    """Train one fresh QAgent from scratch. Returns the trained agent plus
    its own training-time ambulance-delay trend (not used for the
    headline comparison — that comes from a separate held-out evaluation
    below — but useful to see the trajectory)."""
    stdlib_random.seed(seed)
    np.random.seed(seed)

    agent = QAgent(alpha=0.1, gamma=0.95, epsilon=1.0)
    env = TrafficEnv(disable_ambulance_delay_reward=disable_ambulance_delay_reward)
    epsilon_end = 0.05
    epsilon_decay = 1.0 / (episodes * 0.7)

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        agent.epsilon = max(epsilon_end, agent.epsilon - epsilon_decay)

    return agent


def _evaluate_qagent_ambulance_outcome(agent, n_eval_episodes, eval_seed):
    """
    Evaluate a trained agent's REAL (always-unablated) ambulance-delay
    outcome on held-out episodes — held out in the sense of using a
    distinct seed range from training, not a different traffic
    distribution (see the design-review's separate train/test-split
    finding, addressed elsewhere). Both conditions (baseline/ablated) are
    scored with this exact same function, on the exact same seeds, so the
    only thing that differs between the two numbers this produces is what
    each agent learned during training.
    """
    agent_epsilon = agent.epsilon
    agent.epsilon = 0.0  # greedy for evaluation, matching every other benchmark in this codebase
    try:
        stdlib_random.seed(eval_seed)
        np.random.seed(eval_seed)
        env = TrafficEnv(disable_ambulance_delay_reward=False)  # always score on the real objective

        all_delays = []
        stop_count = 0
        for ep in range(n_eval_episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.choose(state)
                state, reward, done, info = env.step(action)
            all_delays.extend(env.episode_ambulance_delays)
            stop_count += env.episode_ambulance_stops

        return {
            "ambulance_trips": len(all_delays),
            "avg_ambulance_delay": round(float(np.mean(all_delays)), 3) if all_delays else 0.0,
            "ambulance_stop_rate": round(stop_count / len(all_delays), 3) if all_delays else 0.0,
        }
    finally:
        agent.epsilon = agent_epsilon


def run_study_c(n_seeds=3, episodes=10000, n_eval_episodes=300):
    """
    Genuine retrained ablation: does removing the ambulance-delay reward
    term from training change what the agent actually learns to do?

    For each of n_seeds independent seeds, trains TWO fresh QAgents from
    scratch from that same seed — one with the real reward (baseline),
    one with TrafficEnv(disable_ambulance_delay_reward=True) (ablated) —
    then evaluates both on the same held-out seed, always scored by the
    real ambulance-delay outcome. Reports paired baseline-vs-ablated
    deltas across seeds, the way every other comparative claim in this
    codebase now does (see Study 1/3/5's paired-seed methodology).
    """
    print("=" * 70)
    print("STUDY C: REWARD-COMPONENT ABLATION (genuinely retrained)")
    print(f"Seeds: {n_seeds} | Episodes/run: {episodes} | Eval episodes: {n_eval_episodes}")
    print("=" * 70)

    baseline_delays, ablated_delays = [], []
    baseline_stop_rates, ablated_stop_rates = [], []

    for i in range(n_seeds):
        train_seed = BENCHMARK_SEED + i
        eval_seed = BENCHMARK_SEED + 10_000 + i  # distinct range from training seeds

        print(
            f"\n  Seed {i+1}/{n_seeds} (train_seed={train_seed}) — training baseline...", flush=True
        )
        t0 = time.time()
        baseline_agent = _train_qagent(episodes, train_seed, disable_ambulance_delay_reward=False)
        print(f"    done in {time.time()-t0:.1f}s")

        print(
            f"  Seed {i+1}/{n_seeds} (train_seed={train_seed}) — training ablated (no delay term)...",
            flush=True,
        )
        t0 = time.time()
        ablated_agent = _train_qagent(episodes, train_seed, disable_ambulance_delay_reward=True)
        print(f"    done in {time.time()-t0:.1f}s")

        baseline_result = _evaluate_qagent_ambulance_outcome(
            baseline_agent, n_eval_episodes, eval_seed
        )
        ablated_result = _evaluate_qagent_ambulance_outcome(
            ablated_agent, n_eval_episodes, eval_seed
        )

        baseline_delays.append(baseline_result["avg_ambulance_delay"])
        ablated_delays.append(ablated_result["avg_ambulance_delay"])
        baseline_stop_rates.append(baseline_result["ambulance_stop_rate"])
        ablated_stop_rates.append(ablated_result["ambulance_stop_rate"])

        print(
            f"    baseline: delay={baseline_result['avg_ambulance_delay']:.2f}s "
            f"stop_rate={baseline_result['ambulance_stop_rate']*100:.1f}%"
        )
        print(
            f"    ablated:  delay={ablated_result['avg_ambulance_delay']:.2f}s "
            f"stop_rate={ablated_result['ambulance_stop_rate']*100:.1f}%"
        )

    from evaluation.statistical_analysis import paired_comparison

    delay_comparison = paired_comparison(
        "baseline",
        baseline_delays,
        "ablated (no delay term)",
        ablated_delays,
        "avg_ambulance_delay",
    )
    stop_rate_comparison = paired_comparison(
        "baseline",
        baseline_stop_rates,
        "ablated (no delay term)",
        ablated_stop_rates,
        "ambulance_stop_rate",
    )

    print(f"\n{'-'*70}")
    print("  Summary (paired across seeds, baseline vs ablated):")
    print(f"{'-'*70}")
    print(
        f"    avg_ambulance_delay:  baseline={np.mean(baseline_delays):.2f}s  "
        f"ablated={np.mean(ablated_delays):.2f}s  "
        f"p={delay_comparison['paired_ttest']['p_value']:.4f}"
    )
    print(
        f"    ambulance_stop_rate:  baseline={np.mean(baseline_stop_rates)*100:.1f}%  "
        f"ablated={np.mean(ablated_stop_rates)*100:.1f}%  "
        f"p={stop_rate_comparison['paired_ttest']['p_value']:.4f}"
    )

    return {
        "study": "C",
        "description": "Reward-component ablation — genuinely retrained (not a "
        "re-scoring of a frozen policy). Removes the ambulance-delay "
        "reward term from training entirely via "
        "TrafficEnv(disable_ambulance_delay_reward=True) and compares "
        "the resulting trained policy's real, held-out ambulance-delay "
        "outcome against an identically-seeded baseline.",
        "config": {"n_seeds": n_seeds, "episodes": episodes, "n_eval_episodes": n_eval_episodes},
        "per_seed": {
            "baseline_avg_ambulance_delay": baseline_delays,
            "ablated_avg_ambulance_delay": ablated_delays,
            "baseline_stop_rate": baseline_stop_rates,
            "ablated_stop_rate": ablated_stop_rates,
        },
        "significance": {
            "avg_ambulance_delay": delay_comparison,
            "ambulance_stop_rate": stop_rate_comparison,
        },
    }


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RescueWay Reward/State Sensitivity Studies + Reward-Component Ablation"
    )
    parser.add_argument("--study", type=str, default="all", choices=["A", "B", "C", "all"])
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
        help="Used by A/B (evaluation seeds) and C (independent training seeds, default overridden to 3 for C unless --seeds is passed explicitly)",
    )
    parser.add_argument(
        "--episodes", type=int, default=10000, help="Training episodes per run for Study C only"
    )
    args = parser.parse_args()

    start = time.time()
    all_results = {}

    if args.study in ("A", "all"):
        r = run_study_a(n_seeds=args.seeds)
        all_results["A"] = r

    if args.study in ("B", "all"):
        r = run_study_b(n_seeds=args.seeds)
        all_results["B"] = r

    if args.study in ("C", "all"):
        # Study C trains 2 fresh agents per seed from scratch — 30 seeds
        # (A/B's default) would mean 60 full training runs. Default to a
        # smaller-but-still-real 3 seeds for "all" unless the caller
        # explicitly asked for a specific --seeds count.
        c_seeds = args.seeds if args.study == "C" else min(args.seeds, 3)
        r = run_study_c(n_seeds=c_seeds, episodes=args.episodes)
        all_results["C"] = r

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(RESULTS_DIR, f"ablation_{args.study}_{timestamp}.json")
    all_results["environment"] = ENV_TAG_SIMPLIFIED
    with open(filepath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {filepath}")
    print(f"Total time: {(time.time()-start)/60:.1f} minutes")
