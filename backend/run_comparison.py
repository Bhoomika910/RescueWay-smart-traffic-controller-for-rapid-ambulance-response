# Quick CLI wrapper around evaluation.comparator.ControllerComparator (the
# flagship, non-legacy 5-controller comparison — see routers/comparison.py
# for the same thing exposed as POST /compare/run).
# run_comparison.py

from evaluation.comparator import ControllerComparator

comparator = ControllerComparator()
results = comparator.run_comparison(episodes=200)
summary = comparator.get_summary()

print("\n")
print("=" * 70)
print("CONTROLLER COMPARISON RESULTS")
print("=" * 70)

for name, metrics in summary["controllers"].items():
    print(f"\n{metrics['name']}:")
    print(f"  Avg Reward:        {metrics['avg_reward']:+.2f}")
    print(f"  Emergency Accuracy: {metrics['emergency_accuracy']:.1f}%")
    print(f"  Avg Queue Length:   {metrics['avg_queue_length']:.2f}")
    print(f"  Avg Throughput:     {metrics['avg_throughput']:.1f}")
    print(f"  Switches/Episode:   {metrics['avg_switches_per_episode']:.1f}")

print(f"\nBest Controller: {summary['best_controller']}")

# rl_improvements' actual keys are dqn_reward/baseline_reward (not
# rl_reward), and improvement_pct can be None when a baseline's reward is
# ~0 (see ControllerComparator._reward_gain's docstring) — this used to
# crash with KeyError: 'rl_reward' and, separately, TypeError comparing
# None > 0 whenever that happened.
print("\nDQN Improvements Over Baselines:")
for baseline, imp in summary["rl_improvements"].items():
    pct = imp["improvement_pct"]
    if pct is None:
        print(
            f"  vs {baseline:12s}: n/a (baseline reward ~0, see absolute_gain)  "
            f"(DQN: {imp['dqn_reward']:.1f} vs {imp['baseline_reward']:.1f}, "
            f"absolute gain: {imp['absolute_gain']:+.1f})"
        )
    else:
        sign = "+" if pct > 0 else ""
        print(
            f"  vs {baseline:12s}: {sign}{pct}%  "
            f"(DQN: {imp['dqn_reward']:.1f} vs {imp['baseline_reward']:.1f})"
        )
