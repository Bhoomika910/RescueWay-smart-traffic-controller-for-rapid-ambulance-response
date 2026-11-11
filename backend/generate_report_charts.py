# LEGACY: quick-look utility script for the custom-simulator benchmark.
# Kept for ad-hoc chart generation; the flagship, statistically-rigorous
# comparison lives in evaluation/comparator.py (via POST /compare/run).
# generate_report_charts.py
import matplotlib.pyplot as plt

from controllers import (
    DQNController,
    FixedTimerController,
    RandomController,
    RLController,
    RuleBasedController,
)
from metrics.calculator import PerformanceCalculator

print("Generating Report Charts...")
print("=" * 60)

# Run evaluations. DQN was previously omitted from this script entirely —
# per the audit, that meant the project's primary agent (config.py's
# primary_agent="DQN") was silently excluded from these charts. Included
# here for parity now that evaluate_run() is seeded and passes
# emergency_eta/distance (see metrics/calculator.py).
controllers = {
    "DQN": DQNController(),
    "Q-Learning": RLController(),
    "Fixed Timer": FixedTimerController(),
    "Rule-Based": RuleBasedController(),
    "Random": RandomController(),
}

results = {}
for name, ctrl in controllers.items():
    print(f"  Evaluating {name}...")
    results[name] = PerformanceCalculator.evaluate_run(ctrl, episodes=200, traffic_pattern="medium")

# Extract data
names = list(results.keys())
colors = ["#8b5cf6", "#3b82f6", "#f59e0b", "#22c55e", "#ef4444"]

avg_rewards = [results[n]["reward"]["mean"] for n in names]
avg_queues = [results[n]["queue"]["mean"] for n in names]
emergency_acc = [results[n]["emergency"]["accuracy"] for n in names]
throughputs = [results[n]["throughput"]["mean"] for n in names]
fairness = [results[n]["efficiency"]["fairness_index"] for n in names]
switches = [results[n]["efficiency"]["switches_per_episode"] for n in names]

# Create 2x3 chart grid
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("RescueWay — Controller Comparison Report", fontsize=16, fontweight="bold")

# Chart 1: Average Reward
ax = axes[0][0]
bars = ax.bar(names, avg_rewards, color=colors, edgecolor="white", linewidth=1.5)
ax.set_title("Average Reward per Episode", fontweight="bold")
ax.set_ylabel("Reward")
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
for bar, val in zip(bars, avg_rewards):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f"{val:.1f}",
        ha="center",
        fontsize=9,
    )
ax.grid(axis="y", alpha=0.3)

# Chart 2: Average Queue Length
ax = axes[0][1]
bars = ax.bar(names, avg_queues, color=colors, edgecolor="white", linewidth=1.5)
ax.set_title("Average Queue Length", fontweight="bold")
ax.set_ylabel("Vehicles in Queue")
for bar, val in zip(bars, avg_queues):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{val:.1f}",
        ha="center",
        fontsize=9,
    )
ax.grid(axis="y", alpha=0.3)

# Chart 3: Emergency Accuracy
ax = axes[0][2]
bars = ax.bar(names, emergency_acc, color=colors, edgecolor="white", linewidth=1.5)
ax.set_title("Emergency Response Accuracy", fontweight="bold")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 105)
for bar, val in zip(bars, emergency_acc):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{val:.1f}%",
        ha="center",
        fontsize=9,
    )
ax.grid(axis="y", alpha=0.3)

# Chart 4: Throughput
ax = axes[1][0]
bars = ax.bar(names, throughputs, color=colors, edgecolor="white", linewidth=1.5)
ax.set_title("Average Throughput", fontweight="bold")
ax.set_ylabel("Vehicles Cleared")
for bar, val in zip(bars, throughputs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{val:.0f}",
        ha="center",
        fontsize=9,
    )
ax.grid(axis="y", alpha=0.3)

# Chart 5: Fairness Index
ax = axes[1][1]
bars = ax.bar(names, fairness, color=colors, edgecolor="white", linewidth=1.5)
ax.set_title("Fairness Index (Jain's)", fontweight="bold")
ax.set_ylabel("Fairness (0-1)")
ax.set_ylim(0, 1.1)
for bar, val in zip(bars, fairness):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.3f}",
        ha="center",
        fontsize=9,
    )
ax.grid(axis="y", alpha=0.3)

# Chart 6: Signal Switches
ax = axes[1][2]
bars = ax.bar(names, switches, color=colors, edgecolor="white", linewidth=1.5)
ax.set_title("Signal Switches per Episode", fontweight="bold")
ax.set_ylabel("Switches")
for bar, val in zip(bars, switches):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{val:.1f}",
        ha="center",
        fontsize=9,
    )
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("comparison_report.png", dpi=150, bbox_inches="tight")
print("\nChart saved as comparison_report.png")

# Print summary table
print("\n" + "=" * 80)
print(
    f"{'Controller':<15} {'Reward':>10} {'Queue':>10} {'Emergency':>12} {'Throughput':>12} {'Fairness':>10}"
)
print("-" * 80)
for name in names:
    r = results[name]
    print(
        f"{name:<15} {r['reward']['mean']:>+10.2f} "
        f"{r['queue']['mean']:>10.2f} "
        f"{r['emergency']['accuracy']:>11.1f}% "
        f"{r['throughput']['mean']:>12.1f} "
        f"{r['efficiency']['fairness_index']:>10.3f}"
    )
print("=" * 80)

# Print RL improvements
rl_reward = results["Q-Learning"]["reward"]["mean"]
print("\nQ-Learning Improvements:")
for name in ["Fixed Timer", "Rule-Based", "Random"]:
    baseline = results[name]["reward"]["mean"]
    if abs(baseline) > 0:
        pct = ((rl_reward - baseline) / abs(baseline)) * 100
        print(f"  vs {name:<15}: {pct:+.1f}%")

dqn_reward = results["DQN"]["reward"]["mean"]
print("\nDQN Improvements:")
for name in ["Fixed Timer", "Rule-Based", "Random"]:
    baseline = results[name]["reward"]["mean"]
    if abs(baseline) > 0:
        pct = ((dqn_reward - baseline) / abs(baseline)) * 100
        print(f"  vs {name:<15}: {pct:+.1f}%")

plt.show()
