# visualize_qtable.py
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from rl.state_encoder import (
    ACTIVE_SIGNALS,
    AMBULANCE_ZONES,
    CLEARANCE_RISK_LEVELS,
    CONGESTED_ZONES,
    CONGESTION_LEVELS,
    EMERGENCY_STATUS,
    N_STATES,
    QUEUE_IMBALANCE,
)

QTABLE_PATH = "rl/q_table_v2.pkl"
with open(QTABLE_PATH, "rb") as f:
    data = pickle.load(f)
    if isinstance(data, dict) and "q_table" in data:
        q = data["q_table"]
    else:
        q = data

if q.shape[0] != N_STATES:
    raise ValueError(
        f"Loaded Q-table has {q.shape[0]} states but the current "
        f"StateEncoder produces N_STATES={N_STATES}. This checkpoint was "
        f"trained against a different (older) state encoding and cannot be "
        f"visualized against the current one — retrain via train_offline.py."
    )


def encode_state(
    f_emergency,
    f_ambulance,
    f_signal,
    f_congestion,
    f_most_congested,
    f_imbalance,
    f_clearance_risk,
):
    """Mirrors rl/state_encoder.py's StateEncoder.encode() mixed-radix
    arithmetic exactly, so this script can never silently drift out of
    sync with the real encoder again (it previously used a stale,
    pre-redesign formula sized for a much smaller state space, which
    meant every index it computed actually pointed at emergency_active=0,
    ambulance_zone=0 states — the "Ambulance in Zone N" heatmaps were
    plotting the wrong Q-values under the wrong labels)."""
    state = 0
    state = state * EMERGENCY_STATUS + f_emergency
    state = state * AMBULANCE_ZONES + f_ambulance
    state = state * ACTIVE_SIGNALS + f_signal
    state = state * CONGESTION_LEVELS + f_congestion
    state = state * CONGESTED_ZONES + f_most_congested
    state = state * QUEUE_IMBALANCE + f_imbalance
    state = state * CLEARANCE_RISK_LEVELS + f_clearance_risk
    return state


ACTION_NAMES = ["Green Z1", "Green Z2", "Green Z3", "Green Z4", "Extend", "Normal"]
# f_congestion's 3 buckets (see StateEncoder._compute_congestion: total
# queue < 15 / 15-30 / > 30) — the closest current analog to the old
# script's "traffic level" axis.
CONGESTION_NAMES = ["Low congestion", "Medium congestion", "High congestion"]
ZONE_NAMES = ["Zone 1", "Zone 2", "Zone 3", "Zone 4"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "RescueWay Q-Table Heatmap\n(Learned action values per state, " "emergency active)", fontsize=14
)

for z_idx, zone_name in enumerate(ZONE_NAMES):
    ax = axes[z_idx // 2][z_idx % 2]
    f_ambulance = z_idx + 1  # ambulance_zone: 0=none, 1-4=zones

    heat = np.zeros((CONGESTION_LEVELS, 6))
    for f_congestion in range(CONGESTION_LEVELS):
        accum = np.zeros(6)
        count = 0
        # Average over every other factor (active signal, most-congested
        # zone, queue imbalance, clearance risk) to collapse them into one
        # row per congestion level — the same "average across dimensions
        # we're not plotting" intent the original script had, extended to
        # the factors that didn't exist in the pre-redesign encoding.
        for f_signal in range(ACTIVE_SIGNALS):
            for f_most_congested in range(CONGESTED_ZONES):
                for f_imbalance in range(QUEUE_IMBALANCE):
                    for f_clearance_risk in range(CLEARANCE_RISK_LEVELS):
                        state = encode_state(
                            1,
                            f_ambulance,
                            f_signal,
                            f_congestion,
                            f_most_congested,
                            f_imbalance,
                            f_clearance_risk,
                        )
                        accum += q[state]
                        count += 1
        heat[f_congestion] = accum / count

    im = ax.imshow(heat, cmap="RdYlGn", aspect="auto")
    ax.set_title(f"Ambulance in {zone_name}", fontweight="bold")
    ax.set_xticks(range(6))
    ax.set_xticklabels(ACTION_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(CONGESTION_LEVELS))
    ax.set_yticklabels(CONGESTION_NAMES)
    ax.set_xlabel("Action")
    ax.set_ylabel("Congestion level")

    for i in range(CONGESTION_LEVELS):
        for j in range(6):
            ax.text(
                j,
                i,
                f"{heat[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if abs(heat[i, j]) < 50 else "white",
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("qtable_heatmap.png", dpi=150, bbox_inches="tight")
print("Q-table heatmap saved as qtable_heatmap.png")
plt.show()
