# LEGACY: validation script from Q-Learning era, now with real assertions.
# test_reward.py
from rl.reward import get_reward_breakdown


def _compute_breakdowns():
    """Shared by the pytest assertions below and the __main__ print block —
    pytest warns if a test function returns a non-None value (it isn't
    meant to double as a data-producing helper), so the computation lives
    here and test_reward_breakdown() below calls it without returning it."""
    # Test 1 vs 2: same queues/traffic, only the action differs — serving
    # the ambulance's actual zone (1) must score higher on the emergency
    # component (and overall) than serving the wrong zone (3).
    b1 = get_reward_breakdown(
        action=0, amb_zone=1, prev_signal=3, traffic="high", queue_lengths={1: 20, 2: 5, 3: 3, 4: 8}
    )
    b2 = get_reward_breakdown(
        action=2, amb_zone=1, prev_signal=3, traffic="high", queue_lengths={1: 20, 2: 5, 3: 3, 4: 8}
    )
    assert b1["emergency"] > b2["emergency"], (
        f"Correct-zone emergency reward ({b1['emergency']}) should exceed "
        f"wrong-zone emergency reward ({b2['emergency']})"
    )
    assert b1["total"] > b2["total"], (
        f"Correct-zone total reward ({b1['total']}) should exceed "
        f"wrong-zone total reward ({b2['total']})"
    )

    # Test 3 vs 4: no ambulance — serving the most-congested zone (2, queue
    # 25) must score higher on the congestion component (and overall) than
    # serving an empty zone (4, queue 0).
    b3 = get_reward_breakdown(
        action=1, amb_zone=0, prev_signal=1, traffic="high", queue_lengths={1: 5, 2: 25, 3: 8, 4: 3}
    )
    b4 = get_reward_breakdown(
        action=3,
        amb_zone=0,
        prev_signal=1,
        traffic="high",
        queue_lengths={1: 15, 2: 20, 3: 12, 4: 0},
    )
    assert b3["congestion"] > b4["congestion"], (
        f"Serving the most-congested zone ({b3['congestion']}) should score "
        f"higher on congestion than serving an empty zone ({b4['congestion']})"
    )
    assert b3["total"] > b4["total"], (
        f"Serving the most-congested zone ({b3['total']}) should score "
        f"higher overall than serving an empty zone ({b4['total']})"
    )

    return {"correct_zone": b1, "wrong_zone": b2, "serve_congested": b3, "serve_empty": b4}


def test_reward_breakdown():
    _compute_breakdowns()


if __name__ == "__main__":
    print("Reward Breakdown Tests")
    print("=" * 60)
    results = _compute_breakdowns()
    for label, breakdown in results.items():
        print(f"\n{label}:")
        for k, v in breakdown.items():
            print(f"  {k:12s}: {v:+.2f}")
    print("\nAll assertions passed.")
