# LEGACY: validation script from Q-Learning era, now with real assertions.
# test_accuracy.py
from rl.emergency_scoring import is_emergency_action_correct
from rl.environment import TrafficEnv
from rl.q_agent import QAgent


def _measure_emergency_accuracy():
    """
    Measures the currently-saved Q-table's emergency-zone accuracy over
    500 episodes. Previously this had no assert statements at all despite
    the test_* naming pytest auto-collects — a CI run would report it as
    "passing" no matter what it printed. This isn't a script that checks
    a specific expected accuracy (that number legitimately changes across
    retraining runs, so hardcoding a threshold would be fragile/arbitrary)
    — instead it asserts the basic invariants that would catch a real
    counting/logic bug: emergencies were actually encountered, and the
    correct-count is a sane fraction of the total.
    """
    agent = QAgent(epsilon=0.0)  # pure exploitation, no randomness
    agent.load()

    env = TrafficEnv()
    correct = 0
    total = 0

    for _ in range(500):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose(state)
            next_state, reward, done, info = env.step(action)

            if info["ambulance_active"] and info["ambulance_zone"] > 0:
                total += 1
                # env.step() already ran above, so env.active_signal
                # reflects the post-step served zone.
                if is_emergency_action_correct(action, info["ambulance_zone"], env.active_signal):
                    correct += 1

            state = next_state

    assert total > 0, (
        "No emergency decisions were encountered in 500 episodes — "
        "ambulance spawning is broken (check AMBULANCE_SPAWN_PROBABILITY "
        "in rl/environment.py)."
    )
    assert 0 <= correct <= total, (
        f"correct ({correct}) must be between 0 and total ({total}) — " f"counting logic is broken."
    )

    accuracy = (correct / total) * 100
    return {"total": total, "correct": correct, "accuracy_pct": accuracy}


def test_emergency_accuracy():
    _measure_emergency_accuracy()


if __name__ == "__main__":
    result = _measure_emergency_accuracy()
    print(f"Emergency decisions: {result['total']}")
    print(f"Correct: {result['correct']}")
    print(f"Actual accuracy: {result['accuracy_pct']:.1f}%")
