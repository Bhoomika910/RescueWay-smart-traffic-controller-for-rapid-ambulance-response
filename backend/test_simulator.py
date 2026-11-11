# LEGACY: Manual validation script from Q-Learning era, now with real
# assertions — previously had none at all (just print statements for
# eyeball inspection), the same gap fixed in test_reward.py/test_accuracy.py.
# backend/test_simulator.py
from config import QUEUE_MAX_CAPACITY
from traffic.simulator import TrafficSimulator


def _run_simulator(verbose=False):
    sim = TrafficSimulator()
    sim.set_traffic_pattern("high")

    if verbose:
        print("Testing Traffic Simulator...")
        print("Pattern: HIGH\n")

    prev_cleared = 0
    snapshots = []
    for step in range(100):
        metrics = sim.step(active_signal=1 if step % 20 < 10 else 2, is_green=(step % 20) < 8)
        system = sim.get_system_metrics()

        # Basic physical invariants a broken queueing model would violate:
        # queues can't go negative or silently exceed configured capacity,
        # and vehicles-cleared is a cumulative counter that must never
        # decrease.
        for zone_metrics in metrics.values():
            assert 0 <= zone_metrics.queue_length <= QUEUE_MAX_CAPACITY, (
                f"step {step}: queue_length {zone_metrics.queue_length} outside "
                f"[0, {QUEUE_MAX_CAPACITY}]"
            )
        assert system["total_vehicles_cleared"] >= prev_cleared, (
            f"step {step}: total_vehicles_cleared decreased "
            f"({prev_cleared} -> {system['total_vehicles_cleared']})"
        )
        assert system["avg_wait_time_all"] >= 0, "avg_wait_time_all must be non-negative"
        prev_cleared = system["total_vehicles_cleared"]

        if verbose and step % 20 == 0:
            snapshots.append((step, dict(system), metrics[1].queue_length, metrics[2].queue_length))

    return snapshots


def test_simulator():
    _run_simulator()


if __name__ == "__main__":
    for step, system, z1, z2 in _run_simulator(verbose=True):
        print(f"Step {step}:")
        print(f"  Total queue: {system['total_queue_length']}")
        print(f"  Vehicles cleared: {system['total_vehicles_cleared']}")
        print(f"  Avg wait time: {system['avg_wait_time_all']:.2f}s")
        print(f"  Zone 1 queue: {z1}")
        print(f"  Zone 2 queue: {z2}")
        print()
    print("All assertions passed.")
