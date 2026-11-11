# metrics/collector.py
import time
from collections import deque

from config import METRICS_EMERGENCY_WINDOW, METRICS_WINDOW_SIZE, QUEUE_MAX_CAPACITY
from rl.emergency_scoring import is_emergency_action_correct


class MetricsCollector:
    """
    Collects real-time traffic metrics during simulation.
    Maintains rolling windows for trend analysis.
    """

    def __init__(
        self,
        window_size: int = METRICS_WINDOW_SIZE,
        emergency_window: int = METRICS_EMERGENCY_WINDOW,
    ):
        self.window_size = window_size
        self.emergency_window = emergency_window
        self.start_time = time.time()

        # Per-step metrics (rolling window)
        self.rewards = deque(maxlen=window_size)
        self.queue_lengths = deque(maxlen=window_size)
        self.wait_times = deque(maxlen=window_size)
        self.throughputs = deque(maxlen=window_size)
        self.congestion_levels = deque(maxlen=window_size)

        # Cumulative counters
        self.total_steps = 0
        self.total_reward = 0
        self.total_vehicles_arrived = 0
        self.total_vehicles_cleared = 0
        self.total_signal_switches = 0
        self.prev_signal = None

        # Emergency tracking
        # emergency_events counts distinct ambulance episodes; emergency_ticks
        # counts every recorded step while ANY emergency is active. accuracy
        # must divide emergency_correct (incremented at most once per tick)
        # by emergency_ticks, NOT emergency_events — dividing by the episode
        # count instead of the tick count is what previously produced
        # nonsensical accuracy values like 10250%.
        self.emergency_events = 0
        self.emergency_ticks = 0
        self.emergency_correct = 0
        self.emergency_start_time = None
        self._emergency_current_zone = None

        # Emergency response time tracking (larger window).
        # This is a fallback approximation ("how long the signal kept
        # serving a given ambulance's zone") used only when the caller
        # doesn't supply the authoritative per-ambulance stats computed by
        # EmergencyManager (true request-to-green latency, including any
        # queueing delay from ETA-based preemption) — see get_current_metrics().
        self.emergency_response_start = None
        self.emergency_response_times = deque(maxlen=emergency_window)
        self.emergency_total_delay = 0
        self.emergency_resolved_count = 0

        # Per-zone tracking
        self.zone_green_time = {1: 0, 2: 0, 3: 0, 4: 0}
        self.zone_total_wait = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self.zone_vehicles_served = {1: 0, 2: 0, 3: 0, 4: 0}

    def record_step(
        self,
        active_signal,
        queue_lengths,
        reward,
        system_metrics,
        emergency_active=False,
        ambulance_zone=0,
        action=None,
    ):
        """Record one simulation step."""
        self.total_steps += 1

        # Reward
        self.rewards.append(reward)
        self.total_reward += reward

        # Queue lengths
        total_queue = sum(queue_lengths.values())
        self.queue_lengths.append(total_queue)

        # Congestion
        avg_congestion = total_queue / (4 * QUEUE_MAX_CAPACITY)
        self.congestion_levels.append(avg_congestion)

        # Throughput
        cleared = system_metrics.get("total_vehicles_cleared", 0)
        self.throughputs.append(cleared)
        self.total_vehicles_cleared = cleared
        self.total_vehicles_arrived = system_metrics.get("total_vehicles_arrived", 0)

        # Wait time
        avg_wait = system_metrics.get("avg_wait_time_all", 0)
        self.wait_times.append(avg_wait)

        # Signal switches
        if self.prev_signal is not None and active_signal != self.prev_signal:
            self.total_signal_switches += 1
        self.prev_signal = active_signal

        # Zone green time
        self.zone_green_time[active_signal] = self.zone_green_time.get(active_signal, 0) + 1

        # Emergency tracking
        if emergency_active and ambulance_zone > 0:
            is_new_ambulance = (
                self.emergency_start_time is None or ambulance_zone != self._emergency_current_zone
            )
            if is_new_ambulance:
                # A fresh ambulance episode starts either when emergency mode
                # begins, or when an ETA-based preemption/handoff switches
                # the served zone to a different ambulance mid-emergency.
                # Close out the previous ambulance's response window (if any)
                # first, so one ambulance's wait time is never folded into
                # the next ambulance's response-time sample.
                if (
                    self.emergency_start_time is not None
                    and self.emergency_response_start is not None
                ):
                    response_time = time.time() - self.emergency_response_start
                    self.emergency_response_times.append(response_time)
                    self.emergency_total_delay += response_time
                    self.emergency_resolved_count += 1

                self.emergency_start_time = time.time()
                self.emergency_response_start = time.time()
                self.emergency_events += 1
                self._emergency_current_zone = ambulance_zone

            # Count every tick spent actively serving an emergency. This is
            # the correct denominator for "accuracy": emergency_correct is
            # also incremented at most once per tick, so both share the
            # same unit (ticks), unlike emergency_events (episodes).
            self.emergency_ticks += 1

            if action is not None:
                if is_emergency_action_correct(action, ambulance_zone, active_signal):
                    self.emergency_correct += 1
        else:
            if self.emergency_start_time is not None and self.emergency_response_start is not None:
                response_time = time.time() - self.emergency_response_start
                self.emergency_response_times.append(response_time)
                self.emergency_total_delay += response_time
                self.emergency_resolved_count += 1

            self.emergency_start_time = None
            self.emergency_response_start = None
            self._emergency_current_zone = None

    def get_current_metrics(self, emergency_stats: dict = None) -> dict:
        """
        Get current snapshot of all metrics.

        emergency_stats: optional dict from EmergencyManager.get_stats().
        When provided, its avg/min/max_response_time values (true
        request-to-green latency per ambulance, including any queueing
        delay from ETA-based preemption) are used instead of this
        collector's own zone-segment-based approximation, which cannot
        see individual ambulance request timestamps.
        """
        if emergency_stats:
            avg_response_time = emergency_stats.get("avg_response_time", 0)
            min_response_time = emergency_stats.get("min_response_time", 0)
            max_response_time = emergency_stats.get("max_response_time", 0)
        else:
            avg_response_time = self._safe_mean(self.emergency_response_times)
            min_response_time = (
                round(min(self.emergency_response_times), 2) if self.emergency_response_times else 0
            )
            max_response_time = (
                round(max(self.emergency_response_times), 2) if self.emergency_response_times else 0
            )

        return {
            "step": self.total_steps,
            "uptime_seconds": round(time.time() - self.start_time, 1),
            # Reward metrics
            "avg_reward": self._safe_mean(self.rewards),
            "total_reward": round(self.total_reward, 2),
            # Queue metrics
            "avg_queue_length": self._safe_mean(self.queue_lengths),
            "current_queue_length": self.queue_lengths[-1] if self.queue_lengths else 0,
            "max_queue_length": max(self.queue_lengths) if self.queue_lengths else 0,
            # Wait time metrics
            "avg_wait_time": self._safe_mean(self.wait_times),
            "current_wait_time": self.wait_times[-1] if self.wait_times else 0,
            # Throughput metrics
            "total_vehicles_arrived": self.total_vehicles_arrived,
            "total_vehicles_cleared": self.total_vehicles_cleared,
            "clearance_rate": round(
                (
                    (self.total_vehicles_cleared / self.total_vehicles_arrived * 100)
                    if self.total_vehicles_arrived > 0
                    else 0
                ),
                1,
            ),
            # Congestion metrics
            "avg_congestion": self._safe_mean(self.congestion_levels),
            "current_congestion": self.congestion_levels[-1] if self.congestion_levels else 0,
            # Signal metrics
            "total_switches": self.total_signal_switches,
            "switches_per_step": round(
                self.total_signal_switches / self.total_steps if self.total_steps > 0 else 0, 3
            ),
            # Emergency metrics
            "emergency_events": self.emergency_events,
            "emergency_accuracy": round(
                (
                    (self.emergency_correct / self.emergency_ticks * 100)
                    if self.emergency_ticks > 0
                    else 0
                ),
                1,
            ),
            "avg_emergency_response_time": avg_response_time,
            "min_emergency_response_time": min_response_time,
            "max_emergency_response_time": max_response_time,
            "emergency_resolved": self.emergency_resolved_count,
            "total_emergency_delay": round(self.emergency_total_delay, 2),
            # Fairness metrics
            "zone_green_distribution": dict(self.zone_green_time),
            "fairness_index": self._compute_fairness(),
        }

    def get_trends(self, window: int = 20) -> dict:
        """Get recent trends for charts."""
        return {
            "rewards": list(self.rewards)[-window:],
            "queue_lengths": list(self.queue_lengths)[-window:],
            "wait_times": list(self.wait_times)[-window:],
            "congestion": list(self.congestion_levels)[-window:],
        }

    def _safe_mean(self, data):
        if not data:
            return 0
        return round(float(sum(data) / len(data)), 2)

    def _compute_fairness(self) -> float:
        """
        Jain's fairness index.
        1.0 = perfectly fair, 0.25 = completely unfair.
        """
        values = list(self.zone_green_time.values())
        if not values or sum(values) == 0:
            return 1.0
        n = len(values)
        numerator = sum(values) ** 2
        denominator = n * sum(v**2 for v in values)
        if denominator == 0:
            return 1.0
        return round(numerator / denominator, 3)

    def reset(self):
        self.__init__(self.window_size, self.emergency_window)
