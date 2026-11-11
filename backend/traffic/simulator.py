# backend/traffic/simulator.py
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from config import QUEUE_MAX_CAPACITY, VEHICLE_CLEARANCE_RATE


@dataclass
class TrafficMetrics:
    """Real-time traffic metrics for a single zone"""

    zone_id: int
    queue_length: int
    waiting_vehicles: int
    total_wait_time: float
    avg_wait_time: float
    vehicles_arrived: int
    vehicles_cleared: int
    congestion_level: float  # 0.0 to 1.0
    throughput: int
    is_green: bool
    green_duration: float
    time_since_last_green: float


class VehicleQueue:
    """Queue of vehicles waiting at a zone"""

    def __init__(self, zone_id: int, max_capacity: int = QUEUE_MAX_CAPACITY):
        self.zone_id = zone_id
        self.queue = deque()
        self.max_capacity = max_capacity
        self.total_vehicles_served = 0
        self.total_wait_time = 0
        self.arrival_count = 0
        self.timestamp_created = time.time()

    def add_vehicle(self, arrival_time: float) -> bool:
        """Add vehicle to queue. Returns True if added, False if full."""
        if len(self.queue) < self.max_capacity:
            self.queue.append(arrival_time)
            self.arrival_count += 1
            return True
        return False  # Queue overflow

    def remove_vehicle(self, departure_time: float) -> float:
        """Remove vehicle from front of queue. Returns wait time."""
        if len(self.queue) > 0:
            arrival_time = self.queue.popleft()
            wait_time = departure_time - arrival_time
            self.total_wait_time += wait_time
            self.total_vehicles_served += 1
            return wait_time
        return 0

    def get_length(self) -> int:
        return len(self.queue)

    def get_avg_wait_time(self) -> float:
        if self.total_vehicles_served == 0:
            return 0
        return self.total_wait_time / self.total_vehicles_served

    def get_congestion(self) -> float:
        """Congestion level from 0.0 (empty) to 1.0 (full)"""
        return len(self.queue) / self.max_capacity


class TrafficSimulator:
    """
    Realistic traffic simulator for a single 4-way intersection.

    Each zone (1-4) has a queue of vehicles.
    Vehicles arrive randomly (Poisson distribution).
    Vehicles depart when signal is green.
    """

    def __init__(self):
        self.zones = {
            1: VehicleQueue(zone_id=1),
            2: VehicleQueue(zone_id=2),
            3: VehicleQueue(zone_id=3),
            4: VehicleQueue(zone_id=4),
        }
        self.current_time = 0.0
        self.step_count = 0
        self.active_signal = 1
        self.is_green = False
        self.green_start_time = 0.0
        self.yellow_start_time = 0.0
        self.yellow_duration = 0
        self.traffic_pattern = "medium"  # low, medium, high, rush_hour
        self.arrival_rates = self._get_arrival_rates()

        # Statistics
        self.total_vehicles_arrived = 0
        self.total_vehicles_cleared = 0
        self.total_wait_time = 0
        self.emergency_waits = []

    def _get_arrival_rates(self) -> dict:
        """Vehicle arrival rates (vehicles per second) by traffic pattern"""
        rates = {
            "low": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},
            "medium": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0},
            "high": {"1": 1.5, "2": 1.5, "3": 1.5, "4": 1.5},
            "rush_hour": {"1": 2.0, "2": 1.8, "3": 1.6, "4": 1.4},
        }
        return rates.get(self.traffic_pattern, rates["medium"])

    def set_traffic_pattern(self, pattern: str):
        """Set traffic pattern: low, medium, high, rush_hour"""
        if pattern in ["low", "medium", "high", "rush_hour"]:
            self.traffic_pattern = pattern
            self.arrival_rates = self._get_arrival_rates()

    def step(
        self,
        active_signal: int,
        is_green: bool,
        yellow_duration: float = 2.0,
        current_time: float = None,
    ):
        """
        Step the simulator forward by one time unit.

        Args:
            active_signal: which zone (1-4) has priority
            is_green: is the signal currently green?
            yellow_duration: duration of yellow light (seconds)
            current_time: optional override of internal time
        """
        if current_time is not None:
            self.current_time = current_time
        else:
            self.current_time += 1.0

        self.step_count += 1
        self.active_signal = active_signal
        self.is_green = is_green
        self.yellow_duration = yellow_duration

        # Step 1: Vehicle arrivals (Poisson process per zone)
        self._generate_arrivals()

        # Step 2: Vehicle departures (if green)
        self._process_departures(active_signal, is_green)

        # Return current metrics
        return self.get_metrics()

    def _generate_arrivals(self):
        """Generate random vehicle arrivals using Poisson distribution"""
        for zone_id in [1, 2, 3, 4]:
            # Lambda = arrival rate (vehicles per second)
            lambda_rate = float(self.arrival_rates[str(zone_id)])

            # Poisson random variable
            arrivals = np.random.poisson(lambda_rate)

            for _ in range(arrivals):
                success = self.zones[zone_id].add_vehicle(self.current_time)
                if success:
                    self.total_vehicles_arrived += 1
                # If queue is full, vehicle is lost (overflow)

    def _process_departures(self, active_signal: int, is_green: bool):
        """Remove vehicles from queue if signal is green"""
        if not is_green:
            return

        # Active signal zone can discharge vehicles per second.
        # VEHICLE_CLEARANCE_RATE is the single source of truth for this
        # number — rl/ambulance_delay.py's queue-clearance-time estimate
        # and the benchmark's ambulance-delay metrics both depend on it
        # matching what actually happens here.
        for _ in range(VEHICLE_CLEARANCE_RATE):
            wait_time = self.zones[active_signal].remove_vehicle(self.current_time)
            if wait_time > 0:
                self.total_vehicles_cleared += 1
                self.total_wait_time += wait_time

    def get_metrics(self) -> dict:
        """Return current metrics for all zones"""
        metrics = {}
        for zone_id in [1, 2, 3, 4]:
            queue = self.zones[zone_id]

            metrics[zone_id] = TrafficMetrics(
                zone_id=zone_id,
                queue_length=queue.get_length(),
                waiting_vehicles=queue.get_length(),
                total_wait_time=queue.total_wait_time,
                avg_wait_time=queue.get_avg_wait_time(),
                vehicles_arrived=queue.arrival_count,
                vehicles_cleared=queue.total_vehicles_served,
                congestion_level=queue.get_congestion(),
                throughput=queue.total_vehicles_served,
                is_green=(zone_id == self.active_signal and self.is_green),
                green_duration=0,  # will be set by traffic controller
                time_since_last_green=0,  # will be set by traffic controller
            )

        return metrics

    def get_system_metrics(self) -> dict:
        """Return aggregate system metrics"""
        total_queue_length = sum(z.get_length() for z in self.zones.values())
        avg_congestion = np.mean([z.get_congestion() for z in self.zones.values()])

        return {
            "current_time": self.current_time,
            "step": self.step_count,
            "total_queue_length": total_queue_length,
            "avg_congestion": avg_congestion,
            "total_vehicles_arrived": self.total_vehicles_arrived,
            "total_vehicles_cleared": self.total_vehicles_cleared,
            "avg_wait_time_all": (
                self.total_wait_time / self.total_vehicles_cleared
                if self.total_vehicles_cleared > 0
                else 0
            ),
            "traffic_pattern": self.traffic_pattern,
        }

    def reset(self):
        """Reset simulator for new episode"""
        self.__init__()
