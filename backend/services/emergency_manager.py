# services/emergency_manager.py
"""
Emergency Manager for RescueWay.

Handles multiple simultaneous ambulance requests using
ETA-based priority queuing with preemption.

Priority Rules:
1. Closest ambulance (shortest ETA) gets immediate priority
2. New closer ambulance preempts current active emergency
3. Same-zone ambulances extend green duration
4. After active clears, next in queue activates immediately
5. RL agent sees only the highest-priority ambulance
"""
import time
from dataclasses import dataclass
from typing import Optional

from config import DEFAULT_EMERGENCY_DURATION
from gps.zone_model import ZONE_CONFIG, GPSZoneModel


@dataclass
class EmergencyRequest:
    """Single ambulance emergency request."""

    id: str
    zone: int
    traffic: str
    distance_m: float
    eta_seconds: float
    green_duration: int
    priority_level: int
    timestamp: float
    status: str = "waiting"  # waiting, active, completed, preempted
    response_start: float = 0.0
    response_end: float = 0.0

    def get_age(self) -> float:
        """Seconds since this request was created."""
        return round(time.time() - self.timestamp, 1)

    def get_response_time(self) -> float:
        """Seconds from request to green signal."""
        if self.response_start > 0:
            return round(self.response_start - self.timestamp, 2)
        return 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "zone": self.zone,
            "traffic": self.traffic,
            "distance_m": self.distance_m,
            "eta_seconds": self.eta_seconds,
            "green_duration": self.green_duration,
            "priority_level": self.priority_level,
            "status": self.status,
            "age_seconds": self.get_age(),
            "response_time": self.get_response_time(),
        }


class EmergencyManager:
    """
    Manages multiple ambulance emergencies with priority queuing.

    Architecture:
    - Maintains a priority queue sorted by ETA (closest first)
    - Supports preemption (closer ambulance takes over)
    - Handles same-zone merging (extend green)
    - Tracks response times for all emergencies
    - Provides queue status for dashboard display
    """

    def __init__(self, default_green_duration: int = DEFAULT_EMERGENCY_DURATION):
        self.queue: list[EmergencyRequest] = []
        self.active: Optional[EmergencyRequest] = None
        self.completed: list[EmergencyRequest] = []
        self.gps_model = GPSZoneModel()
        self._next_id = 1
        # Fallback green duration for get_active_green_duration() when
        # there's no active request (defensive default — in normal
        # operation get_active_green_duration() is only called right
        # after add_emergency() sets self.active). Previously hardcoded
        # to a bare `15` here, independently of config.DEFAULT_EMERGENCY_
        # DURATION and of TrafficController.emergency_duration (settable
        # via POST /simulation/timings, which had no effect on anything —
        # see the emergency_duration property in traffic_controller.py).
        self.default_green_duration = default_green_duration

        # Statistics
        self.total_requests = 0
        self.total_completed = 0
        self.total_preempted = 0
        self.total_merged = 0

    def add_emergency(self, zone: int, traffic: str) -> dict:
        """
        Add a new ambulance emergency request.

        Returns decision about what to do:
        - "activate": no current emergency, activate immediately
        - "preempt": new ambulance is closer, preempt current
        - "queue": current ambulance is closer, queue new one
        - "merge": same zone as active, extend green duration
        """
        if zone not in ZONE_CONFIG:
            raise ValueError(f"Unknown zone {zone!r} — must be one of {sorted(ZONE_CONFIG)}")
        self.gps_model.update_zone(
            zone
        )  # updates internal GPS-model tracking state; return value unused here
        config = ZONE_CONFIG[zone]

        request = EmergencyRequest(
            id=f"AMB-{self._next_id:03d}",
            zone=zone,
            traffic=traffic,
            distance_m=config["avg_distance"],
            eta_seconds=config["eta_seconds"],
            green_duration=config["emergency_green"],
            priority_level=config["priority_level"],
            timestamp=time.time(),
        )

        self._next_id += 1
        self.total_requests += 1

        # Decision logic
        if self.active is None:
            # No active emergency — activate immediately
            request.status = "active"
            request.response_start = time.time()
            self.active = request
            return {
                "decision": "activate",
                "request": request,
                "message": f"{request.id}: Ambulance in Zone {zone} — activating immediately",
            }

        if self.active.zone == zone:
            # Same zone — merge (extend green duration)
            self.active.green_duration += config["emergency_green"] // 2
            self.total_merged += 1
            return {
                "decision": "merge",
                "request": request,
                "active": self.active,
                "message": f"{request.id}: Same zone as active — extending green to {self.active.green_duration}s",
            }

        # Compare ETA: new ambulance vs active
        if request.eta_seconds < self.active.eta_seconds:
            # New ambulance is closer — preempt current
            preempted = self.active
            preempted.status = "preempted"
            self.total_preempted += 1

            # Put preempted back in queue
            self.queue.append(preempted)

            # Activate new request
            request.status = "active"
            request.response_start = time.time()
            self.active = request

            # Re-sort queue by ETA
            self.queue.sort(key=lambda r: r.eta_seconds)

            return {
                "decision": "preempt",
                "request": request,
                "preempted": preempted,
                "message": (
                    f"{request.id}: Zone {zone} (ETA {request.eta_seconds}s) "
                    f"PREEMPTS {preempted.id}: Zone {preempted.zone} (ETA {preempted.eta_seconds}s)"
                ),
            }

        # Current ambulance is closer — queue the new one
        request.status = "waiting"
        self.queue.append(request)
        self.queue.sort(key=lambda r: r.eta_seconds)

        return {
            "decision": "queue",
            "request": request,
            "active": self.active,
            "queue_position": len(self.queue),
            "message": (
                f"{request.id}: Zone {zone} (ETA {request.eta_seconds}s) "
                f"QUEUED — Zone {self.active.zone} is closer"
            ),
        }

    def complete_active(self) -> Optional[EmergencyRequest]:
        """
        Mark current emergency as completed.
        Returns the next emergency to handle (or None).
        """
        if self.active:
            self.active.status = "completed"
            self.active.response_end = time.time()
            self.completed.append(self.active)
            self.total_completed += 1
            self.active = None

        # Check queue for next emergency
        if self.queue:
            next_request = self.queue.pop(0)
            next_request.status = "active"
            next_request.response_start = time.time()
            self.active = next_request
            return next_request

        # Clear GPS model
        self.gps_model.clear()
        return None

    def get_active_zone(self) -> int:
        """Get the zone of the active emergency (0 if none)."""
        if self.active:
            return self.active.zone
        return 0

    def get_active_green_duration(self) -> int:
        """Get green duration for active emergency."""
        if self.active:
            return self.active.green_duration
        return self.default_green_duration

    def get_active_eta(self) -> float:
        """Get ETA for active emergency."""
        if self.active:
            return self.active.eta_seconds
        return 0

    def get_active_distance(self) -> float:
        """Get distance for active emergency."""
        if self.active:
            return self.active.distance_m
        return 0

    def is_emergency_active(self) -> bool:
        """Check if any emergency is active."""
        return self.active is not None

    def has_queued(self) -> bool:
        """Check if there are queued emergencies."""
        return len(self.queue) > 0

    def get_queue_size(self) -> int:
        return len(self.queue)

    def get_status(self) -> dict:
        """Full status for dashboard display. All values JSON-serializable."""
        active_dict = None
        if self.active:
            active_dict = {
                "id": str(self.active.id),
                "zone": int(self.active.zone),
                "traffic": str(self.active.traffic),
                "distance_m": float(self.active.distance_m),
                "eta_seconds": float(self.active.eta_seconds),
                "green_duration": int(self.active.green_duration),
                "priority_level": int(self.active.priority_level),
                "status": str(self.active.status),
                "age_seconds": float(self.active.get_age()),
                "response_time": float(self.active.get_response_time()),
            }

        queue_list = []
        for r in self.queue:
            queue_list.append(
                {
                    "id": str(r.id),
                    "zone": int(r.zone),
                    "distance_m": float(r.distance_m),
                    "eta_seconds": float(r.eta_seconds),
                    "green_duration": int(r.green_duration),
                    "status": str(r.status),
                }
            )

        completed_list = []
        for r in self.completed[-5:]:
            completed_list.append(
                {
                    "id": str(r.id),
                    "zone": int(r.zone),
                    "status": str(r.status),
                    "response_time": float(r.get_response_time()),
                }
            )

        return {
            "active": active_dict,
            "queue": queue_list,
            "queue_size": int(len(self.queue)),
            "completed_count": int(self.total_completed),
            "total_requests": int(self.total_requests),
            "total_preempted": int(self.total_preempted),
            "total_merged": int(self.total_merged),
            "recent_completed": completed_list,
        }

    def get_stats(self) -> dict:
        """Statistics for evaluation. All values JSON-serializable."""
        response_times = []
        for r in self.completed:
            rt = r.get_response_time()
            if rt > 0:
                response_times.append(rt)

        return {
            "total_requests": int(self.total_requests),
            "total_completed": int(self.total_completed),
            "total_preempted": int(self.total_preempted),
            "total_merged": int(self.total_merged),
            "avg_response_time": (
                round(sum(response_times) / len(response_times), 2) if response_times else 0.0
            ),
            "min_response_time": round(min(response_times), 2) if response_times else 0.0,
            "max_response_time": round(max(response_times), 2) if response_times else 0.0,
            "queue_size": int(len(self.queue)),
            "active": self.active is not None,
        }

    def clear_all(self):
        """Reset everything."""
        self.queue.clear()
        self.active = None
        self.completed.clear()
        self.gps_model.clear()
        self.total_requests = 0
        self.total_completed = 0
        self.total_preempted = 0
        self.total_merged = 0
