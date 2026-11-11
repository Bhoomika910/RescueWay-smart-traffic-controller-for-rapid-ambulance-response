# gps/zone_model.py
"""
GPS Zone Model for RescueWay.

In a real deployment, the ambulance transmits GPS coordinates every second.
The system calculates distance and ETA to the intersection.

In our prototype, the 4 zones represent distance bands from the intersection:

  Zone 4 (Far)     → 400-500m from intersection → ETA ~30s
  Zone 3 (Medium)  → 200-300m from intersection → ETA ~18s
  Zone 2 (Near)    → 80-150m from intersection  → ETA ~10s
  Zone 1 (Close)   → 0-80m from intersection    → ETA ~5s

The ambulance "moves" through zones: 4 → 3 → 2 → 1 → intersection
"""

# Zone configuration
ZONE_CONFIG = {
    1: {
        "name": "Close",
        "direction": "North",
        "distance_min": 0,  # meters
        "distance_max": 80,
        "avg_distance": 40,
        "eta_seconds": 5,  # estimated time to reach intersection
        "emergency_green": 8,  # how long to hold green
        "priority_level": 4,  # highest priority (closest)
        "description": "Ambulance is very close to intersection",
    },
    2: {
        "name": "Near",
        "direction": "East",
        "distance_min": 80,
        "distance_max": 200,
        "avg_distance": 140,
        "eta_seconds": 10,
        "emergency_green": 12,
        "priority_level": 3,
        "description": "Ambulance approaching intersection",
    },
    3: {
        "name": "Medium",
        "direction": "South",
        "distance_min": 200,
        "distance_max": 400,
        "avg_distance": 300,
        "eta_seconds": 18,
        "emergency_green": 18,
        "priority_level": 2,
        "description": "Ambulance detected at medium range",
    },
    4: {
        "name": "Far",
        "direction": "West",
        "distance_min": 400,
        "distance_max": 600,
        "avg_distance": 500,
        "eta_seconds": 30,
        "emergency_green": 25,
        "priority_level": 1,
        "description": "Ambulance detected far from intersection",
    },
}

# Single source of truth for "how many zones does this deployment have" —
# design-review finding (category 24 "Scalability"): the zone count was
# hardcoded as a bare `4` independently in rl/state_features.py's one-hot
# slice boundaries, rl/state_encoder.py's dimension constants, and
# elsewhere, rather than derived from this config table (the actual
# per-zone data) in one place. rl/state_features.py and
# rl/state_encoder.py now compute their dimensions from this constant, so
# adding/removing a ZONE_CONFIG entry changes STATE_DIM/N_STATES/N_ACTIONS
# automatically instead of requiring matching hand-edits in three files.
#
# Scope note: this makes the RL STATE/ACTION ENCODING zone-count-agnostic
# and tested as such (see tests/test_zone_parametrization.py) — it does
# NOT make the physical system (Arduino firmware's fixed 4-zone pin
# wiring, the SUMO network's fixed 4-approach topology, the frontend's
# fixed 4-panel layout) zone-count-agnostic. Generalizing those requires a
# new wiring diagram / SUMO topology file / UI layout respectively, not a
# code parametrization — a different, larger engineering program, not a
# refactor oversight. See ARCHITECTURE.md's scalability section for the
# full reasoning (the same honest-scoping principle applied there to the
# real-world-deployment-readiness assessment).
N_ZONES = len(ZONE_CONFIG)

# Average ambulance speed (m/s) — ~50 km/h in urban areas
AMBULANCE_SPEED = 14.0  # meters per second


class GPSZoneModel:
    """
    Models ambulance distance and ETA based on zone selection.

    In a real system, this would receive actual GPS coordinates.
    In our prototype, zone buttons simulate the GPS position.
    """

    def __init__(self):
        self.current_zone = None
        self.zone_history = []
        self.approach_detected = False
        self.approach_start_time = None

    def update_zone(self, zone: int) -> dict:
        """
        Called when ambulance GPS reports a new zone.
        Returns zone info including ETA and recommended green duration.
        """
        if zone not in ZONE_CONFIG:
            return {"error": f"Invalid zone: {zone}"}

        config = ZONE_CONFIG[zone]
        prev_zone = self.current_zone
        self.current_zone = zone
        self.zone_history.append(zone)

        # Detect approach pattern (zone number decreasing = getting closer)
        is_approaching = prev_zone is not None and zone < prev_zone

        # Calculate ETA
        eta = self.calculate_eta(zone)

        # Calculate adaptive green duration
        green_duration = self.calculate_green_duration(zone)

        return {
            "zone": zone,
            "zone_name": config["name"],
            "direction": config["direction"],
            "distance_m": config["avg_distance"],
            "eta_seconds": eta,
            "emergency_green_duration": green_duration,
            "priority_level": config["priority_level"],
            "is_approaching": is_approaching,
            "description": config["description"],
            "previous_zone": prev_zone,
        }

    def calculate_eta(self, zone: int) -> float:
        """Use the predefined ETA values for consistency."""
        config = ZONE_CONFIG[zone]
        return float(config["eta_seconds"])

    def calculate_green_duration(self, zone: int) -> int:
        """
        Calculate how long to hold green based on ambulance distance.

        Close zones → shorter green (ambulance passes quickly)
        Far zones → longer green (need to hold until ambulance arrives)
        """
        config = ZONE_CONFIG[zone]
        return config["emergency_green"]

    def get_preemption_time(self, zone: int) -> float:
        """
        How far in advance should we start clearing the intersection?

        Zone 4 (far): start clearing 25s before arrival
        Zone 1 (close): start clearing immediately
        """
        config = ZONE_CONFIG[zone]
        eta = config["eta_seconds"]
        # Start clearing when ambulance is ~5s away from intersection
        clearance_lead = max(0, eta - 5)
        return clearance_lead

    def get_status(self) -> dict:
        if self.current_zone is None:
            return {
                "active": False,
                "zone": None,
                "eta": None,
                "distance": None,
            }

        config = ZONE_CONFIG[self.current_zone]
        return {
            "active": True,
            "zone": self.current_zone,
            "zone_name": config["name"],
            "direction": config["direction"],
            "distance_m": config["avg_distance"],
            "eta_seconds": self.calculate_eta(self.current_zone),
            "green_duration": config["emergency_green"],
            "priority_level": config["priority_level"],
            "zone_history": list(self.zone_history[-10:]),
        }

    def clear(self):
        """Ambulance has passed. Reset GPS tracking."""
        self.current_zone = None
        self.approach_detected = False
        self.approach_start_time = None
