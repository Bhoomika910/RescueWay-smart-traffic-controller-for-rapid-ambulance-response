# sumo_adapter.py
"""
RescueWay SUMO Adapter

Maps between SUMO's lane-level traffic data and RescueWay's zone-level
state representation.

Link mapping, verified at run time against
traci.trafficlight.getControlledLinks("center") on the network actually
shipped in this repository (intersection.net.xml):

  Link 0-2:  north_in (Zone 1) - left, through, right
  Link 3-5:  east_in  (Zone 2) - left, through, right
  Link 6-8:  south_in (Zone 3) - left, through, right
  Link 9-11: west_in  (Zone 4) - left, through, right

Phase string is 12 characters, one per link: G=green, r=red, y=yellow

CORRECTNESS NOTE (this is the fix for a benchmark-invalidating bug):
this module previously documented and implemented the first two link
groups in the opposite order — east_in on links 0-2 and north_in on links
3-5 — which is NOT what netconvert produced for this network. The
consequence was that every command to serve Zone 1 actually turned Zone
2's approach green and vice versa, silently, with no error raised
anywhere: TraCI accepts any correctly-sized phase string. Zones 3 and 4
were unaffected, so the fault appeared only as a puzzling ~50% failure
rate on emergencies, which is exactly what an under-trained policy also
looks like. It was found by asserting the mapping against
getControlledLinks rather than trusting the comment, and that assertion
now runs on every SUMOAdapter construction (see _verify_link_mapping) so
a future edit to the network cannot reintroduce it silently.
"""
import traci

ZONE_TO_EDGE = {
    1: "north_in",
    2: "east_in",
    3: "south_in",
    4: "west_in",
}

EDGE_TO_ZONE = {v: k for k, v in ZONE_TO_EDGE.items()}

TL_ID = "center"

# Phase strings: 12 characters matching link order
# Links: [Z1, Z1, Z1, Z2, Z2, Z2, Z3, Z3, Z3, Z4, Z4, Z4]
ZONE_PHASES = {
    1: "GGGrrrrrrrrr",  # Zone 1 green (north_in, links 0-2)
    2: "rrrGGGrrrrrr",  # Zone 2 green (east_in,  links 3-5)
    3: "rrrrrrGGGrrr",  # Zone 3 green (south_in, links 6-8)
    4: "rrrrrrrrrGGG",  # Zone 4 green (west_in,  links 9-11)
}

# Link index ranges each zone owns, used by the run-time assertion below.
ZONE_LINK_SPANS = {1: (0, 3), 2: (3, 6), 3: (6, 9), 4: (9, 12)}

YELLOW_PHASE = "yyyyyyyyyyyy"
ALL_RED_PHASE = "rrrrrrrrrrrr"


class SUMOAdapter:
    def __init__(self, verify_links=True):
        self.current_zone = 1
        self.step_count = 0
        self.vehicles_cleared = 0
        if verify_links:
            self._verify_link_mapping()

    @staticmethod
    def _verify_link_mapping():
        """
        Assert that ZONE_PHASES still matches the network's real link
        order. Cheap (one TraCI query per episode) and it converts a
        silent, benchmark-invalidating misconfiguration into a loud
        failure at start-up — see this module's docstring for the bug
        this exists to prevent recurring.
        """
        try:
            links = traci.trafficlight.getControlledLinks(TL_ID)
        except traci.exceptions.TraCIException:
            return
        if len(links) != 12:
            raise RuntimeError(
                f"{TL_ID} controls {len(links)} links, expected 12 — "
                "ZONE_PHASES phase strings no longer match this network."
            )
        for zone, (lo, hi) in ZONE_LINK_SPANS.items():
            edge = ZONE_TO_EDGE[zone]
            for idx in range(lo, hi):
                incoming = links[idx][0][0] if links[idx] else ""
                if not incoming.startswith(edge):
                    raise RuntimeError(
                        f"Traffic-light link {idx} belongs to '{incoming}' but "
                        f"ZONE_PHASES assigns it to zone {zone} ('{edge}'). "
                        "Fix ZONE_PHASES before trusting any SUMO result."
                    )

    def get_queue_lengths(self):
        queues = {}
        for zone, edge in ZONE_TO_EDGE.items():
            try:
                queues[zone] = traci.edge.getLastStepHaltingNumber(edge)
            except traci.exceptions.TraCIException:
                queues[zone] = 0
        return queues

    def get_total_waiting_time(self):
        waits = {}
        for zone, edge in ZONE_TO_EDGE.items():
            try:
                waits[zone] = traci.edge.getWaitingTime(edge)
            except traci.exceptions.TraCIException:
                waits[zone] = 0
        return waits

    def get_vehicle_count(self):
        return traci.vehicle.getIDCount()

    def get_departed_count(self):
        return traci.simulation.getArrivedNumber()

    def set_signal(self, zone):
        if zone in ZONE_PHASES:
            try:
                traci.trafficlight.setRedYellowGreenState(TL_ID, ZONE_PHASES[zone])
                self.current_zone = zone
            except traci.exceptions.TraCIException as e:
                print(f"Signal error: {e}")

    def set_yellow(self):
        try:
            traci.trafficlight.setRedYellowGreenState(TL_ID, YELLOW_PHASE)
        except traci.exceptions.TraCIException:
            pass

    def set_all_red(self):
        try:
            traci.trafficlight.setRedYellowGreenState(TL_ID, ALL_RED_PHASE)
        except traci.exceptions.TraCIException:
            pass

    def detect_ambulance(self):
        try:
            for vid in traci.vehicle.getIDList():
                if "ambulance" in vid.lower():
                    edge = traci.vehicle.getRoadID(vid)
                    if edge in EDGE_TO_ZONE:
                        return True, EDGE_TO_ZONE[edge]
            return False, 0
        except traci.exceptions.TraCIException:
            return False, 0

    def get_ambulance_waiting_time(self):
        try:
            for vid in traci.vehicle.getIDList():
                if "ambulance" in vid.lower():
                    return traci.vehicle.getWaitingTime(vid)
            return 0
        except traci.exceptions.TraCIException:
            return 0

    def step(self):
        traci.simulationStep()
        self.step_count += 1
        self.vehicles_cleared += self.get_departed_count()

    def get_metrics(self):
        queue_lengths = self.get_queue_lengths()
        wait_times = self.get_total_waiting_time()
        total_queue = sum(queue_lengths.values())
        total_wait = sum(wait_times.values())

        return {
            "step": self.step_count,
            "queue_lengths": queue_lengths,
            "total_queue": total_queue,
            "wait_times": wait_times,
            "total_wait": total_wait,
            "avg_wait": total_wait / max(total_queue, 1),
            "vehicles_in_sim": self.get_vehicle_count(),
            "vehicles_cleared": self.vehicles_cleared,
            "active_signal": self.current_zone,
            "congestion": total_queue / (4 * 50),
        }
