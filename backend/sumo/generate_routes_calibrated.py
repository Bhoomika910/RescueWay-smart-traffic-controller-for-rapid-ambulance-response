# sumo/generate_routes_calibrated.py
"""
Calibrated, Poisson-arrival route generation for RescueWay.

Closes reviewer issue "Artificial traffic demand" on two axes:

1. ARRIVAL PROCESS. generate_routes.py draws one Bernoulli trial per
   second per approach — a binomial process with the wrong inter-arrival
   distribution. Real uninterrupted arrivals are (approximately) Poisson;
   this module draws exponential inter-arrival gaps per approach
   (random.expovariate), the standard microsimulation practice.

2. DEMAND LEVELS. The 'hangzhou' pattern reproduces the measured
   real-world intersection demand published by Xu et al., "Cloud-enabled
   Co-optimization of Priority Vehicle Preemption and Traffic Signal
   Control with Deep Reinforcement Learning" (MT-ITS 2025), Table II —
   the Longwang Sha x Shanhu Sha intersection, Hangzhou: flows of
   267 (S->N), 240 (N->S), 180 (W->E), 275 (E->W) veh/h/lane with
   measured left-turn rates 0.11 / 0.16 / 0.20 / 0.14. Mapped onto this
   network's single-lane approaches as veh/h per approach.

   Turning split assumption (documented, not measured): the published
   table specifies only the left-turn rate; the remaining share is split
   through:right = 4:1, a common default for minor right-turn demand.
   State this assumption wherever the regime is used.

Synthetic regimes (low/medium/high/rush_hour) are kept, converted to
Poisson at the same mean rates as generate_routes.py, so existing
results remain comparable while the arrival process becomes standard.

Geometry (approach heading -> exit edge):
  zone 1 north_in (southbound): left=east_out, through=south_out, right=west_out
  zone 2 east_in  (westbound) : left=south_out, through=west_out,  right=north_out
  zone 3 south_in (northbound): left=west_out,  through=north_out, right=east_out
  zone 4 west_in  (eastbound) : left=north_out, through=east_out,  right=south_out
"""
import random
import xml.etree.ElementTree as ET

ZONE_EDGES = {1: "north_in", 2: "east_in", 3: "south_in", 4: "west_in"}

TURNS = {  # zone -> (left, through, right) exit edges
    1: ("east_out", "south_out", "west_out"),
    2: ("south_out", "west_out", "north_out"),
    3: ("west_out", "north_out", "east_out"),
    4: ("north_out", "east_out", "south_out"),
}

# veh/h per approach; left-turn share of that flow.
PATTERNS = {
    # Synthetic (means match generate_routes.py's Bernoulli rates x 3600)
    "low":       {1: (180, 0.15), 2: (180, 0.15), 3: (180, 0.15), 4: (180, 0.15)},
    "medium":    {1: (360, 0.15), 2: (360, 0.15), 3: (360, 0.15), 4: (360, 0.15)},
    "high":      {1: (540, 0.15), 2: (540, 0.15), 3: (540, 0.15), 4: (540, 0.15)},
    "rush_hour": {1: (720, 0.15), 2: (648, 0.15), 3: (576, 0.15), 4: (504, 0.15)},
    # Calibrated to published real-world counts (MT-ITS 2025, Table II).
    "hangzhou":  {1: (240, 0.16), 2: (275, 0.14), 3: (267, 0.11), 4: (180, 0.20)},
}

THROUGH_SHARE_OF_REST = 0.8  # remainder split through:right = 4:1


def _exits(zone, left_rate, rng):
    left, through, right = TURNS[zone]
    r = rng.random()
    if r < left_rate:
        return left
    if r < left_rate + (1 - left_rate) * THROUGH_SHARE_OF_REST:
        return through
    return right


def generate_routes(pattern="hangzhou", duration=600, ambulance_zone=None,
                    ambulance_time=None, seed=42,
                    output_file="routes.rou.xml"):
    rng = random.Random(seed)
    rates = PATTERNS[pattern]

    vehicles = []
    vid = 0
    for zone, (vph, left_rate) in rates.items():
        lam = vph / 3600.0
        t = rng.expovariate(lam) if lam > 0 else float("inf")
        while t < duration:
            vehicles.append({
                "id": f"car_{vid}", "type": "car", "depart": t,
                "frm": ZONE_EDGES[zone],
                "to": _exits(zone, left_rate, rng),
            })
            vid += 1
            t += rng.expovariate(lam)

    if ambulance_zone:
        amb_t = ambulance_time if ambulance_time is not None \
            else rng.randint(60, duration - 60)
        vehicles.append({
            "id": "ambulance_0", "type": "ambulance", "depart": float(amb_t),
            "frm": ZONE_EDGES[ambulance_zone],
            "to": TURNS[ambulance_zone][1],  # emergencies routed through
        })

    vehicles.sort(key=lambda v: v["depart"])

    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="car", accel="2.6", decel="4.5",
                  length="5", maxSpeed="13.89", sigma="0.5")
    ET.SubElement(root, "vType", id="ambulance", accel="3.0", decel="5.0",
                  length="6.5", maxSpeed="16.67", sigma="0.2",
                  color="1,0,0", vClass="emergency")
    for v in vehicles:
        trip = ET.SubElement(root, "trip", id=v["id"], type=v["type"],
                             depart=f"{v['depart']:.2f}")
        trip.set("from", v["frm"])
        trip.set("to", v["to"])

    ET.ElementTree(root).write(output_file, encoding="unicode",
                               xml_declaration=True)
    print(f"Generated {len(vehicles)} vehicles, pattern={pattern}, "
          f"duration={duration}s (Poisson arrivals)")
    if ambulance_zone:
        print(f"Ambulance: zone={ambulance_zone}, "
              f"time={ambulance_time}")
    print(f"Saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    generate_routes(pattern="hangzhou", duration=3600, seed=0,
                    output_file="routes_hangzhou.rou.xml")
