"""
RescueWay SUMO Route Generator
Generates sorted vehicle routes for SUMO evaluation.
"""

import random
import xml.etree.ElementTree as ET

ZONE_EDGES = {1: "north_in", 2: "east_in", 3: "south_in", 4: "west_in"}
ZONE_EXITS = {
    1: ["south_out", "east_out", "west_out"],
    2: ["west_out", "south_out", "north_out"],
    3: ["north_out", "west_out", "east_out"],
    4: ["east_out", "north_out", "south_out"],
}

PATTERNS = {
    "low": {"zone1": 0.05, "zone2": 0.05, "zone3": 0.05, "zone4": 0.05},
    "medium": {"zone1": 0.10, "zone2": 0.10, "zone3": 0.10, "zone4": 0.10},
    "high": {"zone1": 0.15, "zone2": 0.15, "zone3": 0.15, "zone4": 0.15},
    "rush_hour": {"zone1": 0.20, "zone2": 0.18, "zone3": 0.16, "zone4": 0.14},
}


def generate_routes(
    pattern="medium",
    duration=600,
    ambulance_zone=None,
    ambulance_time=None,
    seed=42,
    output_file="routes.rou.xml",
):
    random.seed(seed)
    rates = PATTERNS.get(pattern, PATTERNS["medium"])

    # Collect all vehicles with their departure times
    vehicles = []
    vehicle_id = 0

    for t in range(duration):
        for zone in [1, 2, 3, 4]:
            zone_key = f"zone{zone}"
            rate = rates[zone_key]
            if random.random() < rate:
                entry_edge = ZONE_EDGES[zone]
                exit_edge = random.choice(ZONE_EXITS[zone])
                vehicles.append(
                    {
                        "id": f"car_{vehicle_id}",
                        "type": "car",
                        "depart": t,
                        "edges": f"{entry_edge} {exit_edge}",
                    }
                )
                vehicle_id += 1

    # Add ambulance
    if ambulance_zone:
        amb_time = ambulance_time or random.randint(60, duration - 60)
        entry_edge = ZONE_EDGES[ambulance_zone]
        exit_edge = random.choice(ZONE_EXITS[ambulance_zone])
        vehicles.append(
            {
                "id": "ambulance_0",
                "type": "ambulance",
                "depart": amb_time,
                "edges": f"{entry_edge} {exit_edge}",
            }
        )

    # CRITICAL: Sort by departure time (SUMO requires this)
    vehicles.sort(key=lambda v: v["depart"])

    # Build XML
    root = ET.Element("routes")

    ET.SubElement(
        root,
        "vType",
        id="car",
        accel="2.6",
        decel="4.5",
        sigma="0.5",
        length="5",
        minGap="2.5",
        maxSpeed="13.89",
        color="0.5,0.5,1.0",
    )

    ET.SubElement(
        root,
        "vType",
        id="ambulance",
        accel="3.0",
        decel="5.0",
        sigma="0.3",
        length="7",
        minGap="2.0",
        maxSpeed="16.67",
        color="1.0,1.0,1.0",
        vClass="emergency",
        guiShape="emergency",
    )

    for v in vehicles:
        veh_elem = ET.SubElement(
            root,
            "vehicle",
            id=v["id"],
            type=v["type"],
            depart=str(v["depart"]),
            departLane="free",
            departSpeed="max",
        )
        ET.SubElement(veh_elem, "route", edges=v["edges"])

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_file, xml_declaration=True, encoding="UTF-8")

    print(f"Generated {vehicle_id} vehicles, pattern={pattern}, duration={duration}s")
    print(f"Ambulance: zone={ambulance_zone}, time={ambulance_time or 'random'}")
    print(f"Saved to: {output_file}")

    return vehicle_id


if __name__ == "__main__":
    for pattern in ["low", "medium", "high", "rush_hour"]:
        generate_routes(
            pattern=pattern,
            duration=600,
            ambulance_zone=3,
            ambulance_time=120,
            seed=42,
            output_file=f"routes_{pattern}.rou.xml",
        )
        print()
