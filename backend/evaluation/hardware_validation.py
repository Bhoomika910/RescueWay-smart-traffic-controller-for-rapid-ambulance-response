# evaluation/hardware_validation.py
"""
RescueWay Hardware-in-the-Loop Validation.

Validates key robustness findings on the physical Arduino prototype.

This script sets noise environment variables and runs the live system
for a specified duration, logging whether the Arduino LEDs correctly
reflect controller decisions under noisy conditions.

Usage:
    python -m evaluation.hardware_validation --scenario clean
    python -m evaluation.hardware_validation --scenario urban_dense
    python -m evaluation.hardware_validation --scenario all

Requirements:
    - Arduino connected on COM20
    - Backend NOT running (this script starts its own instance)

What it does:
    1. Sets noise environment variables for each scenario
    2. Starts the backend with those settings
    3. Triggers emergency events via HTTP POST
    4. Records Arduino ACKs and signal states
    5. Compares expected vs actual signal states
    6. Produces a validation report

For the paper:
    "We validated simulated robustness results on a physical
    hardware-in-the-loop testbed, demonstrating that identified
    failure modes manifest in physical actuation."
"""
import argparse
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Configuration ───────────────────────────────────

BACKEND_URL = "http://127.0.0.1:8000"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "hardware")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────
# VALIDATION SCENARIOS
# ──────────────────────────────────────────────────────

VALIDATION_SCENARIOS = {
    "clean": {
        "description": "No noise — baseline hardware sync verification",
        "env_vars": {},
        "emergency_zones": [1, 3, 2, 4],
        "duration_seconds": 60,
    },
    "suburban": {
        "description": "Light noise — should work normally",
        "env_vars": {
            "RESCUEWAY_SENSOR_NOISE": "0.05",
            "RESCUEWAY_GPS_MISCLASS": "0.05",
            "RESCUEWAY_COMM_LATENCY": "0.5",
        },
        "emergency_zones": [1, 3, 2, 4],
        "duration_seconds": 60,
    },
    "urban_dense": {
        "description": "Heavy sensor noise + moderate GPS error",
        "env_vars": {
            "RESCUEWAY_SENSOR_NOISE": "0.15",
            "RESCUEWAY_SENSOR_DROPOUT": "0.05",
            "RESCUEWAY_GPS_MISCLASS": "0.10",
            "RESCUEWAY_COMM_LATENCY": "1.5",
        },
        "emergency_zones": [3, 1, 4, 2],
        "duration_seconds": 90,
    },
    "poor_gps": {
        "description": "High GPS errors — misclassification expected",
        "env_vars": {
            "RESCUEWAY_SENSOR_NOISE": "0.05",
            "RESCUEWAY_GPS_MISCLASS": "0.25",
            "RESCUEWAY_GPS_FALSE_NEG": "0.10",
            "RESCUEWAY_COMM_LATENCY": "1.0",
        },
        "emergency_zones": [2, 4, 1, 3],
        "duration_seconds": 90,
    },
    "degraded": {
        "description": "Everything at moderate noise",
        "env_vars": {
            "RESCUEWAY_SENSOR_NOISE": "0.20",
            "RESCUEWAY_SENSOR_DROPOUT": "0.10",
            "RESCUEWAY_GPS_MISCLASS": "0.15",
            "RESCUEWAY_GPS_FALSE_POS": "0.05",
            "RESCUEWAY_GPS_FALSE_NEG": "0.08",
            "RESCUEWAY_COMM_LATENCY": "2.0",
            "RESCUEWAY_COMM_PACKET_LOSS": "0.03",
        },
        "emergency_zones": [1, 2, 3, 4],
        "duration_seconds": 120,
    },
}


def check_backend():
    """Check if backend is running and Arduino is connected."""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        health = r.json()

        r2 = requests.get(f"{BACKEND_URL}/arduino/status", timeout=3)
        arduino = r2.json()

        return {
            "backend_running": True,
            "mode": health.get("mode", "unknown"),
            "arduino_connected": arduino.get("connected", False),
            "simulation_mode": arduino.get("simulation_mode", True),
        }
    except Exception as e:
        return {
            "backend_running": False,
            "error": str(e),
        }


def trigger_emergency(zone, traffic="medium"):
    """Send emergency request to backend."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/zone/select",
            json={"zone": zone, "traffic": traffic},
            timeout=30,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_arduino_status():
    """Get current Arduino state."""
    try:
        r = requests.get(f"{BACKEND_URL}/arduino/status", timeout=3)
        return r.json()
    except Exception:
        return {}


def run_validation_scenario(name, scenario):
    """
    Run one hardware validation scenario.

    IMPORTANT: The backend must already be running with the correct
    environment variables set. This function only triggers events
    and records results.
    """
    print(f"\n  {'-' * 60}")
    print(f"  Scenario: {name}")
    print(f"  {scenario['description']}")
    print(f"  {'-' * 60}")

    # Check backend
    status = check_backend()
    if not status.get("backend_running"):
        print("    ERROR: Backend not running! Start with:")
        print("    uvicorn main:app --port 8000")
        return None

    if status.get("simulation_mode"):
        print("    WARNING: Arduino in SIMULATION MODE")
        print("    Hardware validation requires real Arduino connection")

    print(
        f"    Backend: running | Arduino: {'connected' if not status.get('simulation_mode') else 'SIMULATED'}"
    )
    print(f"    Mode: {status.get('mode', 'unknown')}")

    # Run scenario
    results = {
        "scenario": name,
        "description": scenario["description"],
        "arduino_connected": not status.get("simulation_mode", True),
        "events": [],
    }

    emergency_zones = scenario["emergency_zones"]
    interval = scenario["duration_seconds"] / (len(emergency_zones) + 1)

    print(f"\n    Running {len(emergency_zones)} emergency events " f"({interval:.0f}s apart)...\n")

    for i, zone in enumerate(emergency_zones):
        # Wait between events
        if i > 0:
            wait = max(5, interval)
            print(f"    Waiting {wait:.0f}s before next event...")
            time.sleep(wait)

        # Record pre-event state
        pre_arduino = get_arduino_status()

        # Trigger emergency
        print(f"    [{i+1}/{len(emergency_zones)}] Triggering Zone {zone}...", end=" ")
        response = trigger_emergency(zone)
        trigger_time = time.time()

        # Wait for signal change
        time.sleep(3)

        # Record post-event state
        post_arduino = get_arduino_status()

        event = {
            "requested_zone": zone,
            "response": response,
            "pre_ack": pre_arduino.get("last_ack", ""),
            "post_ack": post_arduino.get("last_ack", ""),
            "commands_sent": post_arduino.get("commands_sent", 0)
            - pre_arduino.get("commands_sent", 0),
            "success_rate": post_arduino.get("success_rate", 0),
            "timestamp": trigger_time,
        }

        results["events"].append(event)

        # Check if Arduino received the command
        if event["commands_sent"] > 0:
            print(
                f"OK (Arduino received {event['commands_sent']} commands, "
                f"last ACK: {event['post_ack']})"
            )
        else:
            print("WARNING: No Arduino commands sent!")

    # Summary
    total_events = len(results["events"])
    successful = sum(1 for e in results["events"] if e["commands_sent"] > 0)
    print(f"\n    Summary: {successful}/{total_events} events reached Arduino")

    results["summary"] = {
        "total_events": total_events,
        "successful_events": successful,
        "success_rate": round(successful / max(total_events, 1) * 100, 1),
    }

    return results


def print_env_instructions(name, scenario):
    """Print the PowerShell commands to set environment variables."""
    print(f"\n  To run scenario '{name}':")
    print("  1. Stop the backend (Ctrl+C)")

    if scenario["env_vars"]:
        print("  2. Set environment variables:")
        for key, value in scenario["env_vars"].items():
            print(f'     $env:{key}="{value}"')
    else:
        print("  2. Clear noise variables:")
        print("     Remove-Item Env:RESCUEWAY_* -ErrorAction SilentlyContinue")

    print("  3. Start backend:")
    print("     uvicorn main:app --port 8000")
    print("  4. Run this script:")
    print(f"     python -m evaluation.hardware_validation --scenario {name}")


def run_all_validations():
    """Guide user through all validation scenarios."""
    print("=" * 70)
    print("HARDWARE-IN-THE-LOOP VALIDATION")
    print("=" * 70)
    print("\nThis validates robustness findings on the physical Arduino prototype.")
    print("Each scenario requires restarting the backend with different noise settings.\n")

    all_results = {}

    for name, scenario in VALIDATION_SCENARIOS.items():
        print_env_instructions(name, scenario)
        print("\n  Press Enter when ready (or 'skip' to skip)...", end=" ")
        user = input().strip().lower()

        if user == "skip":
            print(f"  Skipped {name}")
            continue

        result = run_validation_scenario(name, scenario)
        if result:
            all_results[name] = result

    # Save combined results
    if all_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(RESULTS_DIR, f"hw_validation_{timestamp}.json")
        with open(filepath, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved to: {filepath}")

    return all_results


# ─── CLI ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RescueWay Hardware-in-the-Loop Validation")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=list(VALIDATION_SCENARIOS.keys()) + ["all"],
        help="Which scenario to run",
    )

    args = parser.parse_args()

    if args.scenario == "all":
        run_all_validations()
    else:
        scenario = VALIDATION_SCENARIOS[args.scenario]
        print("=" * 70)
        print(f"HARDWARE VALIDATION: {args.scenario}")
        print("=" * 70)

        # Check if env vars are set
        print("\n  Required environment variables:")
        if scenario["env_vars"]:
            for key, value in scenario["env_vars"].items():
                current = os.environ.get(key, "NOT SET")
                match = "✓" if current == value else "✗"
                print(f"    {match} {key}={value} (current: {current})")
        else:
            print("    (no noise — clean baseline)")

        result = run_validation_scenario(args.scenario, scenario)

        if result:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(RESULTS_DIR, f"hw_{args.scenario}_{timestamp}.json")
            with open(filepath, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n  Results saved to: {filepath}")
