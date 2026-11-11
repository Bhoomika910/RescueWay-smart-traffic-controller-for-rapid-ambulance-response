# config.py
"""
RescueWay Configuration
All settings in one place. No more hardcoded values.
"""
import os

# ─── Arduino Hardware ─────────────────────────────────
ARDUINO_PORT = os.getenv("RESCUEWAY_ARDUINO_PORT", "COM3")
ARDUINO_BAUD = int(os.getenv("RESCUEWAY_ARDUINO_BAUD", "9600"))

# ─── Server ───────────────────────────────────────────
API_HOST = os.getenv("RESCUEWAY_HOST", "0.0.0.0")
API_PORT = int(os.getenv("RESCUEWAY_PORT", "8000"))
# Comma-separated so one env var can list multiple allowed CORS origins
# (e.g. "http://localhost:5173,http://localhost:5174"). Previously this
# value existed and was documented but main.py's CORSMiddleware hardcoded
# its own literal origin list instead of reading it — the env var override
# silently did nothing. get_frontend_origins() is now the one place that
# turns this into the list CORSMiddleware actually uses.
FRONTEND_ORIGIN = os.getenv("RESCUEWAY_FRONTEND", "http://localhost:5173,http://localhost:5174")


def get_frontend_origins() -> list:
    """Parse FRONTEND_ORIGIN into the list CORSMiddleware(allow_origins=...) expects."""
    return [origin.strip() for origin in FRONTEND_ORIGIN.split(",") if origin.strip()]


# ─── API authentication ───────────────────────────────
# State-mutating routes (trigger/clear emergency, run comparisons, etc.)
# previously had no authentication at all — any client that could reach the
# API could inject or clear an emergency. Design-review finding (Part 8
# review, category 17 "Backend engineering": "zero authentication on any
# endpoint"). Fixed with a minimal shared-secret API key rather than a full
# user/session/OAuth system: this deployment has no concept of distinct
# human users (it's a single-site traffic controller talking to one
# dashboard and one Arduino), so a bearer-token-style shared secret is the
# right-sized mechanism — see routers/auth.py for the dependency and
# ARCHITECTURE.md Part 10 for the full rationale (proportionate auth, not
# security theater).
#
# Unset (empty string, the default) means auth is DISABLED — required so
# existing local-dev workflows and the test suite keep working unchanged
# unless an operator opts in by setting RESCUEWAY_API_KEY. Any non-empty
# value enables enforcement. main.py logs a loud warning at startup when
# auth is disabled so this is never a silent gap in a real deployment.
API_KEY = os.getenv("RESCUEWAY_API_KEY", "")
AUTH_ENABLED = bool(API_KEY)

# ─── Traffic Signal Timings ───────────────────────────
DEFAULT_GREEN_DURATION = 5  # seconds
DEFAULT_YELLOW_DURATION = 2  # seconds
DEFAULT_ALL_RED_DURATION = 1  # seconds (all-red safety clearance interval)
DEFAULT_EMERGENCY_DURATION = 15  # seconds (fallback if GPS model not used)

# ─── Traffic Simulator ────────────────────────────────
DEFAULT_TRAFFIC_PATTERN = "medium"
DEFAULT_ARRIVAL_RATES = {
    "1": 1.0,
    "2": 1.0,
    "3": 1.0,
    "4": 1.0,
}
QUEUE_MAX_CAPACITY = 50
VEHICLE_CLEARANCE_RATE = 5  # vehicles per green step

# ─── Reinforcement Learning ──────────────────────────
# NOTE: no longer applied to the live DQN agent (main.py) — DQNAgent's
# inference_mode=True forces epsilon to its own inference_epsilon (0.0,
# fully greedy) regardless of this value, so the deployed policy matches
# what every benchmark/evaluation script actually measures. Kept here
# (and surfaced via GET /config) as a documented reference value only.
RL_EPSILON = 0.05  # historical runtime exploration rate (unused by main.py)
RL_ALPHA = 0.1  # learning rate
RL_GAMMA = 0.95  # discount factor
RL_QTABLE_PATH = os.path.join(os.path.dirname(__file__), "rl", "q_table_v2.pkl")

# ─── Training ────────────────────────────────────────
TRAINING_EPISODES = 25000
TRAINING_EPSILON_START = 1.0
TRAINING_EPSILON_END = 0.05
TRAINING_AMBULANCE_PROBABILITY = 0.08
TRAINING_MAX_STEPS = 30

# ─── GPS Zone Model ─────────────────────────────────
AMBULANCE_SPEED_MPS = 14.0  # meters per second (~50 km/h)

# ─── Metrics ─────────────────────────────────────────
METRICS_WINDOW_SIZE = int(os.getenv("RESCUEWAY_METRICS_WINDOW", "100"))
METRICS_EMERGENCY_WINDOW = int(os.getenv("RESCUEWAY_EMERGENCY_WINDOW", "200"))

# ─── Logging ─────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5MB
LOG_BACKUP_COUNT = 3
LOG_MEMORY_SIZE = 500

# ─── Database ────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "rescueway.db")

# ─── Benchmark ───────────────────────────────────────
BENCHMARK_MAX_EPISODES = 500
BENCHMARK_EMERGENCY_PROBABILITY = 0.20
BENCHMARK_SEED = int(os.getenv("RESCUEWAY_BENCHMARK_SEED", "42"))

# ─── DQN Model ───────────────────────────────────────
RL_DQN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "rl", "dqn_model.pt")

# ─── Evaluation environment provenance ──────────────
# Every evaluation/robustness/ablation/significance result should declare
# which environment produced it. DQN trains exclusively in SUMO
# (sumo/train_dqn_sumo.py); sumo/evaluate_in_sumo_v2.py is the only
# harness that evaluates it in that same environment. Every other
# evaluation script (comparator.py, robustness_runner.py,
# ablation_studies.py, multi_ambulance_significance.py,
# gps_confusion_analysis.py) evaluates against traffic/simulator.py's
# lower-fidelity simplified physics instead — a real, previously-uncaught
# reporting bug was mixing numbers from the two without saying so. Single
# source of truth for the tag strings so every script says the same thing.
ENV_TAG_SIMPLIFIED = "simplified_traffic_env"
ENV_TAG_SUMO = "sumo"

# ─── Noise Configuration ────────────────────────────
# All noise parameters are managed via environment variables
# and the noise.NoiseConfig module. See noise/noise_models.py
# for the full list of RESCUEWAY_* environment variables.


def get_config_summary():
    """Return all config as dict for API endpoint."""
    return {
        "arduino": {
            "port": ARDUINO_PORT,
            "baud": ARDUINO_BAUD,
        },
        "server": {
            "host": API_HOST,
            "port": API_PORT,
            "frontend_origin": FRONTEND_ORIGIN,
        },
        "timings": {
            "green": DEFAULT_GREEN_DURATION,
            "yellow": DEFAULT_YELLOW_DURATION,
            "emergency": DEFAULT_EMERGENCY_DURATION,
        },
        "simulator": {
            "pattern": DEFAULT_TRAFFIC_PATTERN,
            "arrival_rates": DEFAULT_ARRIVAL_RATES,
            "queue_capacity": QUEUE_MAX_CAPACITY,
            "clearance_rate": VEHICLE_CLEARANCE_RATE,
        },
        "rl": {
            "epsilon": RL_EPSILON,
            "alpha": RL_ALPHA,
            "gamma": RL_GAMMA,
            "qtable_path": RL_QTABLE_PATH,
            "dqn_model_path": RL_DQN_MODEL_PATH,
            "primary_agent": "DQN",
        },
        "training": {
            "episodes": TRAINING_EPISODES,
            "ambulance_probability": TRAINING_AMBULANCE_PROBABILITY,
        },
        "metrics": {
            "window_size": METRICS_WINDOW_SIZE,
            "emergency_window": METRICS_EMERGENCY_WINDOW,
        },
        "logging": {
            "log_dir": LOG_DIR,
            "max_bytes": LOG_MAX_BYTES,
        },
        "database": {
            "path": DB_PATH,
        },
    }
