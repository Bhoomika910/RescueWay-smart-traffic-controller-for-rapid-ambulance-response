# main.py
"""
RescueWay backend entrypoint.

Module map (why each top-level package exists, and where its boundary is —
added because the project has three separately-motivated "environment"
implementations that are easy to conflate without this written down: see
the traffic/ vs rl/ vs sumo/ entries below):

  routers/     HTTP + WebSocket surface only. Parses requests, calls into
               services/ or rl/evaluation/, shapes the response. Business
               logic belongs in the layers below, not here.
  services/    Live orchestration and the app's mutable runtime state —
               TrafficController (the single source of truth for signal
               state; see its own module docstring) and EmergencyManager
               (ambulance priority-queue scheduling).
  rl/          Training and inference code for both agents: DQNAgent
               (PyTorch, continuous 19-dim state) and QAgent (tabular,
               discrete 4320-state). rl/environment.py's TrafficEnv is a
               thin Gym-style wrapper around traffic/simulator.py — the RL
               training loop's view of that simulator, not a second
               physics implementation.
  traffic/     The actual (simplified, non-SUMO) queueing physics used by
               the live dashboard, rl/environment.py, and most of
               evaluation/. Poisson arrivals, constant-rate clearance — a
               fast, low-fidelity stand-in for SUMO, not a competing
               ground truth.
  sumo/        A separate, high-fidelity microscopic simulation (real
               TraCI/SUMO, per-vehicle car-following dynamics) used only
               for DQN's actual training and for sumo/evaluate_in_sumo_v2.py.
               Any result comparing controllers must state which of
               traffic/ or sumo/ produced it — the two are not
               interchangeable ground truth, and mixing them in one table
               without saying so was a real, previously-flagged reporting bug.
  evaluation/  Offline benchmarking, robustness/ablation studies, and
               statistical significance testing — consumes controllers/
               and metrics/calculator.py, writes to results/ and the
               benchmark_results DB table.
  metrics/     Two distinct things sharing a folder: collector.py is the
               *live* dashboard's rolling-window stats (used by
               services/traffic_controller.py every tick); calculator.py
               is the *offline* evaluation suite's metric aggregation
               (used by evaluation/). Neither is a dependency of the other.
  controllers/ The 5 pluggable decision-makers benchmarked against each
               other (DQN, tabular Q-Learning, rule-based, fixed-timer,
               random) — a shared interface so evaluation/ can swap them.
  gps/         The 4-zone ETA/distance lookup model (zone_model.py) —
               deliberately a fixed-constant-per-zone simplification, not
               a real GPS pipeline; see its own module docstring.
  noise/       Sensor/GPS/comm noise injection for robustness testing.
  data/        SQLite persistence (RescueWayDB) — the only place that
               touches the .db file directly.
  arduino/     The physical firmware (traffic_controller.ino) — talked to
               over serial via services/serial_comm.py, not from here.
"""
import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import config
from config import get_frontend_origins
from models.schemas import ArduinoStatusResponse, HealthResponse, RootResponse
from rl.dqn_agent import DQNAgent

# Safe to import at module level (not deferred like `from main import
# traffic_controller` inside route handlers below): none of these router
# modules reference anything from main.py at their own module scope, only
# inside function bodies — so there's no circular-import hazard, and they
# don't need to wait for `app = FastAPI(...)` to exist first. Previously
# interleaved one-per-`app.include_router()` call further down the file,
# which triggered ruff's E402 (import not at top of file) six times for no
# functional reason.
from routers.comparison import router as comparison_router
from routers.data_api import router as data_router
from routers.evaluation import router as evaluation_router
from routers.logs import router as logs_router
from routers.simulation import router as simulation_router
from routers.zone import router as zone_router
from services.serial_comm import SerialBridge
from services.traffic_controller import TrafficController

serial_bridge = SerialBridge()
# inference_mode=True forces epsilon to DQNAgent's inference_epsilon
# (0.0 — fully greedy). Previously this passed epsilon=config.RL_EPSILON
# (0.05), which meant the live production controller took a random action
# ~5% of the time in normal-mode cycling while every benchmark/evaluation
# script in the codebase evaluates this same agent class at epsilon=0.0 —
# the benchmarked numbers described a policy the live system didn't
# actually run. See DQNAgent.__init__'s inference_epsilon docstring.
rl_agent = DQNAgent(inference_mode=True)
traffic_controller = TrafficController()

connected_clients = []

# Per-client send timeout for the live broadcast. This function is awaited
# from inside TrafficController's hot per-tick decision loop (see
# _broadcast_full), so without a timeout a single stalled/slow client
# (e.g. a laptop that went to sleep with a dashboard tab open) can hold up
# delivery to every other client AND stall the live signal-control loop
# itself for however long that client's socket takes to fail — not just a
# dashboard-rendering delay, a control-loop delay. asyncio.wait_for bounds
# that to WS_SEND_TIMEOUT_S regardless of what the slow client does.
WS_SEND_TIMEOUT_S = 2.0


async def broadcast(data: dict):
    message = json.dumps(data)
    disconnected = []
    clients_snapshot = list(connected_clients)
    for client in clients_snapshot:
        try:
            await asyncio.wait_for(client.send_text(message), timeout=WS_SEND_TIMEOUT_S)
        except Exception:
            # Covers both real send failures and asyncio.TimeoutError —
            # either way this client is treated as gone rather than being
            # allowed to block the rest of the broadcast indefinitely.
            disconnected.append(client)
    for client in disconnected:
        if client in connected_clients:
            connected_clients.remove(client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        serial_bridge.connect()
        print("Arduino connected successfully")
    except Exception as e:
        print(f"Arduino not connected: {e}")
        print("Running in simulation mode")

    rl_agent.load()
    print(f"DQN agent ready | epsilon={rl_agent.epsilon} | inference_mode=True")

    if not config.AUTH_ENABLED:
        print(
            "WARNING: RESCUEWAY_API_KEY is not set — state-mutating routes "
            "(/zone/select, /zone/clear, /compare/run, /eval/benchmark, "
            "/eval/stress-test, /logs/export, /simulation/*) are UNAUTHENTICATED. "
            "This is fine for local development; set RESCUEWAY_API_KEY before "
            "exposing this server beyond localhost. See .env.example."
        )

    # Give traffic controller access to serial, broadcast, and RL
    traffic_controller.set_dependencies(serial_bridge, broadcast, rl_agent)

    # Start continuous traffic cycle in background
    task = asyncio.create_task(traffic_controller.run())
    print("Continuous traffic controller started")

    from logging_system import app_logger

    app_logger.log_system_event(
        "Backend started",
        {
            "arduino": serial_bridge.conn is not None,
            "rl_epsilon": rl_agent.epsilon,
        },
    )

    yield

    # Shutdown
    traffic_controller.running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    serial_bridge.close()


app = FastAPI(
    title="RescueWay API",
    description="RL-based Emergency Traffic Signal Control",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_frontend_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(zone_router)
app.include_router(comparison_router)
app.include_router(evaluation_router)
app.include_router(simulation_router)
app.include_router(logs_router)
app.include_router(data_router)


@app.websocket("/ws/live")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    print(f"Dashboard connected. Total clients: {len(connected_clients)}")
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        # Guarded like broadcast()'s own removal (main.py's broadcast
        # function above) — without this check, a client already removed
        # there (e.g. a send failing right before this disconnect handler
        # runs) would raise ValueError on a second .remove() of the same
        # element.
        if ws in connected_clients:
            connected_clients.remove(ws)
        print(f"Dashboard disconnected. Total clients: {len(connected_clients)}")


@app.get("/", response_model=RootResponse)
async def root():
    return RootResponse(
        message="RescueWay API is running",
        mode=traffic_controller.mode,
        current_zone=traffic_controller.current_zone,
        arduino=serial_bridge.conn is not None,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        mode=traffic_controller.mode,
        current_zone=traffic_controller.current_zone,
        arduino=serial_bridge.conn is not None,
        rl_epsilon=rl_agent.epsilon,
        connected_clients=len(connected_clients),
    )


@app.get("/arduino/status", response_model=ArduinoStatusResponse)
async def arduino_status():
    return serial_bridge.get_status()


@app.get("/config")
async def get_config():
    from config import get_config_summary

    return get_config_summary()
