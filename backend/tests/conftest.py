# tests/conftest.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture(scope="module")
def client():
    """
    Module-scoped TestClient running the app's real lifespan (startup +
    shutdown), including the background TrafficController.run() loop —
    these are real integration tests against the actual FastAPI app and
    its actual singleton services (main.py has no dependency-injection
    layer; traffic_controller/serial_bridge/rl_agent are module-level
    globals), not a set of mocks.

    SerialBridge.connect() falls back to simulation_mode automatically
    when no physical Arduino is present (services/serial_comm.py), so
    this suite runs the same way with or without real hardware attached.

    Module-scoped rather than per-test: constructing the app spins up a
    real background asyncio task and loads the DQN checkpoint from disk,
    which is unnecessary overhead to repeat for every single test
    function, and there is currently no test here that depends on a fully
    clean TrafficController per test (each test's assertions are about
    response shape/status code, not about exact accumulated state).
    """
    with TestClient(main.app) as c:
        yield c
