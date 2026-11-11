# tests/test_zone_router.py
"""
Integration tests for POST /zone/select and the rest of routers/zone.py.

This is the router the project's design-review audit found a real,
reachable crash bug in: ZoneSelectRequest.zone was a bare `int`, so an
out-of-range value passed Pydantic validation cleanly and then crashed
deep inside EmergencyManager.add_emergency with an unhandled KeyError —
an uncaught 500 on the busiest, unauthenticated endpoint in the system.
The fix constrained the field to Literal[1, 2, 3, 4] (models/schemas.py)
plus a defense-in-depth ValueError/HTTPException(400) at the service
layer. test_invalid_zone_is_rejected below is a direct regression test.
"""
import pytest


class TestZoneSelectValidation:
    def test_valid_zone_returns_200(self, client):
        r = client.post("/zone/select", json={"zone": 1, "traffic": "medium"})
        assert r.status_code == 200
        body = r.json()
        assert body["zone"] in (1, 2, 3, 4)
        assert body["action_name"]
        assert "arduino_ack" in body

    @pytest.mark.parametrize("bad_zone", [0, 5, 99, -1])
    def test_invalid_zone_is_rejected_not_crashed(self, client, bad_zone):
        """Regression test for the crash bug described above. Previously
        this reached EmergencyManager.add_emergency and raised an
        unhandled KeyError (uncaught 500); zone is now Literal[1,2,3,4],
        so FastAPI/Pydantic rejects it with a clean 422 before it ever
        reaches business logic."""
        r = client.post("/zone/select", json={"zone": bad_zone, "traffic": "medium"})
        assert r.status_code == 422
        assert r.status_code != 500

    def test_invalid_traffic_pattern_is_rejected(self, client):
        r = client.post("/zone/select", json={"zone": 1, "traffic": "not_a_real_pattern"})
        assert r.status_code == 422

    def test_missing_fields_are_rejected(self, client):
        r = client.post("/zone/select", json={"zone": 1})
        assert r.status_code == 422

    def test_wrong_type_zone_is_rejected(self, client):
        r = client.post("/zone/select", json={"zone": "not_an_int", "traffic": "medium"})
        assert r.status_code == 422


class TestZoneStatusAndClear:
    def test_status_returns_expected_shape(self, client):
        r = client.get("/zone/status")
        assert r.status_code == 200
        body = r.json()
        for key in (
            "mode",
            "active_signal",
            "step",
            "queue_lengths",
            "signal_states",
            "traffic_pattern",
            "status",
        ):
            assert key in body

    def test_clear_after_select_succeeds(self, client):
        client.post("/zone/select", json={"zone": 2, "traffic": "medium"})
        r = client.post("/zone/clear")
        assert r.status_code == 200
        assert "queue" in r.json()

    def test_emergency_queue_endpoint(self, client):
        r = client.get("/zone/emergency-queue")
        assert r.status_code == 200

    def test_emergency_stats_endpoint(self, client):
        r = client.get("/zone/emergency-stats")
        assert r.status_code == 200
