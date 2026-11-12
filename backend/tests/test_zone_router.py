# tests/test_zone_router.py
"""
Integration tests for POST /zone/select and the rest of routers/zone.py.

These tests cover zone validation and router behaviour.
Authentication enforcement is tested separately in test_auth.py.
"""
import pytest
import config


@pytest.fixture(autouse=True)
def disable_auth_for_zone_router_tests(monkeypatch):
    """Isolate zone tests from authentication configuration."""
    monkeypatch.setattr(config, "AUTH_ENABLED", False)
    monkeypatch.setattr(config, "ADMIN_AUTH_ENABLED", False)


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
        r = client.post("/zone/select", json={"zone": bad_zone, "traffic": "medium"})
        assert r.status_code == 422
        assert r.status_code != 500

    def test_invalid_traffic_pattern_is_rejected(self, client):
        r = client.post(
            "/zone/select",
            json={"zone": 1, "traffic": "not_a_real_pattern"},
        )
        assert r.status_code == 422

    def test_missing_fields_are_rejected(self, client):
        r = client.post("/zone/select", json={"zone": 1})
        assert r.status_code == 422

    def test_wrong_type_zone_is_rejected(self, client):
        r = client.post(
            "/zone/select",
            json={"zone": "not_an_int", "traffic": "medium"},
        )
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