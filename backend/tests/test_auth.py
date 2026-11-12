# tests/test_auth.py
"""
Regression tests for routers/auth.py's API-key dependency — design-review
finding (category 17 "Backend engineering"): "zero authentication or
authorization on any endpoint."

config.AUTH_ENABLED/config.API_KEY are read per-request inside
require_api_key() (not cached at import time), so monkeypatching them
directly on the config module works without needing an env var set before
the app starts — see routers/auth.py's docstring for the design rationale.
"""
import config


class TestAuthDisabledByDefault:
    """The local-development bypass is allowed only when BOTH
    API-key and administrator JWT authentication are disabled."""

    def test_zone_select_works_without_any_header_when_both_auth_mechanisms_disabled(
        self, client, monkeypatch
    ):
        assert config.AUTH_ENABLED is False
        monkeypatch.setattr(config, "ADMIN_AUTH_ENABLED", False)
        resp = client.post("/zone/select", json={"zone": 1, "traffic": "medium"})
        assert resp.status_code == 200


class TestAuthEnforced:
    def test_protected_route_rejects_missing_key_when_auth_enabled(self, client, monkeypatch):
        monkeypatch.setattr(config, "AUTH_ENABLED", True)
        monkeypatch.setattr(config, "API_KEY", "test-secret-key")
        resp = client.post("/zone/select", json={"zone": 1, "traffic": "medium"})
        assert resp.status_code == 401

    def test_protected_route_rejects_wrong_key_when_auth_enabled(self, client, monkeypatch):
        monkeypatch.setattr(config, "AUTH_ENABLED", True)
        monkeypatch.setattr(config, "API_KEY", "test-secret-key")
        resp = client.post(
            "/zone/select",
            json={"zone": 1, "traffic": "medium"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_protected_route_accepts_correct_key_when_auth_enabled(self, client, monkeypatch):
        monkeypatch.setattr(config, "AUTH_ENABLED", True)
        monkeypatch.setattr(config, "API_KEY", "test-secret-key")
        resp = client.post(
            "/zone/select",
            json={"zone": 1, "traffic": "medium"},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert resp.status_code == 200

    def test_read_only_route_stays_open_even_when_auth_enabled(self, client, monkeypatch):
        # /zone/status is a GET (read-only) — deliberately not gated by
        # require_api_key so the live dashboard can keep polling/observing
        # without embedding the secret in every viewer. See routers/auth.py.
        monkeypatch.setattr(config, "AUTH_ENABLED", True)
        monkeypatch.setattr(config, "API_KEY", "test-secret-key")
        resp = client.get("/zone/status")
        assert resp.status_code == 200

    def test_clear_route_is_also_protected(self, client, monkeypatch):
        monkeypatch.setattr(config, "AUTH_ENABLED", True)
        monkeypatch.setattr(config, "API_KEY", "test-secret-key")
        resp = client.post("/zone/clear")
        assert resp.status_code == 401
