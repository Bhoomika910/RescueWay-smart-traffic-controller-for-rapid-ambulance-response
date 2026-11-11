# tests/test_health.py
"""Integration tests for the top-level status/config endpoints, and a
regression check for each response_model added during the design-review
fix pass (previously only 1 of ~25 endpoints declared one; a response_model
that doesn't actually match what the handler returns fails loudly here
rather than silently shipping an inaccurate /docs schema)."""


class TestHealthAndRoot:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["message"]

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert isinstance(body["connected_clients"], int)
        assert isinstance(body["rl_epsilon"], (int, float))

    def test_arduino_status(self, client):
        r = client.get("/arduino/status")
        assert r.status_code == 200
        body = r.json()
        assert "connected" in body
        assert "simulation_mode" in body

    def test_config(self, client):
        r = client.get("/config")
        assert r.status_code == 200
        assert "arduino" in r.json()


class TestEvalAndDataEndpoints:
    def test_eval_summary(self, client):
        r = client.get("/eval/summary")
        assert r.status_code == 200
        body = r.json()
        assert "performance" in body
        assert "totals" in body

    def test_data_stats(self, client):
        r = client.get("/data/stats")
        assert r.status_code == 200
        body = r.json()
        assert "total_emergencies" in body
        assert "db_path" in body

    def test_compare_results_before_any_run(self, client):
        r = client.get("/compare/results")
        assert r.status_code == 200


class TestLogsErrorHandling:
    def test_download_missing_log_file_returns_404_not_200(self, client, monkeypatch, tmp_path):
        """Regression test: this endpoint previously returned HTTP 200
        with an {"error": "Log file not found"} body when the log file
        didn't exist, indistinguishable from success by status code alone.
        It now raises HTTPException(404)."""
        # Point the log file lookup at a path that can't exist by
        # redirecting os.path.dirname's result — simplest reliable way to
        # force the "not found" branch without deleting the real log file.
        import os

        original_exists = os.path.exists
        monkeypatch.setattr(
            os.path,
            "exists",
            lambda p: False if p.endswith("rescueway.log") else original_exists(p),
        )
        r = client.get("/logs/download")
        assert r.status_code == 404
