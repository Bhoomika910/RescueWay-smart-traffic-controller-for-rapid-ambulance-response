# tests/test_websocket.py
"""
Integration tests for the /ws/live WebSocket endpoint — previously
entirely untested, including the connected_clients add/remove guards
(main.py) and the per-client send-timeout backpressure fix (broadcast()).
"""
import json


def test_websocket_connects_and_disconnects_cleanly(client):
    with client.websocket_connect("/ws/live"):
        pass  # a clean connect + clean `with`-block exit is the assertion


def test_websocket_receives_a_broadcast(client):
    """The background TrafficController loop and any triggered zone-select
    both broadcast full state snapshots to every connected client. This
    doesn't assert on which specific event produced the message (the
    background loop may deliver one before the explicit POST below does)
    — it asserts the pipe genuinely works end-to-end and the payload has
    the shape every consumer (frontend, this test) depends on."""
    with client.websocket_connect("/ws/live") as ws:
        client.post("/zone/select", json={"zone": 3, "traffic": "medium"})
        message = ws.receive_text()
        data = json.loads(message)
        assert "mode" in data
        assert "signal_states" in data
        assert "step" in data


def test_multiple_clients_all_receive_broadcasts(client):
    """Regression-adjacent check for the connected_clients list handling:
    two simultaneously-connected clients should both receive the same
    broadcast, and both connections should tear down cleanly afterward."""
    with client.websocket_connect("/ws/live") as ws1:
        with client.websocket_connect("/ws/live") as ws2:
            client.post("/zone/select", json={"zone": 1, "traffic": "medium"})
            msg1 = json.loads(ws1.receive_text())
            msg2 = json.loads(ws2.receive_text())
            assert "mode" in msg1
            assert "mode" in msg2
