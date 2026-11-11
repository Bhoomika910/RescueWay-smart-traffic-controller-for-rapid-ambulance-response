# tests/test_websocket_load.py
"""
WebSocket broadcast load test — design-review finding (category 25
"Performance"): "no load testing of the WebSocket broadcast path under
many concurrent clients (the backpressure timeout added this session is a
safety net, not a tested-at-scale guarantee)."

Opens N simultaneous /ws/live connections (well beyond this project's
realistic single-dashboard deployment, to actually stress the fan-out
path rather than just confirm 2 clients work — see
test_websocket.py::test_multiple_clients_all_receive_broadcasts for that
baseline case), triggers one broadcast, and measures per-client latency
from trigger to receipt — reporting p50/p99, not just "it eventually
worked," and asserting every client's latency stays well under
main.py's WS_SEND_TIMEOUT_S (2.0s) backpressure cap.
"""
import json
import statistics
import time
from contextlib import ExitStack

N_CLIENTS = 25


def test_broadcast_fanout_latency_under_load(client):
    with ExitStack() as stack:
        connections = [
            stack.enter_context(client.websocket_connect("/ws/live")) for _ in range(N_CLIENTS)
        ]

        trigger_time = time.perf_counter()
        client.post("/zone/select", json={"zone": 2, "traffic": "medium"})

        latencies = []
        for ws in connections:
            message = ws.receive_text()
            latencies.append(time.perf_counter() - trigger_time)
            data = json.loads(message)
            assert "mode" in data  # every client gets the real payload, not a truncated one

    latencies.sort()
    p50 = statistics.median(latencies)
    p99 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]
    max_latency = max(latencies)

    print(
        f"\nWebSocket fan-out latency ({N_CLIENTS} concurrent clients): "
        f"p50={p50*1000:.1f}ms p99={p99*1000:.1f}ms max={max_latency*1000:.1f}ms"
    )

    # WS_SEND_TIMEOUT_S=2.0 is the per-client cap main.py's broadcast()
    # enforces (see its docstring) — every client receiving well within
    # that, even at N_CLIENTS concurrent connections, confirms the
    # backpressure fix doesn't just prevent a hang, it keeps fan-out fast
    # under load in the common case (no actually-slow client present).
    assert max_latency < 2.0, (
        f"slowest of {N_CLIENTS} clients took {max_latency:.2f}s — approaching "
        f"the 2.0s backpressure cap even with no genuinely slow client present"
    )


def test_all_clients_receive_well_formed_broadcasts_under_load(client):
    """Fan-out correctness under load: every one of N_CLIENTS connected
    clients must receive a complete, well-formed broadcast — no client
    left out, no partial/corrupted write reaching only some of them.

    Deliberately NOT asserting byte-identical payloads across clients —
    an earlier version of this test did, and it genuinely failed: this is
    a live system with a continuously-running background TrafficController
    loop broadcasting on every tick (not just in response to this test's
    POST), so a client that connected slightly earlier than another can
    legitimately have an OLDER background-loop broadcast already queued
    ahead of the one this test triggers — real system behavior, not a
    bug, and the same non-determinism test_websocket.py's
    test_websocket_receives_a_broadcast already documents and works
    around. Structural well-formedness (every client got a complete,
    parseable state snapshot) is the correctness property that actually
    matters for "did the fan-out work," and is what this asserts."""
    required_keys = {"mode", "signal_states", "step", "reward"}

    with ExitStack() as stack:
        connections = [
            stack.enter_context(client.websocket_connect("/ws/live")) for _ in range(N_CLIENTS)
        ]

        client.post("/zone/select", json={"zone": 4, "traffic": "medium"})

        payloads = [json.loads(ws.receive_text()) for ws in connections]

    assert len(payloads) == N_CLIENTS
    for data in payloads:
        missing = required_keys - data.keys()
        assert not missing, f"broadcast to a client under load was missing keys: {missing}"
