import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import useWebSocket from "./useWebSocket";

/**
 * Minimal mock WebSocket. Real WebSocket delivers open/message/close/error
 * via async events; here we just expose the same on* handler slots the hook
 * assigns to, and let tests invoke them synchronously to drive behavior.
 */
class MockWebSocket {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = MockWebSocket.OPEN;
    this.onopen = null;
    this.onmessage = null;
    this.onclose = null;
    this.onerror = null;
    this.close = vi.fn(() => {
      this.readyState = MockWebSocket.CLOSED;
    });
    MockWebSocket.instances.push(this);
  }
}
MockWebSocket.CONNECTING = 0;
MockWebSocket.OPEN = 1;
MockWebSocket.CLOSING = 2;
MockWebSocket.CLOSED = 3;

describe("useWebSocket", () => {
  let originalWebSocket;

  beforeEach(() => {
    originalWebSocket = global.WebSocket;
    MockWebSocket.instances = [];
    global.WebSocket = MockWebSocket;
  });

  afterEach(() => {
    global.WebSocket = originalWebSocket;
    vi.restoreAllMocks();
  });

  it("updates lastEvent when a valid JSON message arrives", () => {
    const { result } = renderHook(() => useWebSocket("ws://test/live"));
    const socket = MockWebSocket.instances[0];

    act(() => {
      socket.onmessage({ data: JSON.stringify({ mode: "normal", step: 5 }) });
    });

    expect(result.current.lastEvent).toEqual({ mode: "normal", step: 5 });
  });

  it("does not throw on a malformed message and leaves lastEvent unchanged", () => {
    const { result } = renderHook(() => useWebSocket("ws://test/live"));
    const socket = MockWebSocket.instances[0];

    act(() => {
      socket.onmessage({ data: JSON.stringify({ mode: "normal", step: 1 }) });
    });
    const prev = result.current.lastEvent;

    expect(() => {
      act(() => {
        socket.onmessage({ data: "{not valid json" });
      });
    }).not.toThrow();

    expect(result.current.lastEvent).toEqual(prev);
  });

  it("closes the underlying socket when the consuming component unmounts", () => {
    const { unmount } = renderHook(() => useWebSocket("ws://test/live"));
    const socket = MockWebSocket.instances[0];

    unmount();

    expect(socket.close).toHaveBeenCalled();
  });
});
