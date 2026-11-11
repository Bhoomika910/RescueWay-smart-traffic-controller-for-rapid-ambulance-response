import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import TopBar from "./TopBar";

// Design-review finding (category 18 "Frontend engineering"): TopBar is
// where a real, previously-shipped bug lived (ARCHITECTURE.md: "a
// dead-state bug in App.jsx where the bottom status bar showed a
// hardcoded 'connected' dot regardless of actual WebSocket/Arduino
// state") — these tests specifically lock in that the connection/mode/
// Arduino badges are actually DERIVED from props, not hardcoded, so that
// class of regression can't silently reappear.

afterEach(() => {
  vi.useRealTimers();
});

describe("TopBar", () => {
  it('shows "Live" when connected is true', () => {
    render(<TopBar connected={true} lastEvent={{ mode: "normal" }} />);
    expect(screen.getByText("Live")).toBeInTheDocument();
  });

  it('shows "Offline" when connected is false — the exact case the dead-state bug got wrong', () => {
    render(<TopBar connected={false} lastEvent={{ mode: "normal" }} />);
    expect(screen.getByText("Offline")).toBeInTheDocument();
    expect(screen.queryByText("Live")).not.toBeInTheDocument();
  });

  it("shows Normal mode and default styling when not in an emergency", () => {
    const { container } = render(<TopBar connected={true} lastEvent={{ mode: "normal" }} />);
    expect(screen.getByText("Normal")).toBeInTheDocument();
    expect(container.firstChild.className).not.toMatch(/bg-red-900/);
  });

  it("shows the emergency zone and red styling when mode is emergency", () => {
    const { container } = render(
      <TopBar connected={true} lastEvent={{ mode: "emergency", emergency_zone: 3 }} />
    );
    expect(screen.getByText("Emergency Z3")).toBeInTheDocument();
    expect(container.firstChild.className).toMatch(/bg-red-900/);
  });

  it('shows "SIM" when no Arduino ACK has been received', () => {
    render(<TopBar connected={true} lastEvent={{ mode: "normal" }} />);
    expect(screen.getByText("SIM")).toBeInTheDocument();
  });

  it('shows "SIM" when the ACK is a simulation ACK (SIM:...)', () => {
    // Regression test: simulated ACKs are truthy strings and previously
    // flipped the badge to the hardware label even with no board attached.
    render(<TopBar connected={true} lastEvent={{ mode: "normal", arduino_ack: "SIM:zone=1,signal=green" }} />);
    expect(screen.getByText("SIM")).toBeInTheDocument();
  });

  it("shows the hardware label only when a real Arduino ACK (OK:...) is present", () => {
    render(<TopBar connected={true} lastEvent={{ mode: "normal", arduino_ack: "OK:zone=1" }} />);
    expect(screen.getByText("HW")).toBeInTheDocument();
  });

  it("displays the current step count from lastEvent", () => {
    render(<TopBar connected={true} lastEvent={{ mode: "normal", step: 42 }} />);
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("defaults to normal mode and step 0 when lastEvent is not yet available", () => {
    render(<TopBar connected={false} lastEvent={null} />);
    expect(screen.getByText("Normal")).toBeInTheDocument();
    expect(screen.getByText("0")).toBeInTheDocument();
  });

  it("cleans up its clock interval on unmount (no leaked timer)", () => {
    vi.useFakeTimers();
    const clearSpy = vi.spyOn(global, "clearInterval");
    const { unmount } = render(<TopBar connected={true} lastEvent={{ mode: "normal" }} />);
    unmount();
    expect(clearSpy).toHaveBeenCalled();
  });
});
