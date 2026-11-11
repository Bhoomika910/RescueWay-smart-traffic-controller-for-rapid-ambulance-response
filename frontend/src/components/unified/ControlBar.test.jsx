import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ControlBar from "./ControlBar";
import * as api from "../../api/rescueway";

vi.mock("../../api/rescueway", () => ({
  selectZone: vi.fn().mockResolvedValue({}),
  setTrafficPattern: vi.fn().mockResolvedValue({}),
  setTimings: vi.fn().mockResolvedValue({}),
  resetSimulation: vi.fn().mockResolvedValue({}),
  setArrivalRates: vi.fn().mockResolvedValue({}),
}));

describe("ControlBar", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calls setTrafficPattern with the clicked pattern's id", async () => {
    const user = userEvent.setup();
    render(<ControlBar lastEvent={{ mode: "normal", traffic: "medium" }} connected={true} />);

    await user.click(screen.getByRole("button", { name: "Low" }));

    await waitFor(() => {
      expect(api.setTrafficPattern).toHaveBeenCalledWith("low");
    });
  });

  it("derives the active-pattern highlight from lastEvent.traffic, including on external changes", () => {
    const { rerender } = render(
      <ControlBar lastEvent={{ mode: "normal", traffic: "medium" }} connected={true} />
    );

    // Initially "medium" is active per lastEvent.traffic.
    expect(screen.getByRole("button", { name: "Medium" }).className).toMatch(/bg-yellow-600/);
    expect(screen.getByRole("button", { name: "Low" }).className).not.toMatch(/bg-emerald-600/);

    // Simulate the pattern changing from some other source (not this component's
    // own click handler) by re-rendering with a new lastEvent.traffic value.
    rerender(<ControlBar lastEvent={{ mode: "normal", traffic: "low" }} connected={true} />);

    expect(screen.getByRole("button", { name: "Low" }).className).toMatch(/bg-emerald-600/);
    expect(screen.getByRole("button", { name: "Medium" }).className).not.toMatch(/bg-yellow-600/);
  });
});
