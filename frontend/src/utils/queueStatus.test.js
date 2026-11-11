import { describe, it, expect } from "vitest";
import {
  queueLevel,
  QUEUE_WARNING_THRESHOLD,
  QUEUE_DANGER_THRESHOLD,
} from "./queueStatus";

describe("queueLevel", () => {
  it("returns 'ok' for an empty queue", () => {
    expect(queueLevel(0)).toBe("ok");
  });

  it("stays 'ok' at exactly the warning threshold, and flips to 'warning' just above it", () => {
    expect(queueLevel(QUEUE_WARNING_THRESHOLD)).toBe("ok");
    expect(queueLevel(QUEUE_WARNING_THRESHOLD + 1)).toBe("warning");
  });

  it("stays 'warning' at exactly the danger threshold, and flips to 'danger' just above it", () => {
    expect(queueLevel(QUEUE_DANGER_THRESHOLD)).toBe("warning");
    expect(queueLevel(QUEUE_DANGER_THRESHOLD + 1)).toBe("danger");
  });

  it("returns 'danger' for a clearly-overloaded queue", () => {
    expect(queueLevel(50)).toBe("danger");
  });
});
