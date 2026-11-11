import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import ErrorBoundary from "./ErrorBoundary";

// Design-review finding (category 18 "Frontend engineering"): only two
// test files existed for the entire live dashboard, and ErrorBoundary
// (a safety-relevant component — its whole job is preventing one panel's
// crash from blanking the rest of the dashboard) had none. These verify
// its actual documented behavior: render children normally, catch a
// child's render error, and NOT swallow errors raised outside of render
// (React error boundaries only catch render/lifecycle errors by design,
// not e.g. event-handler exceptions — asserting that boundary matters
// because it is often misunderstood as "catches all errors").

function Bomb() {
  throw new Error("boom");
}

describe("ErrorBoundary", () => {
  it("renders children normally when nothing throws", () => {
    render(
      <ErrorBoundary>
        <p>all good</p>
      </ErrorBoundary>
    );
    expect(screen.getByText("all good")).toBeInTheDocument();
  });

  it("renders a fallback panel instead of crashing when a child throws during render", () => {
    // React logs the error to console during the throw — expected noise,
    // suppressed here so the test output stays readable.
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    render(
      <ErrorBoundary>
        <Bomb />
      </ErrorBoundary>
    );

    expect(screen.getByText("Something went wrong displaying this panel")).toBeInTheDocument();
    expect(screen.getByText("boom")).toBeInTheDocument();

    consoleSpy.mockRestore();
  });

  it("isolates the failure to this boundary's own subtree, not siblings", () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    render(
      <div>
        <ErrorBoundary>
          <Bomb />
        </ErrorBoundary>
        <p>sibling panel still rendered</p>
      </div>
    );

    expect(screen.getByText("sibling panel still rendered")).toBeInTheDocument();
    consoleSpy.mockRestore();
  });
});
