// src/App.jsx
import { useState, useEffect, lazy, Suspense } from "react";
import useWebSocket      from "./hooks/useWebSocket";
import TopBar            from "./components/unified/TopBar";
import IntersectionView  from "./components/simulation/IntersectionView";
import AnalyticsPanel    from "./components/unified/AnalyticsPanel";
import ControlBar        from "./components/unified/ControlBar";
import LogViewer         from "./components/analytics/LogViewer";
import ErrorBoundary     from "./components/ErrorBoundary";
import { API_BASE_URL }  from "./api/rescueway";

// Derived from the same VITE_API_BASE_URL as the REST client (see
// src/api/rescueway.js) so the WebSocket and HTTP clients can never point
// at different backends — previously hardcoded independently here as
// "ws://127.0.0.1:8000/ws/live".
const WS_URL = API_BASE_URL.replace(/^http/, "ws") + "/ws/live";

// Lazy-loaded: pulls in recharts, only needed on the benchmark comparison page.
const ComparisonChart = lazy(() => import("./components/analytics/ComparisonChart"));

/** @typedef {import('./types').LiveEvent} LiveEvent */

export default function App() {
  const [page, setPage]     = useState("main");
  const isPublicDemo = import.meta.env.VITE_PUBLIC_DEMO_MODE === "true";

  /** @type {{ lastEvent: LiveEvent | null, connected: boolean }} */
  const { lastEvent, connected } = useWebSocket(WS_URL);

  useEffect(() => {
    const h = (e) => setPage(e.detail);
    window.addEventListener("navigate", h);
    return () => window.removeEventListener("navigate", h);
  }, []);

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "#0f1117" }}>
      <TopBar connected={connected} lastEvent={lastEvent} />

      {page !== "main" && (
  <div className="px-6 py-3 flex items-center justify-between border-b border-[#2a2d3a] bg-[#1a1d27]">
    <button
      onClick={() => setPage("main")}
      className="flex items-center gap-2 px-3 py-2 rounded-lg border border-[#2a2d3a] text-sm text-blue-400 hover:bg-[#252838] hover:text-blue-300 transition-all"
    >
      ← Back to Dashboard
    </button>

    <h2 className="text-lg font-semibold text-white">
      {page === "comparison"
        ? "Controller Benchmark"
        : page === "logs"
        ? "System Logs"
        : page}
    </h2>

    {/* Keeps the title centered */}
    <div className="w-[170px]" />
  </div>
)}

      <div className="flex-1 p-3">
        {page === "main" && (
          <ErrorBoundary>
            <div className="max-w-[1440px] mx-auto space-y-3 h-full">
              {/* Main: Intersection + Analytics */}
              <div className="grid grid-cols-1 lg:grid-cols-5 gap-3" style={{ minHeight: "480px" }}>
                <div className="lg:col-span-3">
                  <IntersectionView lastEvent={lastEvent} />
                </div>
                <div className="lg:col-span-2">
                  <AnalyticsPanel lastEvent={lastEvent} />
                </div>
              </div>

              {/* Controls */}
              {!isPublicDemo && (
                <ControlBar lastEvent={lastEvent} connected={connected} />
              )}
            </div>
          </ErrorBoundary>
        )}

        {page === "comparison" && (
          <div className="max-w-4xl mx-auto">
            <Suspense fallback={
              <div className="rounded-xl border border-[#2a2d3a] bg-[#1e2130] p-6 text-center text-sm text-[#9ca0b0]">
                Loading benchmark comparison…
              </div>
            }>
              <ComparisonChart />
            </Suspense>
          </div>
        )}

        {page === "logs" && (
          <div className="max-w-4xl mx-auto">
            <LogViewer />
          </div>
        )}
      </div>

      {/* Bottom bar */}
      <div className="px-5 py-1.5 flex items-center justify-between text-[10px] text-[#6b7080] border-t border-[#2a2d3a] bg-[#0f1117]">
        <div className="flex items-center gap-5">
          <span>Vehicles: {lastEvent?.system_metrics?.total_vehicles_arrived || 0}</span>
          <span>Cleared: {lastEvent?.system_metrics?.total_vehicles_cleared || 0}</span>
          <span className="capitalize">Pattern: {lastEvent?.traffic || "medium"}</span>
        </div>
        <div className="flex items-center gap-5">
          <span><span className={lastEvent?.arduino_ack?.startsWith("OK") ? "text-emerald-400" : "text-amber-400"}>●</span> {lastEvent?.arduino_ack?.startsWith("OK") ? "Arduino (HW)" : "Arduino (SIM)"}</span>
          <span><span className={connected?"text-emerald-400":"text-red-400"}>●</span> WebSocket</span>
          <span>RescueWay v2.0</span>
        </div>
      </div>
    </div>
  );
}