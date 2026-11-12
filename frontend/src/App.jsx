
// src/App.jsx
import { useState, useEffect, lazy, Suspense } from "react";
import useWebSocket from "./hooks/useWebSocket";
import TopBar from "./components/unified/TopBar";
import IntersectionView from "./components/simulation/IntersectionView";
import AnalyticsPanel from "./components/unified/AnalyticsPanel";
import ControlBar from "./components/unified/ControlBar";
import LogViewer from "./components/analytics/LogViewer";
import ErrorBoundary from "./components/ErrorBoundary";
import { API_BASE_URL } from "./api/rescueway";

import AdminLogin from "./components/admin/AdminLogin";
import AdminDashboard from "./components/admin/AdminDashboard";

const WS_URL = API_BASE_URL.replace(/^http/, "ws") + "/ws/live";

const ComparisonChart = lazy(() =>
  import("./components/analytics/ComparisonChart")
);

export default function App() {
  const [page, setPage] = useState("main");
  const [route, setRoute] = useState(window.location.pathname);
  const [isAdmin, setIsAdmin] = useState(false);
  const [adminUsername, setAdminUsername] = useState("");

  const isPublicDemo =
    import.meta.env.VITE_PUBLIC_DEMO_MODE === "true";

  // Keep the live connection active for the public dashboard.
  const { lastEvent, connected } = useWebSocket(WS_URL);

  const navigateTo = (path, replace = false) => {
    if (replace) {
      window.history.replaceState({}, "", path);
    } else {
      window.history.pushState({}, "", path);
    }
    setRoute(path);
  };


  useEffect(() => {
    const handlePopState = () => {
      setRoute(window.location.pathname);
    };

    const handleDashboardNavigation = (event) => {
      setPage(event.detail);
    };

    const handleAdminSessionExpired = () => {
      setIsAdmin(false);
      setAdminUsername("");
      navigateTo("/admin/login", true);
    };

    window.addEventListener("popstate", handlePopState);
    window.addEventListener("navigate", handleDashboardNavigation);
    window.addEventListener(
      "admin-session-expired",
      handleAdminSessionExpired
    );

    return () => {
      window.removeEventListener("popstate", handlePopState);
      window.removeEventListener("navigate", handleDashboardNavigation);
      window.removeEventListener(
        "admin-session-expired",
        handleAdminSessionExpired
      );
    };
  }, []);

  // Require authentication when opening the admin dashboard directly.
  useEffect(() => {
    if (route === "/admin" && !isAdmin) {
      navigateTo("/admin/login", true);
    }
  }, [route, isAdmin]);

  const handleAuthenticated = (username) => {
    setAdminUsername(username);
    setIsAdmin(true);
    navigateTo("/admin");
  };

  const handleLogout = () => {
    setIsAdmin(false);
    setAdminUsername("");
    navigateTo("/admin/login", true);
  };

  // Private administrator login.
  if (route === "/admin/login") {
    return (
      <AdminLogin onAuthenticated={handleAuthenticated} />
    );
  }

  // Private administrator dashboard.
  if (route === "/admin") {
    if (!isAdmin) {
      return (
        <div className="min-h-screen bg-[#0f1117]" />
      );
    }

    return (
      <AdminDashboard
        username={adminUsername}
        onLogout={handleLogout}
        lastEvent={lastEvent}
        connected={connected}
      />
    );
  }

  // Existing public dashboard and local operator interface.
  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ background: "#0f1117" }}
    >
      <TopBar connected={connected} lastEvent={lastEvent} />

      {page !== "main" && (
        <div className="px-6 py-3 flex items-center justify-between border-b border-[#2a2d3a] bg-[#1a1d27]">
          <button
            onClick={() => setPage("main")}
            className="flex items-center gap-2 px-3 py-2 rounded-lg border border-[#2a2d3a] text-sm text-blue-400 hover:bg-[#252838] hover:text-blue-300 transition-all"
          >
            Back to Dashboard
          </button>

          <h2 className="text-lg font-semibold text-white">
            {page === "comparison"
              ? "Controller Benchmark"
              : page === "logs"
              ? "System Logs"
              : page}
          </h2>

          <div className="w-[170px]" />
        </div>
      )}

      <div className="flex-1 p-3">
        {page === "main" && (
          <ErrorBoundary>
            <div className="max-w-[1440px] mx-auto space-y-3 h-full">
              {/* Main: Intersection + Analytics */}
              <div
                className="grid grid-cols-1 lg:grid-cols-5 gap-3"
                style={{ minHeight: "480px" }}
              >
                <div className="lg:col-span-3">
                  <IntersectionView lastEvent={lastEvent} />
                </div>

                <div className="lg:col-span-2">
                  <AnalyticsPanel lastEvent={lastEvent} />
                </div>
              </div>

              {/* Public visitors can see controls but cannot operate them. */}
              <ControlBar
                lastEvent={lastEvent}
                connected={connected}
                readOnly={isPublicDemo}
              />
            </div>
          </ErrorBoundary>
        )}

        {page === "comparison" && (
          <div className="max-w-4xl mx-auto">
            <Suspense
              fallback={
                <div className="rounded-xl border border-[#2a2d3a] bg-[#1e2130] p-6 text-center text-sm text-[#9ca0b0]">
                  Loading benchmark comparison...
                </div>
              }
            >
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
          <span>
            Vehicles: {lastEvent?.system_metrics?.total_vehicles_arrived || 0}
          </span>
          <span>
            Cleared: {lastEvent?.system_metrics?.total_vehicles_cleared || 0}
          </span>
          <span className="capitalize">
            Pattern: {lastEvent?.traffic || "medium"}
          </span>
        </div>

        <div className="flex items-center gap-5">
          <span>
            <span
              className={
                lastEvent?.arduino_ack?.startsWith("OK")
                  ? "text-emerald-400"
                  : "text-amber-400"
              }
            >
              ●
            </span>{" "}
            {lastEvent?.arduino_ack?.startsWith("OK")
              ? "Arduino (HW)"
              : "Arduino (SIM)"}
          </span>

          <span>
            <span
              className={
                connected ? "text-emerald-400" : "text-red-400"
              }
            >
              ●
            </span>{" "}
            WebSocket
          </span>

          <span>RescueWay v2.0</span>
        </div>
      </div>
    </div>
  );
}
