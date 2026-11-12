
import { useEffect, useState } from "react";
import ControlBar from "../unified/ControlBar";
import {
  getHealth,
  getSimulationStatus,
  getZoneStatus,
  setAdminToken,
} from "../../api/rescueway";

export default function AdminDashboard({ username, onLogout, lastEvent, connected }) {
  const [health, setHealth] = useState(null);
  const [simulation, setSimulation] = useState(null);
  const [zones, setZones] = useState(null);
  const [error, setError] = useState("");

  const refreshData = async () => {
    try {
      const [healthData, simulationData, zoneData] = await Promise.all([
        getHealth(),
        getSimulationStatus(),
        getZoneStatus(),
      ]);

      setHealth(healthData);
      setSimulation(simulationData);
      setZones(zoneData);
      setError("");
    } catch (err) {
      setError(err.response?.data?.detail || "Unable to load dashboard data.");
    }
  };

  useEffect(() => {
    refreshData();
    const interval = setInterval(refreshData, 10000);
    return () => clearInterval(interval);
  }, []);

  const logout = () => {
    setAdminToken(null);
    onLogout();
  };

  const Stat = ({ label, value }) => (
    <div className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-5">
      <p className="text-sm text-[#9ca0b0]">{label}</p>
      <p className="mt-2 text-2xl font-bold text-white">
        {value ?? "—"}
      </p>
    </div>
  );

  return (
    <div className="min-h-screen p-5 md:p-8" style={{ background: "#0f1117" }}>
      <header className="mx-auto flex max-w-6xl items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">RescueWay Admin</h1>
          <p className="mt-1 text-sm text-[#9ca0b0]">
            Signed in as {username}
          </p>
        </div>




        <button
          onClick={logout}
          className="rounded-lg border border-[#343849] px-4 py-2 text-sm text-red-300 hover:bg-red-500/10"
        >
          Sign out
        </button>

      </header>

      <main className="mx-auto mt-8 max-w-6xl space-y-7">
        <section className="space-y-3">
          <div>
            <h2 className="text-lg font-semibold text-white">Operator controls</h2>
            <p className="text-sm text-[#9ca0b0]">
              Administrator-only controls for the shared cloud simulation
            </p>
          </div>
          <ControlBar
            lastEvent={lastEvent}
            connected={connected}
            readOnly={false}
            showGpsSimulator={true}
            showBenchmark={false}
          />
        </section>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-white">System overview</h2>
            <p className="text-sm text-[#9ca0b0]">
              Live cloud simulation status
            </p>
          </div>

          <button
            onClick={refreshData}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-500"
          >
            Refresh
          </button>
        </div>

        {error && (
          <div role="alert" className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300">
            {error}
          </div>
        )}

        <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="Backend status" value={health?.status} />
          <Stat label="Current traffic zone" value={health?.current_zone} />
          <Stat label="Simulation mode" value={simulation?.mode} />
          <Stat label="Traffic pattern" value={simulation?.traffic_pattern} />
        </section>



        <section className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-5">
          <h2 className="mb-4 text-lg font-semibold text-white">
            Simulation settings
          </h2>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Stat
              label="Green duration"
              value={simulation?.timings?.green_duration != null
                ? `${simulation.timings.green_duration}s`
                : "—"}
            />
            <Stat
              label="Yellow duration"
              value={simulation?.timings?.yellow_duration != null
                ? `${simulation.timings.yellow_duration}s`
                : "—"}
            />
            <Stat
              label="Emergency duration"
              value={simulation?.timings?.emergency_duration != null
                ? `${simulation.timings.emergency_duration}s`
                : "—"}
            />
          </div>
        </section>

        <section className="rounded-xl border border-[#2a2d3a] bg-[#1a1d27] p-5">
          <h2 className="mb-4 text-lg font-semibold text-white">
            Zone information
          </h2>
          <pre className="overflow-x-auto whitespace-pre-wrap text-xs text-[#c5c8d3]">
            {zones ? JSON.stringify(zones, null, 2) : "Loading zone information..."}
          </pre>
        </section>

        <p className="text-center text-xs text-[#6b7080]">
          RescueWay · Private Administrator Console
        </p>
      </main>
    </div>
  );
}
