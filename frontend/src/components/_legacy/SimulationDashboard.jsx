// src/components/simulation/SimulationDashboard.jsx
import IntersectionView from "../simulation/IntersectionView";
import LiveMetricsPanel from "./LiveMetricsPanel";
import QueueVisualization from "./QueueVisualization";
import SimulationControls from "./SimulationControls";

export default function SimulationDashboard({ lastEvent, connected }) {
  return (
    <div className="space-y-4">
      {/* Main split view: animation + metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left: Live intersection */}
        <IntersectionView lastEvent={lastEvent} />

        {/* Right: Live metrics + queues */}
        <div className="space-y-4">
          <LiveMetricsPanel lastEvent={lastEvent} />
          <QueueVisualization lastEvent={lastEvent} />
        </div>
      </div>

      {/* Simulation controls (full width below) */}
      <SimulationControls />
    </div>
  );
}