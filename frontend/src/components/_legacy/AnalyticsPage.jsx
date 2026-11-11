// src/components/analytics/AnalyticsPage.jsx
import LiveMetricsPanel    from "./LiveMetricsPanel";
import QueueVisualization  from "./QueueVisualization";
import ComparisonChart     from "../analytics/ComparisonChart";
import SimulationControls  from "./SimulationControls";
import LogViewer           from "../analytics/LogViewer";

export default function AnalyticsPage({ lastEvent, connected }) {
  return (
    <div className="space-y-4">
      <SimulationControls />
      <LiveMetricsPanel lastEvent={lastEvent} />
      <QueueVisualization lastEvent={lastEvent} />
      <ComparisonChart />
      <LogViewer />
    </div>
  );
}