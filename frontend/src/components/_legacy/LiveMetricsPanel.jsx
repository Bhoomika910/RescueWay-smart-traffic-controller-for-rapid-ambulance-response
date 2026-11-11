// src/components/analytics/LiveMetricsPanel.jsx
import { useState, useEffect } from "react";
import { getLiveMetrics } from "../../api/rescueway";

function MetricCard({ label, value, unit, color, subtitle }) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <p className="text-xs text-gray-400 mb-1">{label}</p>
      <p className={`text-2xl font-semibold ${color || "text-gray-800"}`}>
        {value}
        {unit && <span className="text-sm text-gray-400 ml-1">{unit}</span>}
      </p>
      {subtitle && <p className="text-xs text-gray-400 mt-1">{subtitle}</p>}
    </div>
  );
}

export default function LiveMetricsPanel({ lastEvent }) {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const data = await getLiveMetrics();
        setMetrics(data);
      } catch (err) {
        console.error("Failed to fetch metrics:", err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  if (!metrics) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <p className="text-sm text-gray-400">Loading metrics...</p>
      </div>
    );
  }

  const queueColor = metrics.avg_queue_length > 15
    ? "text-red-500" : metrics.avg_queue_length > 8
    ? "text-yellow-600" : "text-green-600";

  const congestionColor = metrics.avg_congestion > 0.5
    ? "text-red-500" : metrics.avg_congestion > 0.25
    ? "text-yellow-600" : "text-green-600";

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-medium text-gray-700">Live metrics</h2>
        <span className="text-xs text-gray-400">
          Step {metrics.step} · Uptime {Math.round(metrics.uptime_seconds)}s
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          label="Avg reward"
          value={metrics.avg_reward > 0 ? "+" + metrics.avg_reward : metrics.avg_reward}
          color={metrics.avg_reward >= 0 ? "text-green-600" : "text-red-500"}
        />
        <MetricCard
          label="Avg queue length"
          value={metrics.avg_queue_length}
          unit="vehicles"
          color={queueColor}
        />
        <MetricCard
          label="Vehicles cleared"
          value={metrics.total_vehicles_cleared}
          subtitle={`${metrics.clearance_rate}% clearance rate`}
          color="text-blue-600"
        />
        <MetricCard
          label="Congestion"
          value={`${(metrics.avg_congestion * 100).toFixed(1)}%`}
          color={congestionColor}
        />
        <MetricCard
          label="Emergency accuracy"
          value={`${metrics.emergency_accuracy}%`}
          subtitle={`${metrics.emergency_events} events`}
          color={metrics.emergency_accuracy > 70 ? "text-green-600" : "text-yellow-600"}
        />
        <MetricCard
          label="Fairness index"
          value={metrics.fairness_index}
          subtitle="1.0 = perfectly fair"
          color={metrics.fairness_index > 0.8 ? "text-green-600" : "text-yellow-600"}
        />
        <MetricCard
          label="Signal switches"
          value={metrics.total_switches}
          subtitle={`${metrics.switches_per_step} per step`}
        />
        <MetricCard
          label="Avg wait time"
          value={metrics.avg_wait_time}
          unit="s"
          color={metrics.avg_wait_time > 10 ? "text-red-500" : "text-green-600"}
        />
      </div>
    </div>
  );
}