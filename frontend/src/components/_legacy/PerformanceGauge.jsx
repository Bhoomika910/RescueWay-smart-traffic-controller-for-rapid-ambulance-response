// src/components/analytics/PerformanceGauge.jsx
export default function PerformanceGauge({ label, value, max, unit, thresholds }) {
  const pct = Math.min((value / max) * 100, 100);

  let color = "text-green-600";
  let barColor = "bg-green-400";

  if (thresholds) {
    if (value > thresholds.danger) {
      color = "text-red-500";
      barColor = "bg-red-400";
    } else if (value > thresholds.warning) {
      color = "text-yellow-600";
      barColor = "bg-yellow-400";
    }
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs text-gray-400">{label}</p>
        <p className={`text-lg font-semibold ${color}`}>
          {typeof value === "number" ? value.toFixed(1) : value}
          {unit && <span className="text-xs text-gray-400 ml-1">{unit}</span>}
        </p>
      </div>
      <div className="w-full bg-gray-100 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-700 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}