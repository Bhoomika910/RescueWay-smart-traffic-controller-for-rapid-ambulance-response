// src/components/analytics/QueueVisualization.jsx
export default function QueueVisualization({ lastEvent }) {
  const queues = lastEvent?.queue_lengths || {};
  const maxCapacity = 50;

  const zones = [1, 2, 3, 4].map(z => ({
    zone: z,
    length: queues[z] || 0,
    pct: ((queues[z] || 0) / maxCapacity) * 100,
    isActive: z === lastEvent?.active_signal,
    isEmergency: z === lastEvent?.emergency_zone,
  }));

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <h2 className="text-base font-medium text-gray-700 mb-4">Queue lengths</h2>

      <div className="space-y-3">
        {zones.map(z => {
          const barColor = z.isEmergency
            ? "bg-red-400"
            : z.isActive
            ? "bg-green-400"
            : z.pct > 60
            ? "bg-orange-400"
            : "bg-blue-400";

          const bgColor = z.isEmergency
            ? "bg-red-50 border-red-200"
            : z.isActive
            ? "bg-green-50 border-green-200"
            : "bg-gray-50 border-gray-100";

          return (
            <div key={z.zone} className={`rounded-lg border p-3 ${bgColor}`}>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-gray-700">
                    Zone {z.zone}
                  </span>
                  {z.isActive && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700">
                      GREEN
                    </span>
                  )}
                  {z.isEmergency && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-red-100 text-red-700">
                      AMBULANCE
                    </span>
                  )}
                </div>
                <span className="text-sm font-semibold text-gray-800">
                  {z.length} / {maxCapacity}
                </span>
              </div>

              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className={`h-3 rounded-full transition-all duration-500 ${barColor}`}
                  style={{ width: `${Math.min(z.pct, 100)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-3 flex gap-4 text-xs text-gray-400">
        <span>Total: {zones.reduce((a, z) => a + z.length, 0)} vehicles</span>
        <span>Capacity: {maxCapacity * 4} max</span>
      </div>
    </div>
  );
}