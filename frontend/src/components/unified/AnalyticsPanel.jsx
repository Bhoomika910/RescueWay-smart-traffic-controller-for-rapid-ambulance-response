// src/components/unified/AnalyticsPanel.jsx
// Pure WebSocket-driven — no HTTP polling
import { queueLevel, QUEUE_MAX_CAPACITY } from "../../utils/queueStatus";

function Metric({ label, value, unit, color }) {
  return (
    <div className="bg-[#141620] rounded-lg p-2.5 border border-[#2a2d3a]">
      <p className="text-[9px] uppercase tracking-wider text-[#6b7080] mb-1">{label}</p>
      <div className="flex items-baseline gap-1">
        <span className={`text-lg font-bold ${color || "text-white"}`}>{value}</span>
        {unit && <span className="text-[10px] text-[#6b7080]">{unit}</span>}
      </div>
    </div>
  );
}

/** @typedef {import('../../types').LiveEvent} LiveEvent */

/**
 * @param {Object} props
 * @param {LiveEvent} props.lastEvent - Most recent live event pushed over the WebSocket.
 */
export default function AnalyticsPanel({ lastEvent }) {
  const lm = lastEvent?.live_metrics || {};
  const ql = lastEvent?.queue_lengths || {};
  const ss = lastEvent?.signal_states || {};
  const mode = lastEvent?.mode || "normal";
  const active = lastEvent?.active_signal || 1;
  const emZ = lastEvent?.emergency_zone || 0;
  const phase = lastEvent?.phase || "green";

  const qColor = (n) => {
    const level = queueLevel(n);
    return level === "danger" ? "text-red-400" : level === "warning" ? "text-yellow-400" : "text-emerald-400";
  };
  const barColor = (z, n) => {
    if (z === emZ && mode === "emergency") return "bg-red-500";
    if (z === active && (phase === "green" || phase === "emergency_hold")) return "bg-emerald-500";
    if (n > 30) return "bg-orange-500";
    return "bg-blue-500";
  };

  return (
    <div className="h-full flex flex-col gap-3">
      {/* Metrics — all from WebSocket live_metrics */}
      <div className="bg-[#1e2130] rounded-xl border border-[#2a2d3a] p-3">
        <p className="text-[9px] uppercase tracking-wider text-[#6b7080] mb-2 font-semibold">Performance</p>
        <div className="grid grid-cols-3 gap-2">
          <Metric label="Avg Wait Time" value={lm.avg_wait_time ?? 0} unit="s"
            color={(lm.avg_wait_time ?? 0) > 10 ? "text-red-400" : "text-emerald-400"} />
          <Metric label="Avg Queue" value={lm.avg_queue_length ?? 0} unit="veh"
            color={(lm.avg_queue_length ?? 0) > 15 ? "text-red-400" : "text-emerald-400"} />
          <Metric label="Total Cleared" value={lm.total_vehicles_cleared ?? 0} unit="veh" color="text-blue-400" />
          {/* clearance_rate = vehicles cleared / vehicles arrived, a ratio (0-100%),
              not a raw vehicles/time throughput figure */}
          <Metric label="Clearance Rate" value={`${lm.clearance_rate ?? 0}%`}
            color={(lm.clearance_rate ?? 0) > 50 ? "text-emerald-400" : "text-yellow-400"} />
          <Metric label="Emergency Accuracy" value={`${lm.emergency_accuracy ?? 0}%`}
            color={(lm.emergency_accuracy ?? 0) > 70 ? "text-emerald-400" : "text-yellow-400"} />
          {/* Jain's fairness index over per-zone green time: always in [0,1],
              1.0 = perfectly fair. Shown as a raw ratio, not a percentage. */}
          <Metric label="Fairness Index" value={lm.fairness_index ?? 1}
            color={(lm.fairness_index ?? 1) > 0.8 ? "text-emerald-400" : "text-yellow-400"} />
        </div>
        {(lm.avg_emergency_response_time ?? 0) > 0 && (
          <div className="mt-2">
            {/* True request-to-green latency per ambulance (EmergencyManager),
                including any queueing delay from ETA-based preemption. */}
            <Metric label="Emergency Response Time" value={lm.avg_emergency_response_time} unit="s"
              color={(lm.avg_emergency_response_time ?? 0) < 5 ? "text-emerald-400" : "text-yellow-400"} />
          </div>
        )}
      </div>

      {/* Queue Lengths */}
      <div className="bg-[#1e2130] rounded-xl border border-[#2a2d3a] p-3 flex-1">
        <p className="text-[9px] uppercase tracking-wider text-[#6b7080] mb-2 font-semibold">Queue Lengths</p>
        <div className="space-y-2.5">
          {[1, 2, 3, 4].map(z => {
            const n = ql[z] || 0;
            const pct = Math.min((n / QUEUE_MAX_CAPACITY) * 100, 100);
            const isActive = z === active;
            const isEm = z === emZ && mode === "emergency";
            const zoneSignal = ss[z] || "red";
            return (
              <div key={z}>
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs font-medium text-[#9ca0b0]">Z{z}</span>
                    {isActive && <span className={`w-2 h-2 rounded-full ${
                      zoneSignal === "green" ? "bg-emerald-400" :
                      zoneSignal === "yellow" ? "bg-yellow-400" : "bg-red-400"
                    }`} />}
                    {isEm && <span className="text-[10px] text-red-400">🚑</span>}
                  </div>
                  <span className={`text-xs font-bold ${qColor(n)}`}>{n}</span>
                </div>
                <div className="w-full bg-[#0f1117] rounded-full h-2">
                  <div className={`h-2 rounded-full transition-all duration-500 ${barColor(z, n)}`}
                    style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
        </div>
        <p className="text-[9px] text-[#6b7080] mt-2">
          Total: {Object.values(ql).reduce((a, b) => a + b, 0)} vehicles
        </p>
      </div>

      {/* Signals + RL */}
      <div className="bg-[#1e2130] rounded-xl border border-[#2a2d3a] p-3">
        <div className="grid grid-cols-4 gap-2 mb-2">
          {[1, 2, 3, 4].map(z => {
            const sc = ss[z] || "red";
            const c = { red: "bg-red-500", yellow: "bg-yellow-400", green: "bg-emerald-500" };
            return (
              <div key={z} className="text-center">
                <p className="text-[9px] text-[#6b7080] mb-1">Z{z}</p>
                <div className={`w-3.5 h-3.5 rounded-full mx-auto ${c[sc] || "bg-gray-600"} ${
                  sc === "green" ? "shadow-lg shadow-emerald-500/30" : ""
                }`} />
                <p className="text-[9px] text-[#9ca0b0] mt-1 capitalize">{sc}</p>
              </div>
            );
          })}
        </div>

        {lastEvent?.action_name && (
          <div className="pt-2 border-t border-[#2a2d3a] flex items-center justify-between">
            <div>
              <p className="text-[9px] text-[#6b7080]">RL Decision</p>
              <p className="text-xs font-medium text-white">{lastEvent.action_name}</p>
            </div>
            <p className={`text-sm font-bold ${
              (lastEvent.reward ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"
            }`}>
              {(lastEvent.reward ?? 0) > 0 ? "+" : ""}{lastEvent.reward ?? 0}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}