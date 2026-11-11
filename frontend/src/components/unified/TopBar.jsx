// src/components/unified/TopBar.jsx
import { useState, useEffect } from "react";

/** @typedef {import('../../types').LiveEvent} LiveEvent */

/**
 * @param {Object} props
 * @param {boolean} props.connected - Whether the live WebSocket is currently connected.
 * @param {LiveEvent} props.lastEvent - Most recent live event pushed over the WebSocket.
 */
export default function TopBar({ connected, lastEvent }) {
  const [time, setTime] = useState(new Date());
  const mode = lastEvent?.mode || "normal";
  const step = lastEvent?.step || 0;

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className={`px-5 py-2 flex items-center justify-between border-b transition-colors duration-300 ${
      mode === "emergency"
        ? "bg-red-900/50 border-red-800"
        : "bg-[#1a1d27] border-[#2a2d3a]"
    }`}>
      <div className="flex items-center gap-3">
        <div>
          <h1 className="text-sm font-bold text-white tracking-wide">RescueWay</h1>
          <p className="text-[10px] text-[#6b7080]">AI-Powered Emergency Vehicle Priority</p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <StatusBadge
          label="System"
          value={connected ? "Live" : "Offline"}
          color={connected ? "green" : "red"}
          dot
        />
        <StatusBadge
          label="Mode"
          value={mode === "emergency" ? `Emergency Z${lastEvent?.emergency_zone}` : "Normal"}
          color={mode === "emergency" ? "red" : "green"}
        />
        <StatusBadge
          label="Controller"
          value="DQN"
          color="blue"
        />
        <StatusBadge
          label="Arduino"
          value={lastEvent?.arduino_ack?.startsWith("OK") ? "HW" : "SIM"}
          color={lastEvent?.arduino_ack?.startsWith("OK") ? "green" : "gray"}
          dot
        />
        <StatusBadge
          label="Step"
          value={step}
          color="gray"
        />
      </div>

      <div className="text-right">
        <p className="text-lg font-mono text-white font-bold">
          {time.toLocaleTimeString()}
        </p>
        <p className="text-[10px] text-[#6b7080]">
          {time.toLocaleDateString()}
        </p>
      </div>
    </div>
  );
}

function StatusBadge({ label, value, color, dot }) {
  const colors = {
    green: "text-emerald-400",
    red: "text-red-400",
    blue: "text-blue-400",
    gray: "text-gray-400",
  };
  return (
    <div className="bg-[#0f1117] border border-[#2a2d3a] rounded-lg px-3 py-1.5 text-center min-w-[90px]">
      <p className="text-[9px] uppercase tracking-wider text-[#6b7080]">{label}</p>
      <p className={`text-xs font-semibold ${colors[color]} flex items-center justify-center gap-1`}>
        {dot && <span className={`w-1.5 h-1.5 rounded-full bg-current`} />}
        {value}
      </p>
    </div>
  );
}