// src/components/analytics/SimulationControls.jsx
import { useState, useEffect } from "react";
import {
  getSimulationStatus, setTrafficPattern, setArrivalRates,
  setTimings, resetSimulation
} from "../../api/rescueway";

const PATTERNS = [
  { id: "low",       name: "Low",       color: "bg-green-100 text-green-700 border-green-300",  desc: "Quiet roads" },
  { id: "medium",    name: "Medium",    color: "bg-yellow-100 text-yellow-700 border-yellow-300", desc: "Normal traffic" },
  { id: "high",      name: "High",      color: "bg-orange-100 text-orange-700 border-orange-300", desc: "Heavy congestion" },
  { id: "rush_hour", name: "Rush Hour", color: "bg-red-100 text-red-700 border-red-300",      desc: "Peak traffic" },
];

export default function SimulationControls() {
  const [status, setStatus]       = useState(null);
  const [activePattern, setActive] = useState("medium");
  const [rates, setRates]         = useState({ z1: 1.0, z2: 1.0, z3: 1.0, z4: 1.0 });
  const [green, setGreen]         = useState(5);
  const [yellow, setYellow]       = useState(2);
  const [emergency, setEmergency] = useState(15);
  const [feedback, setFeedback]   = useState("");

  useEffect(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadStatus = async () => {
    try {
      const data = await getSimulationStatus();
      setStatus(data);
      setActive(data.traffic_pattern);
      setGreen(data.timings.green_duration);
      setYellow(data.timings.yellow_duration);
      setEmergency(data.timings.emergency_duration);
    } catch (err) {
      console.error("Failed to load simulation status:", err);
    }
  };

  const showFeedback = (msg) => {
    setFeedback(msg);
    setTimeout(() => setFeedback(""), 3000);
  };

  const handlePattern = async (pattern) => {
    try {
      await setTrafficPattern(pattern);
      setActive(pattern);
      showFeedback(`Traffic pattern → ${pattern}`);
    } catch (err) {
      console.error(err);
    }
  };

  const handleRates = async () => {
    try {
      await setArrivalRates(rates.z1, rates.z2, rates.z3, rates.z4);
      showFeedback("Custom arrival rates applied");
    } catch (err) {
      console.error(err);
    }
  };

  const handleTimings = async () => {
    try {
      await setTimings(green, yellow, emergency);
      showFeedback("Signal timings updated");
    } catch (err) {
      console.error(err);
    }
  };

  const handleReset = async () => {
    try {
      await resetSimulation();
      await loadStatus();
      showFeedback("Simulation reset");
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-medium text-gray-700">Simulation controls</h2>
        {feedback && (
          <span className="text-xs px-3 py-1 rounded-full bg-blue-100 text-blue-700">
            {feedback}
          </span>
        )}
      </div>

      {/* Traffic Pattern Selector */}
      <div>
        <p className="text-sm text-gray-500 mb-2">Traffic pattern</p>
        <div className="grid grid-cols-4 gap-2">
          {PATTERNS.map(p => (
            <button
              key={p.id}
              onClick={() => handlePattern(p.id)}
              className={`px-3 py-2.5 rounded-lg text-sm font-medium border transition-all ${
                activePattern === p.id
                  ? p.color
                  : "bg-gray-50 text-gray-500 border-gray-200 hover:border-gray-300"
              }`}
            >
              <span className="block">{p.name}</span>
              <span className="block text-xs opacity-70 mt-0.5">{p.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Arrival Rates */}
      <div>
        <p className="text-sm text-gray-500 mb-2">
          Vehicle arrival rates
          <span className="text-xs text-gray-400 ml-1">(vehicles/second per zone)</span>
        </p>
        <div className="grid grid-cols-4 gap-3">
          {[1, 2, 3, 4].map(z => {
            const key = `z${z}`;
            return (
              <div key={z}>
                <label className="text-xs text-gray-400 block mb-1">Zone {z}</label>
                <input
                  type="range"
                  min="0"
                  max="4"
                  step="0.1"
                  value={rates[key]}
                  onChange={e => setRates({ ...rates, [key]: parseFloat(e.target.value) })}
                  className="w-full"
                />
                <p className="text-xs text-center text-gray-600 mt-1">{rates[key].toFixed(1)}</p>
              </div>
            );
          })}
        </div>
        <button
          onClick={handleRates}
          className="mt-2 px-4 py-1.5 text-xs bg-gray-800 text-white rounded-lg hover:bg-gray-900"
        >
          Apply rates
        </button>
      </div>

      {/* Signal Timings */}
      <div>
        <p className="text-sm text-gray-500 mb-2">Signal timings</p>
        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className="text-xs text-gray-400 block mb-1">Green (s)</label>
            <input
              type="number"
              min="2"
              max="30"
              value={green}
              onChange={e => setGreen(Number(e.target.value))}
              className="w-full border border-gray-200 rounded-lg px-3 py-1.5 text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Yellow (s)</label>
            <input
              type="number"
              min="1"
              max="5"
              value={yellow}
              onChange={e => setYellow(Number(e.target.value))}
              className="w-full border border-gray-200 rounded-lg px-3 py-1.5 text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Emergency (s)</label>
            <input
              type="number"
              min="5"
              max="60"
              value={emergency}
              onChange={e => setEmergency(Number(e.target.value))}
              className="w-full border border-gray-200 rounded-lg px-3 py-1.5 text-sm"
            />
          </div>
        </div>
        <button
          onClick={handleTimings}
          className="mt-2 px-4 py-1.5 text-xs bg-gray-800 text-white rounded-lg hover:bg-gray-900"
        >
          Update timings
        </button>
      </div>

      {/* System Info and Reset */}
      {status && (
        <div className="flex items-center justify-between pt-2 border-t border-gray-100">
          <div className="flex gap-4 text-xs text-gray-400">
            <span>Step: {status.step}</span>
            <span>Mode: {status.mode}</span>
            <span>Active: Zone {status.current_zone}</span>
            <span>Arrived: {status.system_metrics?.total_vehicles_arrived || 0}</span>
            <span>Cleared: {status.system_metrics?.total_vehicles_cleared || 0}</span>
          </div>
          <button
            onClick={handleReset}
            className="px-3 py-1 text-xs text-red-600 border border-red-200 rounded-lg hover:bg-red-50"
          >
            Reset simulation
          </button>
        </div>
      )}
    </div>
  );
}