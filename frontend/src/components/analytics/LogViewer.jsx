// src/components/analytics/LogViewer.jsx
import { useState, useEffect, useCallback } from "react";
import { getRecentLogs, getLogSummary, exportLogs } from "../../api/rescueway";

const LEVEL_COLORS = {
  DEBUG:   "text-[#6b7080]",
  INFO:    "text-blue-400",
  WARNING: "text-yellow-400",
  ERROR:   "text-red-400",
};

const CATEGORY_COLORS = {
  decision:  "bg-blue-900/40 text-blue-300",
  signal:    "bg-emerald-900/40 text-emerald-300",
  emergency: "bg-red-900/40 text-red-300",
  arduino:   "bg-purple-900/40 text-purple-300",
  traffic:   "bg-yellow-900/40 text-yellow-300",
  system:    "bg-[#252838] text-[#9ca0b0]",
};

export default function LogViewer() {
  const [logs, setLogs]         = useState([]);
  const [summary, setSummary]   = useState(null);
  const [filter, setFilter]     = useState({ level: "", category: "" });
  const [autoRefresh, setAuto]  = useState(true);
  const [exporting, setExporting] = useState(false);

  const loadLogs = useCallback(async () => {
    try {
      const data = await getRecentLogs(
        100,
        filter.level || null,
        filter.category || null
      );
      setLogs(data.logs || []);
      const sum = await getLogSummary();
      setSummary(sum);
    } catch (err) {
      console.error("Failed to load logs:", err);
    }
  }, [filter.level, filter.category]);

  useEffect(() => {
    loadLogs();
    if (autoRefresh) {
      const interval = setInterval(loadLogs, 3000);
      return () => clearInterval(interval);
    }
  }, [loadLogs, autoRefresh]);

  const handleExport = async () => {
    setExporting(true);
    try {
      const result = await exportLogs();
      alert(`Logs exported to:\n${result.filepath}`);
    } catch (err) {
      console.error("Export failed:", err);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="bg-[#1e2130] rounded-xl border border-[#2a2d3a] p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-medium text-white">System logs</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setAuto(!autoRefresh)}
            className={`px-3 py-1 text-xs rounded-lg border ${
              autoRefresh
                ? "bg-emerald-900/40 text-emerald-300 border-emerald-800/50"
                : "bg-[#141620] text-[#6b7080] border-[#2a2d3a]"
            }`}
          >
            {autoRefresh ? "Auto-refresh ON" : "Auto-refresh OFF"}
          </button>
          <button
            onClick={handleExport}
            disabled={exporting}
            className="px-3 py-1 text-xs bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {exporting ? "Exporting..." : "Export logs"}
          </button>
        </div>
      </div>

      {/* Summary counters */}
      {summary && (
        <div className="grid grid-cols-6 gap-2">
          {Object.entries(summary.counts).map(([key, val]) => (
            <div key={key} className="bg-[#141620] border border-[#2a2d3a] rounded-lg p-2 text-center">
              <p className="text-lg font-semibold text-white">{val}</p>
              <p className="text-xs text-[#6b7080]">{key}</p>
            </div>
          ))}
        </div>
      )}

      {/* Filters */}
      <div className="flex gap-2">
        <select
          value={filter.level}
          onChange={e => setFilter({ ...filter, level: e.target.value })}
          className="text-xs bg-[#141620] border border-[#2a2d3a] text-white rounded-lg px-2 py-1"
        >
          <option value="">All levels</option>
          <option value="DEBUG">DEBUG</option>
          <option value="INFO">INFO</option>
          <option value="WARNING">WARNING</option>
          <option value="ERROR">ERROR</option>
        </select>
        <select
          value={filter.category}
          onChange={e => setFilter({ ...filter, category: e.target.value })}
          className="text-xs bg-[#141620] border border-[#2a2d3a] text-white rounded-lg px-2 py-1"
        >
          <option value="">All categories</option>
          <option value="decision">Decisions</option>
          <option value="signal">Signals</option>
          <option value="emergency">Emergencies</option>
          <option value="arduino">Arduino</option>
          <option value="traffic">Traffic</option>
          <option value="system">System</option>
        </select>
        <button
          onClick={() => setFilter({ level: "", category: "" })}
          className="text-xs text-[#6b7080] hover:text-[#9ca0b0]"
        >
          Clear filters
        </button>
      </div>

      {/* Log entries */}
      <div className="max-h-80 overflow-y-auto space-y-1 font-mono text-xs">
        {logs.length === 0 && (
          <p className="text-sm text-[#6b7080] py-4 text-center">No logs matching filter</p>
        )}
        {[...logs].reverse().map((log, i) => (
          <div
            key={i}
            className={`flex items-start gap-2 px-2 py-1 rounded ${
              log.level === "ERROR" ? "bg-red-950/30" :
              log.level === "WARNING" ? "bg-yellow-900/20" : ""
            }`}
          >
            <span className="text-[#6b7080] whitespace-nowrap">
              {log.timestamp?.split("T")[1]?.split(".")[0] || ""}
            </span>
            <span className={`font-medium w-14 ${LEVEL_COLORS[log.level] || "text-[#9ca0b0]"}`}>
              {log.level}
            </span>
            <span className={`px-1.5 py-0.5 rounded text-xs ${
              CATEGORY_COLORS[log.category] || "bg-[#252838] text-[#9ca0b0]"
            }`}>
              {log.category}
            </span>
            <span className="text-[#9ca0b0] flex-1">{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
