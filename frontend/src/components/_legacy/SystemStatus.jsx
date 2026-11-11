// src/components/SystemStatus.jsx
export default function SystemStatus({ connected, lastEvent }) {
  const mode     = lastEvent?.mode || "normal";
  const isEmerg  = mode === "emergency";

  return (
    <div className={`rounded-xl border p-4 transition-all duration-500 ${
      isEmerg
        ? "bg-red-50 border-red-300"
        : "bg-white border-gray-200"
    }`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* Connection status */}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${connected ? "bg-green-400" : "bg-red-400"}`} />
            <span className="text-sm text-gray-500">
              {connected ? "WebSocket live" : "Reconnecting..."}
            </span>
          </div>

          {/* Mode badge */}
          <div className={`px-3 py-1 rounded-full text-xs font-medium ${
            isEmerg
              ? "bg-red-100 text-red-700 border border-red-300"
              : "bg-green-100 text-green-700 border border-green-300"
          }`}>
            {isEmerg
              ? `EMERGENCY — Zone ${lastEvent?.emergency_zone}`
              : "NORMAL CYCLE"
            }
          </div>

          {/* Step counter */}
          {lastEvent && (
            <span className="text-sm text-gray-500">
              Step {lastEvent.step}
            </span>
          )}
        </div>

        {/* Arduino status */}
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-400" />
          <span className="text-sm text-gray-500">Arduino COM20</span>
        </div>
      </div>

      {/* Emergency info bar */}
      {isEmerg && (
        <div className="mt-3 flex items-center gap-2">
          <div className="w-full bg-red-200 rounded-full h-1.5">
            <div className="bg-red-500 h-1.5 rounded-full animate-pulse" style={{ width: "100%" }} />
          </div>
          <span className="text-xs text-red-600 whitespace-nowrap">
            Priority active
          </span>
        </div>
      )}
    </div>
  );
}