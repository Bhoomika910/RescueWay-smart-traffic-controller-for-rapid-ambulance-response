// src/components/RLDecision.jsx
export default function RLDecision({ result }) {
  if (!result) return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <h2 className="text-base font-medium text-gray-700 mb-2">RL decision</h2>
      <p className="text-sm text-gray-400">Select a zone to see the RL agent decision</p>
    </div>
  );

  const rewardColor = result.reward >= 20
    ? "text-green-600"
    : result.reward >= 0
    ? "text-yellow-600"
    : "text-red-500";

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-4">
      <h2 className="text-base font-medium text-gray-700">RL decision</h2>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-50 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">Action taken</p>
          <p className="text-sm font-medium text-gray-800">{result.action_name}</p>
        </div>
        <div className="bg-gray-50 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">Reward</p>
          <p className={`text-xl font-semibold ${rewardColor}`}>
            {result.reward > 0 ? "+" : ""}{result.reward}
          </p>
        </div>
        <div className="bg-gray-50 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">Active signal</p>
          <p className="text-sm font-medium text-gray-800">Zone {result.active_signal}</p>
        </div>
        <div className="bg-gray-50 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">Duration</p>
          <p className="text-sm font-medium text-gray-800">{result.duration_s}s</p>
        </div>
      </div>

      <div className="bg-gray-50 rounded-lg p-3">
        <p className="text-xs text-gray-400 mb-1">Arduino</p>
        <p className="text-xs font-mono text-gray-600">{result.arduino_ack}</p>
      </div>
    </div>
  );
}