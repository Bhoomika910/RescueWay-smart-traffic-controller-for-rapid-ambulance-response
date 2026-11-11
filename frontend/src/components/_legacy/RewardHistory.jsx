// src/components/RewardHistory.jsx
export default function RewardHistory({ history }) {
  if (history.length === 0) return null;

  const total = history.reduce((a, b) => a + b, 0);
  const avg   = (total / history.length).toFixed(1);
  const last  = history[history.length - 1];

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <h2 className="text-base font-medium text-gray-700 mb-4">Reward history</h2>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-gray-50 rounded-lg p-3 text-center">
          <p className="text-xs text-gray-400">Steps</p>
          <p className="text-lg font-semibold text-gray-800">{history.length}</p>
        </div>
        <div className="bg-gray-50 rounded-lg p-3 text-center">
          <p className="text-xs text-gray-400">Avg reward</p>
          <p className={`text-lg font-semibold ${avg >= 0 ? "text-green-600" : "text-red-500"}`}>
            {avg > 0 ? "+" : ""}{avg}
          </p>
        </div>
        <div className="bg-gray-50 rounded-lg p-3 text-center">
          <p className="text-xs text-gray-400">Last reward</p>
          <p className={`text-lg font-semibold ${last >= 0 ? "text-green-600" : "text-red-500"}`}>
            {last > 0 ? "+" : ""}{last}
          </p>
        </div>
      </div>

      {/* Reward bar chart */}
      <div className="flex items-end gap-1 h-16">
        {history.slice(-20).map((r, i) => {
          const height = Math.max(4, Math.min(64, ((r + 25) / 50) * 64));
          const color  = r >= 20 ? "bg-green-400" : r >= 0 ? "bg-yellow-400" : "bg-red-400";
          return (
            <div
              key={i}
              title={`Reward: ${r}`}
              className={`flex-1 rounded-sm ${color} transition-all duration-300`}
              style={{ height: `${height}px` }}
            />
          );
        })}
      </div>
      <p className="text-xs text-gray-400 mt-1 text-right">Last 20 steps</p>
    </div>
  );
}