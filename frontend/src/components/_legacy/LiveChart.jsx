// src/components/LiveChart.jsx
export default function LiveChart({ history }) {
  if (history.length === 0) return null;

  const max     = Math.max(...history.map(h => h.reward), 30);
  const min     = Math.min(...history.map(h => h.reward), -25);
  const range   = max - min || 1;
  const W       = 600;
  const H       = 120;
  const PAD     = 20;
  const plotW   = W - PAD * 2;
  const plotH   = H - PAD * 2;

  const points  = history.slice(-30).map((h, i, arr) => {
    const x = PAD + (i / Math.max(arr.length - 1, 1)) * plotW;
    const y = PAD + plotH - ((h.reward - min) / range) * plotH;
    return `${x},${y}`;
  });

  const zeroY   = PAD + plotH - ((0 - min) / range) * plotH;

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-base font-medium text-gray-700">Live reward chart</h2>
        <span className="text-xs text-gray-400">Last 30 steps</span>
      </div>

      <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="overflow-visible">
        {/* Zero line */}
        <line
          x1={PAD} y1={zeroY} x2={W - PAD} y2={zeroY}
          stroke="#e5e7eb" strokeWidth="1" strokeDasharray="4 4"
        />
        <text x={PAD - 4} y={zeroY + 4} fontSize="9" fill="#9ca3af" textAnchor="end">0</text>

        {/* Max line */}
        <text x={PAD - 4} y={PAD + 4} fontSize="9" fill="#9ca3af" textAnchor="end">{max}</text>

        {/* Min line */}
        <text x={PAD - 4} y={H - PAD + 4} fontSize="9" fill="#9ca3af" textAnchor="end">{min}</text>

        {/* Reward line */}
        {points.length > 1 && (
          <polyline
            points={points.join(" ")}
            fill="none"
            stroke="#3b82f6"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}

        {/* Dots colored by reward */}
        {history.slice(-30).map((h, i, arr) => {
          const x = PAD + (i / Math.max(arr.length - 1, 1)) * plotW;
          const y = PAD + plotH - ((h.reward - min) / range) * plotH;
          const color = h.reward >= 20 ? "#22c55e" : h.reward >= 0 ? "#f59e0b" : "#ef4444";
          return (
            <circle key={i} cx={x} cy={y} r="4" fill={color}>
              <title>Step {h.step}: {h.reward > 0 ? "+" : ""}{h.reward}</title>
            </circle>
          );
        })}
      </svg>

      {/* Legend */}
      <div className="flex gap-4 mt-2">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-green-400" />
          <span className="text-xs text-gray-500">Correct (≥+20)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-yellow-400" />
          <span className="text-xs text-gray-500">Partial (0 to +20)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-red-400" />
          <span className="text-xs text-gray-500">Wrong (&lt;0)</span>
        </div>
      </div>
    </div>
  );
}