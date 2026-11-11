// src/components/ZoneSelector.jsx
const TRAFFIC_LEVELS = ["low", "medium", "high"];

const TRAFFIC_COLORS = {
  low:    "bg-green-100 text-green-800 border-green-300",
  medium: "bg-yellow-100 text-yellow-800 border-yellow-300",
  high:   "bg-red-100 text-red-800 border-red-300"
};

export default function ZoneSelector({ traffic, onTrafficChange, onZoneSelect, loading }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-5">
      <h2 className="text-base font-medium text-gray-700">
        Ambulance location
      </h2>

      {/* Traffic level selector */}
      <div>
        <p className="text-sm text-gray-500 mb-2">Traffic density</p>
        <div className="flex gap-2">
          {TRAFFIC_LEVELS.map(t => (
            <button
              key={t}
              onClick={() => onTrafficChange(t)}
              className={`px-4 py-1.5 rounded-full text-sm border font-medium transition-all ${
                traffic === t
                  ? TRAFFIC_COLORS[t]
                  : "bg-gray-50 text-gray-500 border-gray-200 hover:border-gray-300"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Zone buttons */}
      <div>
        <p className="text-sm text-gray-500 mb-2">Select zone</p>
        <div className="grid grid-cols-2 gap-3">
          {[1, 2, 3, 4].map(zone => (
            <button
              key={zone}
              onClick={() => onZoneSelect(zone)}
              disabled={loading}
              className="p-5 rounded-xl border-2 border-gray-200
                         hover:border-blue-400 hover:bg-blue-50
                         active:scale-95 transition-all
                         text-lg font-semibold text-gray-700
                         disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loading === zone ? (
                <span className="text-blue-500">Sending...</span>
              ) : (
                `Zone ${zone}`
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}