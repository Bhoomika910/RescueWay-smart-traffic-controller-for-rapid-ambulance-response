// src/components/SignalDisplay.jsx
function LED({ color, active }) {
  const colors = {
    red:    active ? "bg-red-500 shadow-red-300"    : "bg-gray-200",
    yellow: active ? "bg-yellow-400 shadow-yellow-200" : "bg-gray-200",
    green:  active ? "bg-green-500 shadow-green-300"  : "bg-gray-200",
  };
  return (
    <div className={`w-8 h-8 rounded-full transition-all duration-300 ${colors[color]} ${active ? "shadow-lg" : ""}`} />
  );
}

export default function SignalDisplay({ activeZone, activeSignal }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <h2 className="text-base font-medium text-gray-700 mb-4">
        Signal status
      </h2>
      <div className="grid grid-cols-4 gap-3">
        {[1, 2, 3, 4].map(zone => (
          <div
            key={zone}
            className={`rounded-xl p-3 border-2 transition-all duration-300 ${
              zone === activeZone
                ? "border-blue-400 bg-blue-50"
                : "border-gray-100 bg-gray-50"
            }`}
          >
            <p className={`text-xs font-medium mb-3 text-center ${
              zone === activeZone ? "text-blue-600" : "text-gray-400"
            }`}>
              Zone {zone}
              {zone === activeZone && (
                <span className="block text-blue-400">ambulance</span>
              )}
            </p>
            <div className="flex flex-col items-center gap-1.5">
              <LED color="red"    active={zone !== activeSignal} />
              <LED color="yellow" active={false} />
              <LED color="green"  active={zone === activeSignal} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}