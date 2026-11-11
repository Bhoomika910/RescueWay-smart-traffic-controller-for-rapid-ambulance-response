// src/components/unified/ControlBar.jsx
import { useState } from "react";
import { selectZone, setTrafficPattern, setTimings, resetSimulation, setArrivalRates } from "../../api/rescueway";

const PATTERNS = [
  { id: "low",       label: "Low",       c: "bg-emerald-600" },
  { id: "medium",    label: "Medium",    c: "bg-yellow-600" },
  { id: "high",      label: "High",      c: "bg-orange-600" },
  { id: "rush_hour", label: "Rush Hour", c: "bg-red-600" },
];

/** @typedef {import('../../types').LiveEvent} LiveEvent */

/**
 * @param {Object} props
 * @param {LiveEvent} props.lastEvent - Most recent live event pushed over the WebSocket.
 * @param {boolean} props.connected - Whether the live WebSocket is currently connected.
 */
export default function ControlBar({ lastEvent, connected }) {
  // Derived directly from lastEvent.traffic (same source App.jsx's footer reads)
  // rather than local state, so it can never silently drift from the live signal
  // if the pattern changes from a source other than this component's own button.
  const activePattern = lastEvent?.traffic || "medium";
  const [loading, setLoading]      = useState(null);
  const [rates, setRates]          = useState({ z1: 1.0, z2: 1.0, z3: 1.0, z4: 1.0 });
  const [green, setGreen]          = useState(5);
  const [yellow, setYellow]        = useState(2);
  const [emDur, setEmDur]          = useState(15);
  const [feedback, setFeedback]    = useState("");

  const mode = lastEvent?.mode || "normal";

  const flash = (msg) => { setFeedback(msg); setTimeout(()=>setFeedback(""),2500); };

  const handleGPS = async (zone) => {
    setLoading(zone);
    try { await selectZone(zone, activePattern); flash(`GPS → Zone ${zone}`); }
    catch(e) { console.error(e); }
    finally { setLoading(null); }
  };

  const handlePattern = async (p) => {
    try { await setTrafficPattern(p); flash(`Pattern → ${p}`); }
    catch(e) { console.error(e); }
  };

  const handleRates = async () => {
    try {
      const validated = {
        z1: Math.max(0, Math.min(4, rates.z1 || 0)),
        z2: Math.max(0, Math.min(4, rates.z2 || 0)),
        z3: Math.max(0, Math.min(4, rates.z3 || 0)),
        z4: Math.max(0, Math.min(4, rates.z4 || 0)),
      };

      if (Object.values(validated).some(v => isNaN(v))) {
        flash("Invalid arrival rates");
        return;
      }

      await setArrivalRates(validated.z1, validated.z2, validated.z3, validated.z4);
      flash("Arrival rates applied");
    } catch(e) { console.error(e); }
  };

  const handleTimings = async () => {
    try { await setTimings(green, yellow, emDur); flash("Timings applied"); }
    catch(e) { console.error(e); }
  };

  const handleReset = async () => {
    try { await resetSimulation(); flash("Simulation reset"); }
    catch(e) { console.error(e); }
  };

  return (
    <div className={`rounded-xl border p-4 transition-colors ${
      mode==="emergency"?"bg-red-950/30 border-red-800":"bg-[#1e2130] border-[#2a2d3a]"
    }`}>
      {feedback && (
        <div className="mb-3 px-3 py-1.5 bg-blue-900/40 text-blue-300 text-xs rounded-lg text-center border border-blue-800/50">
          {feedback}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">

        {/* 1: Simulation Controls */}
        <div className="space-y-2.5">
          <p className="text-[9px] uppercase tracking-wider text-[#6b7080] font-semibold">Simulation Controls</p>
          <div className="grid grid-cols-2 gap-1">
            {PATTERNS.map(p=>(
              <button key={p.id} onClick={()=>handlePattern(p.id)}
                className={`px-2 py-1.5 rounded text-[10px] font-medium transition-all ${
                  activePattern===p.id?`${p.c} text-white`:"bg-[#141620] text-[#6b7080] hover:bg-[#252838] border border-[#2a2d3a]"
                }`}>{p.label}</button>
            ))}
          </div>

          {/* Per-zone arrival rates */}
          <div>
            <p className="text-[9px] text-[#6b7080] mb-1">Arrival Rates (v/s)</p>
            <div className="grid grid-cols-4 gap-1">
              {[1,2,3,4].map(z=>(
                <div key={z}>
                  <label className="text-[8px] text-[#6b7080]">Z{z}</label>
                  <input type="number" min="0" max="4" step="0.1"
                    value={rates[`z${z}`]}
                    onChange={e=>setRates({...rates,[`z${z}`]:parseFloat(e.target.value)||0})}
                    className="w-full bg-[#141620] border border-[#2a2d3a] rounded px-1.5 py-0.5 text-[10px] text-white"/>
                </div>
              ))}
            </div>
            <button onClick={handleRates}
              className="w-full mt-1 px-2 py-1 text-[9px] bg-[#252838] text-[#9ca0b0] rounded hover:bg-[#2a2d3a] border border-[#2a2d3a]">
              Apply Rates
            </button>
          </div>
        </div>

        {/* 2: GPS Simulator */}
        <div className="space-y-2.5">
          <p className="text-[9px] uppercase tracking-wider text-[#6b7080] font-semibold">GPS Simulator</p>
          <div className="grid grid-cols-4 gap-1">
            {[1,2,3,4].map(z=>{
              const isAct=lastEvent?.emergency_zone===z&&mode==="emergency";
              const dirs=["N","E","S","W"];
              return (
                <button key={z} onClick={()=>handleGPS(z)} disabled={loading!==null}
                  className={`py-2 rounded-lg text-center transition-all border ${
                    isAct?"border-red-500 bg-red-900/40 text-red-300"
                    :"border-[#2a2d3a] bg-[#141620] text-[#9ca0b0] hover:border-blue-500 hover:bg-blue-900/20"
                  } disabled:opacity-30`}>
                  <span className="block text-xs font-bold">Z{z}</span>
                  <span className="block text-[9px] text-[#6b7080]">{dirs[z-1]}</span>
                </button>
              );
            })}
          </div>
          {mode==="emergency"&&(
            <div className="bg-red-900/30 rounded-lg p-2 border border-red-800/50 space-y-1.5">
              {/* Active ambulance */}
              <div className="flex items-center justify-between">
                <p className="text-[10px] text-red-400 font-medium">
                  🚑 Active: Zone {lastEvent?.emergency_queue?.active?.zone || lastEvent?.emergency_zone}
                </p>
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-red-800 text-red-200">
                  {lastEvent?.emergency_queue?.active?.id || "AMB"}
                </span>
              </div>

              {/* ETA / Distance / Green */}
              {lastEvent?.emergency_eta !== undefined && (
                <div className="grid grid-cols-3 gap-1 text-center">
                  <div>
                    <p className="text-[8px] text-[#6b7080]">Distance</p>
                    <p className="text-[10px] font-bold text-orange-400">
                      {lastEvent.emergency_distance}m
                    </p>
                  </div>
                  <div>
                    <p className="text-[8px] text-[#6b7080]">ETA</p>
                    <p className="text-[10px] font-bold text-yellow-400">
                      {lastEvent.emergency_eta}s
                    </p>
                  </div>
                  <div>
                    <p className="text-[8px] text-[#6b7080]">Green</p>
                    <p className="text-[10px] font-bold text-emerald-400">
                      {lastEvent.emergency_green_adaptive}s
                    </p>
                  </div>
                </div>
              )}

              {/* Queued ambulances */}
              {lastEvent?.emergency_queue?.queue?.length > 0 && (
                <div className="border-t border-red-800/50 pt-1.5">
                  <p className="text-[8px] text-[#6b7080] mb-1">
                    QUEUED ({lastEvent.emergency_queue.queue.length})
                  </p>
                  {lastEvent.emergency_queue.queue.map((amb) => (
                    <div key={amb.id} className="flex items-center justify-between text-[9px] py-0.5">
                      <span className="text-yellow-400">
                        {amb.id}: Zone {amb.zone}
                      </span>
                      <span className="text-[#6b7080]">
                        {amb.distance_m}m · ETA {amb.eta_seconds}s
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Stats */}
              {lastEvent?.emergency_queue?.total_requests > 1 && (
                <div className="border-t border-red-800/50 pt-1 flex gap-3 text-[8px] text-[#6b7080]">
                  <span>Total: {lastEvent.emergency_queue.total_requests}</span>
                  <span>Done: {lastEvent.emergency_queue.completed_count}</span>
                  {lastEvent.emergency_queue.total_preempted > 0 && (
                    <span className="text-orange-400">
                      Preempted: {lastEvent.emergency_queue.total_preempted}
                    </span>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

       

        {/* 4: Signal Timings */}
        <div className="space-y-2.5">
          <p className="text-[9px] uppercase tracking-wider text-[#6b7080] font-semibold">Signal Timings</p>
          <div className="grid grid-cols-3 gap-1.5">
            {[
              {l:"Green",v:green,s:setGreen,min:2,max:30},
              {l:"Yellow",v:yellow,s:setYellow,min:1,max:5},
              {l:"Emerg",v:emDur,s:setEmDur,min:5,max:60},
            ].map(({l,v,s,min,max})=>(
              <div key={l}>
                <label className="text-[9px] text-[#6b7080]">{l}</label>
                <input type="number" min={min} max={max} value={v}
                  onChange={e=>s(Number(e.target.value))}
                  className="w-full bg-[#141620] border border-[#2a2d3a] rounded px-2 py-1 text-xs text-white"/>
              </div>
            ))}
          </div>
          <button onClick={handleTimings}
            className="w-full px-2 py-1.5 text-[10px] bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium">
            Apply
          </button>
        </div>

        {/* 5: Quick Actions */}
        <div className="space-y-2.5">
          <p className="text-[9px] uppercase tracking-wider text-[#6b7080] font-semibold">Quick Actions</p>
          <div className="space-y-1.5">
            <button onClick={handleReset}
              className="w-full px-2 py-1.5 text-[10px] border border-red-800 text-red-400 rounded-lg hover:bg-red-900/30 font-medium">
              Reset Simulation
            </button>
            <button onClick={()=>window.dispatchEvent(new CustomEvent("navigate",{detail:"comparison"}))}
              className="w-full px-2 py-1.5 text-[10px] border border-[#2a2d3a] text-[#9ca0b0] rounded-lg hover:bg-[#252838] font-medium">
              Run Benchmark
            </button>
          </div>
          <div className="bg-[#141620] rounded-lg p-2 border border-[#2a2d3a] space-y-1">
            <p className="text-[9px] text-[#6b7080]">System</p>
            {/* Previously a hardcoded ["Arduino COM20","WebSocket Live","DQN Agent"]
                list with an unconditional green dot on every item, regardless of
                actual connection state. Arduino/WebSocket now derive from the same
                live signals TopBar uses (lastEvent.arduino_ack, the connected prop).
                "DQN Agent" stays a static label — the live backend only ever runs
                DQN (config.py's primary_agent), there's no dynamic controller-switch
                signal in the broadcast to bind this to. */}
            {[
              { label: lastEvent?.arduino_ack ? "Arduino COM20" : "Arduino (SIM)", ok: !!lastEvent?.arduino_ack },
              { label: "WebSocket Live", ok: !!connected },
              { label: "DQN Agent", ok: true },
            ].map(({ label, ok }) => (
              <div key={label} className="flex items-center gap-1">
                <div className={`w-1.5 h-1.5 rounded-full ${ok ? "bg-emerald-400" : "bg-red-400"}`}/>
                <span className="text-[10px] text-[#9ca0b0]">{label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}