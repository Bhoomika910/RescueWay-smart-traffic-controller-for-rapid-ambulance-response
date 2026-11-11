// src/components/simulation/IntersectionView.jsx
import { useMemo } from "react";
import { queueLevel } from "../../utils/queueStatus";

const W = 580;
const H = 580;
const ROAD_W = 76;
const HALF = W / 2;
const CAR_W = 12;
const CAR_H = 22;
const CAR_GAP = 26;
const LIGHT_R = 5;

const CAR_COLORS = ["#4a9eff","#ff6b6b","#ffd43b","#69db7c","#da77f2","#ff922b","#20c997","#74c0fc","#e599f7","#ffa94d"];

function getVehicles(zone, count) {
  const n = Math.min(count, 14);
  const out = [];
  for (let i = 0; i < n; i++) {
    let x, y, rot;
    const off = i * CAR_GAP;
    switch (zone) {
      case 1: x = HALF + ROAD_W/4; y = HALF - ROAD_W/2 - 18 - off; rot = 180; break;
      case 2: x = HALF + ROAD_W/2 + 18 + off; y = HALF + ROAD_W/4; rot = 270; break;
      case 3: x = HALF - ROAD_W/4; y = HALF + ROAD_W/2 + 18 + off; rot = 0; break;
      case 4: x = HALF - ROAD_W/2 - 18 - off; y = HALF - ROAD_W/4; rot = 90; break;
      default: x=0;y=0;rot=0;
    }
    out.push({ x, y, rot, color: CAR_COLORS[i % CAR_COLORS.length] });
  }
  return out;
}

function getAmbPos(zone) {
  if (!zone) return null;
  switch (zone) {
    case 1: return { x: HALF + ROAD_W/4, y: HALF - ROAD_W/2 - 50, rot: 180 };
    case 2: return { x: HALF + ROAD_W/2 + 50, y: HALF + ROAD_W/4, rot: 270 };
    case 3: return { x: HALF - ROAD_W/4, y: HALF + ROAD_W/2 + 50, rot: 0 };
    case 4: return { x: HALF - ROAD_W/2 - 50, y: HALF - ROAD_W/4, rot: 90 };
    default: return null;
  }
}

function getQueuedAmbPositions(emergencyQueue) {
  if (!emergencyQueue?.queue) return [];
  return emergencyQueue.queue
    .map(amb => {
      const pos = getAmbPos(amb.zone);
      if (!pos) return null;
      // Offset queued ambulances slightly further from intersection
      const offset = 35;
      switch (amb.zone) {
        case 1: return { ...pos, y: pos.y - offset, id: amb.id, zone: amb.zone };
        case 2: return { ...pos, x: pos.x + offset, id: amb.id, zone: amb.zone };
        case 3: return { ...pos, y: pos.y + offset, id: amb.id, zone: amb.zone };
        case 4: return { ...pos, x: pos.x - offset, id: amb.id, zone: amb.zone };
        default: return null;
      }
    })
    .filter(Boolean);
}

function getLightPos(zone) {
  const o = ROAD_W/2 + 16;
  switch (zone) {
    case 1: return { x: HALF+o, y: HALF-o };
    case 2: return { x: HALF+o, y: HALF+o };
    case 3: return { x: HALF-o, y: HALF+o };
    case 4: return { x: HALF-o, y: HALF-o };
    default: return { x:0, y:0 };
  }
}

const SIG = { red:"#ef4444", yellow:"#f59e0b", green:"#22c55e" };

/** @typedef {import('../../types').LiveEvent} LiveEvent */

/**
 * @param {Object} props
 * @param {LiveEvent} props.lastEvent - Most recent live event pushed over the WebSocket.
 */
export default function IntersectionView({ lastEvent }) {
  const ql = lastEvent?.queue_lengths || {};
  const ss = lastEvent?.signal_states || {};
  const phase = lastEvent?.phase || "green";
  const active = lastEvent?.active_signal || 1;
  const mode = lastEvent?.mode || "normal";
  const emZ = lastEvent?.emergency_zone || 0;
  const step = lastEvent?.step || 0;
  const pattern = lastEvent?.traffic || "medium";
  const gd = lastEvent?.green_duration || 5;
  const signalColor = lastEvent?.signal_color || "red";

  // Named outside the useMemo call so each dependency is a plain
  // identifier react-hooks/exhaustive-deps can statically verify, rather
  // than a bracket-access expression inside the array literal — the
  // eslint rule can't confirm a computed member expression's stability,
  // even though ql[1..4] were already exactly the right, minimal
  // dependencies (recompute only when a queue COUNT changes, not on
  // every new `ql` object reference, which a `[ql]` dependency would
  // trigger on every single WebSocket message regardless of whether the
  // counts actually changed).
  const q1 = ql[1] || 0;
  const q2 = ql[2] || 0;
  const q3 = ql[3] || 0;
  const q4 = ql[4] || 0;
  const vehicles = useMemo(() => {
    const r = {};
    for (const [z, count] of [[1, q1], [2, q2], [3, q3], [4, q4]]) r[z] = getVehicles(z, count);
    return r;
  }, [q1, q2, q3, q4]);

  const amb = getAmbPos(emZ);
  const R = ROAD_W/2;

  return (
    <div className={`rounded-xl border overflow-hidden h-full flex flex-col ${
      mode === "emergency" ? "border-red-700" : "border-[#2a2d3a]"
    }`} style={{ background: "#141620" }}>
      {/* Header */}
      <div className={`px-4 py-2 flex items-center justify-between ${
        mode === "emergency" ? "bg-red-900/60" : "bg-[#1a1d27]"
      }`}>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${
            mode === "emergency" ? "bg-red-400 animate-pulse" : "bg-emerald-400"
          }`} />
          <span className="text-xs font-semibold text-white">LIVE INTERSECTION</span>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-[#6b7080]">
          <span>Step {step}</span>
          <span className="capitalize">{pattern}</span>
          {mode === "emergency" ? (
            <span className="px-2 py-0.5 rounded bg-red-800 text-red-200 font-medium">
              EMERGENCY Z{emZ} · ETA {lastEvent?.emergency_eta || "?"}s · {lastEvent?.emergency_distance || "?"}m
            </span>
          ) : (
            <span className={`px-2 py-0.5 rounded font-medium ${
              phase === "yellow" ? "bg-yellow-900/50 text-yellow-300" :
              phase === "all_red" ? "bg-red-900/50 text-red-300" :
              "bg-emerald-900/50 text-emerald-300"
            }`}>
              {phase === "yellow" ? "YELLOW PHASE" :
               phase === "all_red" ? "ALL RED" :
               "NORMAL"}
            </span>
          )}
        </div>
      </div>

      {/* SVG */}
      <div className="flex-1 flex items-center justify-center p-2">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxHeight: "100%" }}>
          {/* Grass */}
          <rect x={0} y={0} width={HALF-R} height={HALF-R} fill="#0d2818" rx="3"/>
          <rect x={HALF+R} y={0} width={HALF-R} height={HALF-R} fill="#0d2818" rx="3"/>
          <rect x={0} y={HALF+R} width={HALF-R} height={HALF-R} fill="#0d2818" rx="3"/>
          <rect x={HALF+R} y={HALF+R} width={HALF-R} height={HALF-R} fill="#0d2818" rx="3"/>

          {/* Roads */}
          <rect x={HALF-R} y={0} width={ROAD_W} height={H} fill="#1e1e24"/>
          <rect x={0} y={HALF-R} width={W} height={ROAD_W} fill="#1e1e24"/>
          <rect x={HALF-R} y={HALF-R} width={ROAD_W} height={ROAD_W} fill="#1e1e24"/>

          {/* Lane markings */}
          {Array.from({length:20}).map((_,i)=>{
            const y1=i*30+5; const y2=H-(i*30+5);
            const x1=i*30+5; const x2=W-(i*30+5);
            const skip=(v)=>v>HALF-R-5&&v<HALF+R+5;
            return <g key={`m${i}`}>
              {!skip(y1)&&<rect x={HALF-1} y={y1} width={2} height={16} fill="#333" rx="1"/>}
              {!skip(y2)&&<rect x={HALF-1} y={y2-16} width={2} height={16} fill="#333" rx="1"/>}
              {!skip(x1)&&<rect x={x1} y={HALF-1} width={16} height={2} fill="#333" rx="1"/>}
              {!skip(x2)&&<rect x={x2-16} y={HALF-1} width={16} height={2} fill="#333" rx="1"/>}
            </g>;
          })}

          {/* Stop lines */}
          <rect x={HALF} y={HALF-R-2} width={R} height={2} fill="#555" rx="1"/>
          <rect x={HALF+R} y={HALF} width={2} height={R} fill="#555" rx="1"/>
          <rect x={HALF-R} y={HALF+R} width={R} height={2} fill="#555" rx="1"/>
          <rect x={HALF-R-2} y={HALF-R} width={2} height={R} fill="#555" rx="1"/>

          {/* Traffic Lights */}
          {[1,2,3,4].map(z=>{
            const p=getLightPos(z);
            const sc=ss[z]||(z===active?signalColor:"red");
            const isEm=z===emZ&&mode==="emergency";
            return <g key={`l${z}`}>
              <rect x={p.x-9} y={p.y-16} width={18} height={32} rx="3" fill="#111" stroke={isEm?"#ef4444":"#333"} strokeWidth={isEm?2:0.5}/>
              <circle cx={p.x} cy={p.y-9} r={LIGHT_R} fill={sc==="red"?SIG.red:"#2a1010"} opacity={sc==="red"?1:0.25}/>
              <circle cx={p.x} cy={p.y} r={LIGHT_R} fill={sc==="yellow"?SIG.yellow:"#2a2510"} opacity={sc==="yellow"?1:0.25}/>
              <circle cx={p.x} cy={p.y+9} r={LIGHT_R} fill={sc==="green"?SIG.green:"#102a10"} opacity={sc==="green"?1:0.25}/>
            </g>;
          })}

          {/* Vehicles */}
          {[1,2,3,4].map(z=>vehicles[z]?.map((v,i)=>(
            <g key={`c${z}-${i}`} transform={`translate(${v.x},${v.y}) rotate(${v.rot})`} style={{transition:"transform 0.4s ease-out"}}>
              <rect x={-CAR_W/2} y={-CAR_H/2} width={CAR_W} height={CAR_H} rx="2.5" fill={v.color} opacity="0.85"/>
              <rect x={-CAR_W/2+1.5} y={-CAR_H/2+2} width={CAR_W-3} height={5} rx="1" fill="#000" opacity="0.35"/>
              <rect x={-CAR_W/2+1} y={CAR_H/2-3} width={2.5} height={1.5} rx="0.5" fill="#ff4444" opacity="0.6"/>
              <rect x={CAR_W/2-3.5} y={CAR_H/2-3} width={2.5} height={1.5} rx="0.5" fill="#ff4444" opacity="0.6"/>
            </g>
          )))}

          {/* Ambulance */}
          {amb&&(
            <g transform={`translate(${amb.x},${amb.y}) rotate(${amb.rot})`} style={{transition:"transform 0.5s ease-out"}}>
              <rect x={-8} y={-14} width={16} height={28} rx="3" fill="#fff" stroke="#ef4444" strokeWidth="1.5"/>
              <rect x={-1} y={-8} width={2} height={12} fill="#ef4444" rx="0.5"/>
              <rect x={-4.5} y={-4} width={9} height={2} fill="#ef4444" rx="0.5"/>
              <circle cx={-3.5} cy={-11} r={2} fill="#ef4444"><animate attributeName="opacity" values="1;0.2;1" dur="0.5s" repeatCount="indefinite"/></circle>
              <circle cx={3.5} cy={-11} r={2} fill="#3b82f6"><animate attributeName="opacity" values="0.2;1;0.2" dur="0.5s" repeatCount="indefinite"/></circle>
            </g>
          )}

          {/* Queued Ambulances (slightly dimmer, further from intersection) */}
          {getQueuedAmbPositions(lastEvent?.emergency_queue).map((qAmb, i) => (
            <g key={`qamb-${i}`}
              transform={`translate(${qAmb.x}, ${qAmb.y}) rotate(${getAmbPos(qAmb.zone)?.rot || 0})`}
              style={{transition:"transform 0.5s ease-out"}}
              opacity="0.6"
            >
              <rect x={-8} y={-14} width={16} height={28} rx="3" fill="#fff" stroke="#f59e0b" strokeWidth="1.5"/>
              <rect x={-1} y={-8} width={2} height={12} fill="#f59e0b" rx="0.5"/>
              <rect x={-4.5} y={-4} width={9} height={2} fill="#f59e0b" rx="0.5"/>
              <circle cx={-3.5} cy={-11} r={2} fill="#f59e0b">
                <animate attributeName="opacity" values="1;0.3;1" dur="0.8s" repeatCount="indefinite"/>
              </circle>
              <circle cx={3.5} cy={-11} r={2} fill="#3b82f6">
                <animate attributeName="opacity" values="0.3;1;0.3" dur="0.8s" repeatCount="indefinite"/>
              </circle>
            </g>
          ))}

          {/* Emergency glow */}
          {mode==="emergency"&&<circle cx={HALF} cy={HALF} r={R+4} fill="none" stroke="#22c55e" strokeWidth="1.5" opacity="0.4">
            <animate attributeName="opacity" values="0.2;0.5;0.2" dur="1.5s" repeatCount="indefinite"/>
          </circle>}

          {/* Zone labels + queue counts */}
          {[
            {z:1,x:HALF+R+28,y:28},
            {z:2,x:W-28,y:HALF+R+28},
            {z:3,x:HALF-R-28,y:H-20},
            {z:4,x:28,y:HALF-R-20},
          ].map(({z,x,y})=>{
            const c=ql[z]||0;
            const bg=z===emZ&&mode==="emergency"?"#ef4444":z===active&&signalColor==="green"?"#22c55e":"#2a2d3a";
            const cLevel=queueLevel(c);
            const cc=cLevel==="danger"?"#ef4444":cLevel==="warning"?"#f59e0b":"#22c55e";
            return <g key={`zl${z}`}>
              <rect x={x-22} y={y-9} width={44} height={18} rx="4" fill={bg} opacity="0.9"/>
              <text x={x} y={y+4} textAnchor="middle" fontSize="9" fill="#fff" fontWeight="700">ZONE {z}</text>
              <circle cx={x} cy={y+24} r={11} fill="#111" stroke={cc} strokeWidth="1.5"/>
              <text x={x} y={y+28} textAnchor="middle" fontSize="9" fill={cc} fontWeight="700">{c}</text>
            </g>;
          })}
        </svg>
      </div>

      {/* Footer */}
      <div className={`px-4 py-1.5 flex items-center justify-between text-[10px] ${
        mode==="emergency"?"bg-red-900/40 text-red-300":"bg-[#1a1d27] text-[#6b7080]"
      }`}>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1"><span className="w-2.5 h-1.5 rounded-sm bg-blue-400"/> Vehicle</span>
          <span className="flex items-center gap-1"><span className="w-2.5 h-1.5 rounded-sm bg-white border border-red-400"/> Active</span>
          <span className="flex items-center gap-1"><span className="w-2.5 h-1.5 rounded-sm bg-white border border-yellow-400" style={{opacity:0.6}}/> Queued</span>
          <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-green-400"/> Signal</span>
        </div>
        <span>Total: {Object.values(ql).reduce((a,b)=>a+b,0)} vehicles · Green: {gd}s</span>
      </div>
    </div>
  );
}