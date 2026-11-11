// src/types.js
//
// Plain-JS JSDoc typedefs (no TypeScript) describing the shape of the
// WebSocket "live event" payload pushed by the backend over
// ws://.../ws/live and consumed as `lastEvent` throughout the app
// (App.jsx, TopBar, IntersectionView, AnalyticsPanel, ControlBar).
//
// This is documentation only — it exists so editors can offer autocomplete
// and so future readers have one place to see every field the frontend
// actually reads off `lastEvent`. It is assembled from actual usage across
// the live components, not from a backend schema, so treat it as a living
// document: if a component starts reading a new field, add it here too.

/**
 * A single queued (not-yet-active) ambulance/emergency request.
 * @typedef {Object} EmergencyQueueEntry
 * @property {string} id - Ambulance/request identifier (e.g. "AMB-3").
 * @property {number} zone - Zone number (1-4) the request originated from.
 * @property {number} distance_m - Distance to the intersection, in meters.
 * @property {number} eta_seconds - Estimated time of arrival, in seconds.
 */

/**
 * The currently-active emergency request, if any.
 * @typedef {Object} EmergencyQueueActive
 * @property {number} zone - Zone number (1-4) currently being served.
 * @property {string} id - Ambulance/request identifier.
 */

/**
 * Emergency vehicle preemption queue state.
 * @typedef {Object} EmergencyQueue
 * @property {EmergencyQueueActive} [active] - The request currently being served, if any.
 * @property {EmergencyQueueEntry[]} [queue] - Requests waiting behind the active one.
 * @property {number} [total_requests] - Total emergency requests seen this run.
 * @property {number} [completed_count] - Requests that have finished being served.
 * @property {number} [total_preempted] - Requests that preempted another in-progress request.
 */

/**
 * Rolling performance metrics computed by the backend evaluator.
 * @typedef {Object} LiveMetrics
 * @property {number} [avg_wait_time] - Average vehicle wait time, in seconds.
 * @property {number} [avg_queue_length] - Average queue length, in vehicles.
 * @property {number} [total_vehicles_cleared] - Vehicles cleared through the intersection so far.
 * @property {number} [clearance_rate] - Vehicles cleared / vehicles arrived, as a percentage (0-100).
 * @property {number} [emergency_accuracy] - Emergency-response accuracy, as a percentage (0-100).
 * @property {number} [fairness_index] - Jain's fairness index over per-zone green time, in [0, 1].
 * @property {number} [avg_emergency_response_time] - Average request-to-green latency per ambulance, in seconds.
 */

/**
 * Cumulative system-wide counters (distinct from the rolling LiveMetrics window).
 * @typedef {Object} SystemMetrics
 * @property {number} [total_vehicles_arrived] - Total vehicles that have arrived since simulation start.
 * @property {number} [total_vehicles_cleared] - Total vehicles cleared since simulation start.
 */

/**
 * Per-zone queue lengths, keyed by zone number (1-4). Object keys are
 * numeric-looking strings when they arrive over JSON (e.g. `{ "1": 3 }`)
 * but are indexed in components as `ql[1]`, `ql[z]`, etc.
 * @typedef {Object.<number, number>} QueueLengths
 */

/**
 * Per-zone signal colors, keyed by zone number (1-4).
 * @typedef {Object.<number, ("red"|"yellow"|"green")>} SignalStates
 */

/**
 * Shape of the JSON payload broadcast over the live WebSocket
 * (`ws://127.0.0.1:8000/ws/live`) and stored as `lastEvent` by
 * `useWebSocket` in App.jsx, then passed as a prop to TopBar,
 * IntersectionView, AnalyticsPanel, and ControlBar.
 *
 * All fields are optional from the frontend's point of view — every
 * consumer reads them with `?.` and a fallback default, since `lastEvent`
 * is `null` until the first message arrives and individual fields may be
 * absent on any given tick.
 *
 * @typedef {Object} LiveEvent
 * @property {("normal"|"emergency")} [mode] - Current operating mode.
 * @property {number} [step] - Simulation step counter.
 * @property {("low"|"medium"|"high"|"rush_hour")} [traffic] - Active traffic pattern.
 * @property {string} [arduino_ack] - Arduino handshake/status string (e.g. starts with "OK" or "SIM"); falsy when not connected.
 * @property {SystemMetrics} [system_metrics] - Cumulative vehicle counters.
 * @property {number} [active_signal] - Zone number (1-4) currently holding the green/active phase.
 * @property {number} [emergency_zone] - Zone number (1-4) associated with the active emergency, if `mode === "emergency"`.
 * @property {("red"|"yellow"|"green")} [signal_color] - Color of the active signal.
 * @property {("green"|"yellow"|"all_red"|"emergency_hold")} [phase] - Current signal phase.
 * @property {QueueLengths} [queue_lengths] - Per-zone vehicle queue lengths.
 * @property {SignalStates} [signal_states] - Per-zone signal colors.
 * @property {LiveMetrics} [live_metrics] - Rolling performance metrics.
 * @property {string} [action_name] - Human-readable name of the last RL agent action.
 * @property {number} [reward] - Reward the RL agent received for the last action.
 * @property {number} [emergency_eta] - ETA of the active emergency vehicle, in seconds.
 * @property {number} [emergency_distance] - Distance of the active emergency vehicle, in meters.
 * @property {number} [emergency_green_adaptive] - Adaptive green duration granted to the active emergency, in seconds.
 * @property {EmergencyQueue} [emergency_queue] - Emergency preemption queue state (active + waiting requests).
 * @property {number} [green_duration] - Configured/current green phase duration, in seconds.
 */

export {};
