// src/api/rescueway.js
import axios from "axios";

// Previously hardcoded to "http://127.0.0.1:8000" — worked for local dev
// only, and broke silently for any Docker/deployed setup where the API
// isn't on localhost. VITE_API_BASE_URL is read at build time (Vite only
// exposes import.meta.env.VITE_* vars to client code); see
// frontend/.env.example and docker-compose.yml for how it's set per
// environment. Falls back to the previous local-dev default so nothing
// changes for existing developers who haven't set it.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const API = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000
});

export { API_BASE_URL };

export const selectZone = (zone, traffic) =>
  API.post("/zone/select", { zone, traffic }).then(r => r.data);

export const getHealth = () =>
  API.get("/health").then(r => r.data);

export const getLiveMetrics = () =>
  API.get("/eval/live").then(r => r.data);

export const getTrends = (window = 50) =>
  API.get(`/eval/trends?window=${window}`).then(r => r.data);

export const getSummary = () =>
  API.get("/eval/summary").then(r => r.data);

export const runBenchmark = (episodes = 50, pattern = "medium") =>
  API.post(`/compare/run?episodes=${episodes}&pattern=${pattern}`).then(r => r.data);

export const getBenchmarkHistory = (limit = 10) =>
  API.get(`/compare/history?limit=${limit}`).then(r => r.data);

export const runStressTest = (episodes = 30) =>
  API.post(`/eval/stress-test?episodes=${episodes}`).then(r => r.data);

export const runComparison = (episodes = 50) =>
  API.post(`/compare/run?episodes=${episodes}`).then(r => r.data);

export const getZoneStatus = () =>
  API.get("/zone/status").then(r => r.data);

export const getSimulationStatus = () =>
  API.get("/simulation/status").then(r => r.data);

export const setTrafficPattern = (pattern) =>
  API.post("/simulation/pattern", { pattern }).then(r => r.data);

export const setArrivalRates = (zone_1, zone_2, zone_3, zone_4) =>
  API.post("/simulation/arrival-rates", { zone_1, zone_2, zone_3, zone_4 }).then(r => r.data);

export const setTimings = (green_duration, yellow_duration, emergency_duration) =>
  API.post("/simulation/timings", { green_duration, yellow_duration, emergency_duration }).then(r => r.data);

export const resetSimulation = () =>
  API.post("/simulation/reset").then(r => r.data);

export const getPresets = () =>
  API.get("/simulation/presets").then(r => r.data);

export const getRecentLogs = (count = 50, level = null, category = null) => {
  let url = `/logs/recent?count=${count}`;
  if (level) url += `&level=${level}`;
  if (category) url += `&category=${category}`;
  return API.get(url).then(r => r.data);
};

export const getLogSummary = () =>
  API.get("/logs/summary").then(r => r.data);

export const exportLogs = () =>
  API.post("/logs/export").then(r => r.data);