// src/utils/queueStatus.js
//
// Single source of truth for "how full is this queue" thresholds and
// capacity. Previously reimplemented independently — with inconsistent
// values — in AnalyticsPanel.jsx, IntersectionView.jsx, and the orphaned
// (unmounted) QueueVisualization.jsx/LiveMetricsPanel.jsx. The two live
// components already agreed on the same 15/8 vehicle thresholds, so this
// is a pure refactor (identical behavior), not a visual change.

export const QUEUE_DANGER_THRESHOLD = 15;
export const QUEUE_WARNING_THRESHOLD = 8;
export const QUEUE_MAX_CAPACITY = 50;

/** Returns "danger" | "warning" | "ok" for a given queue vehicle count. */
export function queueLevel(count) {
  if (count > QUEUE_DANGER_THRESHOLD) return "danger";
  if (count > QUEUE_WARNING_THRESHOLD) return "warning";
  return "ok";
}
