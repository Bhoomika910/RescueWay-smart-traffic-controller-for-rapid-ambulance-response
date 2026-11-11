# _legacy

The 12 components in this directory are pre-refactor / reference components.
They predate the current unified dashboard (`TopBar` / `IntersectionView` /
`AnalyticsPanel` / `ControlBar` under `src/components/unified` and
`src/components/simulation`) and were superseded by it.

They are **not part of the live app** — nothing under `src/` imports any file
in this directory. They're kept (not deleted) intentionally, for design
history and as a reference for earlier UI approaches, per product decision.

Because nothing imports them, Vite's tree-shaking excludes them from the
production build automatically — keeping this directory around has no
runtime or bundle-size cost.

Files:
- `LiveChart.jsx`
- `RLDecision.jsx`
- `RewardHistory.jsx`
- `SignalDisplay.jsx`
- `SystemStatus.jsx`
- `ZoneSelector.jsx`
- `AnalyticsPage.jsx`
- `LiveMetricsPanel.jsx`
- `QueueVisualization.jsx`
- `SimulationControls.jsx`
- `PerformanceGauge.jsx`
- `SimulationDashboard.jsx`

If you're looking for the live equivalents of these, see:
`src/components/unified/`, `src/components/simulation/IntersectionView.jsx`,
and `src/components/analytics/LogViewer.jsx` / `ComparisonChart.jsx`.
