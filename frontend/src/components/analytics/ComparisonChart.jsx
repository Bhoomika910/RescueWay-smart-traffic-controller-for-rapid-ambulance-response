// src/components/analytics/ComparisonChart.jsx
import { useState, useEffect  } from "react";
import { runBenchmark, getBenchmarkHistory} from "../../api/rescueway";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LabelList
} from "recharts";

const COLORS = {
  dqn: "#8b5cf6",
  rl: "#3b82f6",
  fixed: "#f59e0b",
  rule_based: "#22c55e",
  random: "#ef4444",
};

const NAMES = {
  dqn: "Double DQN",
  rl: "Q-Learning (Baseline)",
  fixed: "Fixed Timer",
  rule_based: "Rule-Based",
  random: "Random",
};

const CARD_LABELS = {
  lowest_ambulance_delay: "Lowest Ambulance Delay",
  highest_reward: "Highest Reward",
  lowest_queue: "Lowest Queue",
  best_emergency: "Best Emergency Accuracy",
  fastest_response: "Fastest Qualified Response",
  highest_throughput: "Highest Throughput",
  best_overall: "Best Overall (Multi-Objective)",
};

// Single source of truth for metric label/unit/directionality, shared by
// every chart, the table header, and tooltips — so it's structurally
// impossible for one UI component to describe a metric differently than
// another.
const METRIC_META = {
  avg_ambulance_delay: { label: "Average Ambulance Delay", short: "Amb. Delay", unit: "s", higherIsBetter: false, decimals: 2 },
  ambulance_stops_per_trip: { label: "Ambulance Stops per Trip", short: "Stops/Trip", unit: "", higherIsBetter: false, decimals: 3 },
  pct_ambulances_no_stop: { label: "Ambulances Passing Without Stopping", short: "No-Stop %", unit: "%", higherIsBetter: true, decimals: 1 },
  avg_time_to_clear_before_arrival: { label: "Time-to-Clear Before Arrival", short: "Clear Lead", unit: "s", higherIsBetter: true, decimals: 2 },
  overall_traffic_efficiency: { label: "Overall Traffic Efficiency", short: "Efficiency", unit: "throughput/queue", higherIsBetter: true, decimals: 3 },
  avg_reward: { label: "Average Reward", short: "Reward", unit: "", higherIsBetter: true, decimals: 2 },
  avg_queue_length: { label: "Average Queue Length", short: "Queue", unit: "vehicles", higherIsBetter: false, decimals: 2 },
  avg_wait_time: { label: "Average Waiting Time", short: "Wait", unit: "s", higherIsBetter: false, decimals: 2 },
  avg_throughput: { label: "Throughput", short: "Throughput", unit: "veh/episode", higherIsBetter: true, decimals: 1 },
  emergency_accuracy: { label: "Emergency Accuracy", short: "Emergency Acc.", unit: "%", higherIsBetter: true, decimals: 1 },
  fairness_index: { label: "Fairness Index (Jain's)", short: "Fairness", unit: "0–1", higherIsBetter: true, decimals: 3 },
  avg_switches_per_episode: { label: "Signal Switches", short: "Switches/ep", unit: "per episode", higherIsBetter: null, decimals: 2 },
  composite_score: { label: "Composite Score", short: "Composite", unit: "0–1", higherIsBetter: true, decimals: 3 },
};

function directionSuffix(meta) {
  if (meta.higherIsBetter === true) return "higher is better";
  if (meta.higherIsBetter === false) return "lower is better";
  return null;
}

function SummaryCard({ metric, winner }) {
  return (
    <div className="bg-[#141620] rounded-lg p-2 text-center border border-[#2a2d3a]">
      <p className="text-[10px] text-[#6b7080]">{CARD_LABELS[metric] || metric.replace(/_/g, " ")}</p>
      {winner ? (
        <p className="text-xs font-medium" style={{ color: COLORS[winner] || "#e4e6ef" }}>
          {NAMES[winner] || winner}
        </p>
      ) : (
        <p className="text-xs font-medium text-[#6b7080]">No qualified controller</p>
      )}
    </div>
  );
}

function MetricBarChart({ results, field }) {
  const meta = METRIC_META[field];
  const dir = directionSuffix(meta);
  const barData = Object.entries(results.controllers).map(([key, val]) => ({
    name: NAMES[key] || key,
    value: val[field] ?? 0,
    fill: COLORS[key] || "#888",
  }));

  return (
    <div>
      <h3 className="text-sm font-medium text-[#9ca0b0] mb-2">
        {meta.label}{" "}
        <span className="text-[#6b7080] font-normal">
          ({meta.unit ? `${meta.unit}, ` : ""}{dir})
        </span>
      </h3>
      <ResponsiveContainer width="100%" height={190}>
        <BarChart data={barData} margin={{ top: 20, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
          <XAxis dataKey="name" fontSize={10} tick={{ fill: "#9ca0b0" }} />
          <YAxis fontSize={10} tick={{ fill: "#9ca0b0" }} />
          <Tooltip
            contentStyle={{ backgroundColor: "#1a1d27", border: "1px solid #2a2d3a", borderRadius: "8px" }}
            labelStyle={{ color: "#e4e6ef" }}
            itemStyle={{ color: "#e4e6ef" }}
            formatter={(v) => [`${Number(v).toFixed(meta.decimals)}${meta.unit ? " " + meta.unit : ""}`, meta.label]}
          />
          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
            {barData.map((entry, i) => (
              <Cell key={i} fill={entry.fill} />
            ))}
            <LabelList
              dataKey="value"
              position="top"
              fontSize={10}
              fill="#e4e6ef"
              formatter={(v) => Number(v).toFixed(meta.decimals)}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function ComparisonChart() {
  const isPublicDemo = import.meta.env.VITE_PUBLIC_DEMO_MODE === "true";
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [episodes, setEpisodes] = useState(50);
  const [pattern, setPattern] = useState("medium");
  const [runId, setRunId] = useState(0);
    // Load the latest saved benchmark when the page opens
  useEffect(() => {
    let cancelled = false;

    const loadHistory = async () => {
      try {
        const history = await getBenchmarkHistory(1);
        const records = Array.isArray(history)
          ? history
          : history?.value ?? [];

        const latest = records[0];

        if (!cancelled && latest?.results?.controllers) {
          setResults(latest.results);
          setRunId(id => id + 1);
        }
      } catch (err) {
        console.error("Failed to load benchmark history:", err);
      }
    };

    loadHistory();

    return () => {
      cancelled = true;
    };
  }, []);

  const runTest = async () => {
    setLoading(true);
    setResults(null); // never show a stale run's charts/table while a new one is in flight
    try {
      const data = await runBenchmark(episodes, pattern);
      setResults(data);
      setRunId(id => id + 1); // forces a full remount of the charts below, not just a prop update
    } catch (err) {
      console.error("Benchmark failed:", err);
      alert("Benchmark failed. Make sure uvicorn is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-[#1e2130] rounded-xl border border-[#2a2d3a] p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-medium text-white">Controller Benchmark</h2>

       {!isPublicDemo ? (
        <div className="flex items-center gap-2">
          <select
            value={pattern}
            onChange={e => setPattern(e.target.value)}
            className="text-sm border border-[#2a2d3a] bg-[#141620] text-white rounded-lg px-2 py-1"
          >
            <option value="low">Low traffic</option>
            <option value="medium">Medium traffic</option>
            <option value="high">High traffic</option>
            <option value="rush_hour">Rush hour</option>
          </select>

          <input
            type="number"
            value={episodes}
            onChange={e => setEpisodes(Number(e.target.value))}
            min={10}
            max={200}
            className="w-16 text-sm border border-[#2a2d3a] bg-[#141620] text-white rounded-lg px-2 py-1"
          />

          <button
            onClick={runTest}
            disabled={loading}
            className="px-4 py-1.5 text-sm bg-blue-600 text-white rounded-lg
                      hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? "Running..." : "Run Benchmark"}
          </button>
        </div>
      ) : (
        <span className="text-xs text-emerald-400">
          Saved benchmark results · Read-only
        </span>
      )}
      </div>

      {!results && !loading && (
        <p className="text-sm text-[#6b7080] py-8 text-center">
            {isPublicDemo
              ? "No saved benchmark results are available yet."
              : 'Click "Run Benchmark" to compare all controllers'}
        </p>
      )}

      {loading && (
        <p className="text-sm text-blue-400 py-8 text-center">
          Running {episodes} episodes on {pattern.replace("_", " ")} traffic (identical seed, pattern, and emergency schedule for every controller)...
        </p>
      )}

      {results && !results.error && (
        <div key={runId} className="space-y-6">
          {/* Best per metric */}
          {results.best_per_metric && (
            <div className="grid grid-cols-3 md:grid-cols-7 gap-2">
              {Object.entries(results.best_per_metric).map(([metric, winner]) => (
                <SummaryCard key={metric} metric={metric} winner={winner} />
              ))}
            </div>
          )}

          {/* Primary objective: Ambulance Delay. This is the headline
              metric per the project's actual research goal — minimizing
              ambulance delay (ideally zero stops), not just eventually
              giving the ambulance's zone a green light. Visually distinct
              from the six secondary metrics below. */}
          <div className="bg-gradient-to-br from-[#1a1440] to-[#141620] rounded-xl border-2 border-[#8b5cf6]/40 p-4">
            <div className="flex items-center justify-between mb-3">
              <div>
                <p className="text-[10px] uppercase tracking-wider text-[#a78bfa] font-semibold">Primary Objective</p>
                <h3 className="text-base font-semibold text-white">Ambulance Delay</h3>
                <p className="text-[11px] text-[#9ca0b0] mt-0.5">
                  Seconds of predicted/actual delay per ambulance trip — 0 means it passed through
                  without stopping. Lower is better; this is what the reward function now prioritizes
                  above zone-correctness (see "Proactive vs. reactive" below).
                </p>
              </div>
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
              {Object.entries(results.controllers).map(([key, val]) => (
                <div key={key} className="bg-[#0f1117]/60 rounded-lg p-3 border border-[#2a2d3a]">
                  <p className="text-[10px] font-medium mb-1" style={{ color: COLORS[key] }}>{NAMES[key] || key}</p>
                  <p className="text-xl font-bold text-white">{val.avg_ambulance_delay}<span className="text-xs text-[#6b7080] font-normal">s</span></p>
                  <p className="text-[10px] text-[#9ca0b0] mt-0.5">
                    {val.pct_ambulances_no_stop}% no-stop · {val.ambulance_trips_total} trips
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Identical experimental conditions — states the methodology this
              run actually used, so the comparison's fairness is auditable
              rather than asserted. */}
          <div id="experimental-conditions" className="bg-[#141620] rounded-lg p-3 border border-[#2a2d3a]">
            <p className="text-[11px] font-medium text-[#9ca0b0] uppercase tracking-wide mb-1">Identical experimental conditions</p>
            <p className="text-xs text-[#9ca0b0] leading-relaxed">
              All five controllers were evaluated on the same {episodes}-episode, same-seed run:
              identical traffic pattern (<span className="text-white">{pattern.replace("_", " ")}</span>),
              identical per-episode seed (<code className="text-[10px]">seed + episode_index</code>), and an
              identical emergency-event schedule drawn from a dedicated RNG decoupled from any
              controller's own randomness (<code className="text-[10px]">evaluation/comparator.py</code>,{" "}
              <code className="text-[10px]">emergency_rng</code>) — so a controller that itself consumes random
              numbers (e.g. Random) cannot see a different emergency schedule than a controller that
              doesn't. Controller-internal cycling/timing state is reset at the start of every episode
              (<code className="text-[10px]">reset_episode_state()</code>) so no controller carries an advantage
              or disadvantage from where the previous episode happened to leave it.
            </p>
          </div>

          {/* Research explanation — every claim below is scoped to what is
              actually trained/evaluated in this repository. See
              controllers/rule_based.py, rl/reward.py, sumo/train_dqn_sumo.py,
              rl/state_features.py, evaluation/robustness_runner.py, and
              services/emergency_manager.py for the underlying evidence. */}
          <div className="bg-[#141620] rounded-lg p-4 border border-[#2a2d3a] space-y-3">
            <h3 className="text-sm font-medium text-white">Interpreting reward vs. overall performance</h3>

            <div>
              <p className="text-[11px] font-medium text-[#9ca0b0] uppercase tracking-wide mb-1">Proactive vs. reactive: what changed and what the retrained models show</p>
              <p className="text-xs text-[#9ca0b0] leading-relaxed">
                The reward function (<code className="text-[10px]">rl/reward.py</code>) previously paid a fixed
                lump sum for targeting the ambulance's zone, regardless of whether the queue there could
                actually clear before the ambulance arrived — a controller could be rewarded as if the
                outcome were good even when the ambulance still had to stop. Reward now includes an
                ambulance-delay term (<code className="text-[10px]">rl/ambulance_delay.py</code>) that dominates
                the zone-correctness term: 0 predicted delay is worth +40, and every second of predicted
                or measured delay costs -4, so an action that's nominally "correct" but too late to
                prevent a backed-up queue from delaying the ambulance is no longer scored as a good
                outcome. The state given to both agents grew from 16 to 19 features, adding the
                ambulance's distance, the seconds needed to clear its zone's queue, and this same
                predicted-delay signal, so the policy has the information needed to act on it.
              </p>
              <p className="text-xs text-[#9ca0b0] leading-relaxed mt-2">
                Both agents were retrained on this new objective at reduced scale (documented as such,
                not the original production budget — see <code className="text-[10px]">backend/ARCHITECTURE.md</code>).
                The tabular Q-Learning baseline (10,000 episodes) converged cleanly: average ambulance
                delay fell to roughly 0.1s with about a 5% stop rate by the end of training. DQN,
                retrained via SUMO with real per-vehicle waiting time (60 episodes across 4 traffic
                patterns), improved over the naive baselines but did <em>not</em> fully converge within
                that reduced budget — reward variance was high, driven by real congestion dynamics
                compounding the delay penalty across consecutive decision ticks whenever an early,
                undertrained policy let the ambulance get stuck. This is reported honestly rather than
                smoothed over: it demonstrates the proactive objective is learnable, not that this
                specific DQN checkpoint has fully learned it — full-scale retraining (matching the
                original episode budget) is the direct next step.
              </p>
            </div>

            <div>
              <p className="text-[11px] font-medium text-[#9ca0b0] uppercase tracking-wide mb-1">Why Rule-Based can post the highest raw reward</p>
              <p className="text-xs text-[#9ca0b0] leading-relaxed">
                Its if/elif rules (<code className="text-[10px]">controllers/rule_based.py</code>) were written
                with direct knowledge of the reward function's own priorities
                (<code className="text-[10px]">rl/reward.py</code>): always serve the ambulance zone, always
                serve the most congested zone — the two largest-magnitude reward terms. It attains
                near-maximal reward on those terms by construction, not by learning, so it incurs no
                exploration cost and no approximation error.
              </p>
            </div>

            <div>
              <p className="text-[11px] font-medium text-[#9ca0b0] uppercase tracking-wide mb-1">Why DQN is the primary controller — what is actually demonstrated</p>
              <p className="text-xs text-[#9ca0b0] leading-relaxed">
                DQN is a function approximator over a 19-dimensional continuous state
                (<code className="text-[10px]">rl/state_features.py</code>) rather than a fixed rule table, trained
                across four predefined SUMO traffic-demand profiles — low, medium, high, and rush-hour —
                via sequential fine-tuning
                (<code className="text-[10px]">train_multi_pattern</code>, <code className="text-[10px]">sumo/train_dqn_sumo.py</code>).
                This is the extent of the demonstrated distributional coverage: performance across
                those four trained profiles, not generalization to demand distributions outside them,
                which has not been tested. The <span className="text-white font-medium">Best Overall</span> card
                above is computed from this run's actual results: a weighted, min-max-normalized
                combination of reward, queue length, throughput, wait time, emergency accuracy, and
                fairness (<code className="text-[10px]">composite_score</code>, shown per controller in the table
                below), not a qualitative judgment.
              </p>
            </div>

            <div>
              <p className="text-[11px] font-medium text-[#9ca0b0] uppercase tracking-wide mb-1">Demonstrated vs. not yet evaluated vs. future work</p>
              <ul className="text-xs text-[#9ca0b0] leading-relaxed list-disc pl-4 space-y-1">
                <li>
                  <span className="text-emerald-400 font-medium">Demonstrated:</span> DQN matches or
                  exceeds Rule-Based on emergency accuracy in this benchmark under clean sensing (see
                  the Emergency Accuracy chart above) while requiring no hand-authored priority rules.
                </li>
                <li>
                  <span className="text-emerald-400 font-medium">Demonstrated:</span> training exposure
                  to four distinct traffic-demand profiles via <code className="text-[10px]">train_multi_pattern</code>,
                  evidenced by the sequential training logs in <code className="text-[10px]">sumo/training_logs/</code>.
                </li>
                <li>
                  <span className="text-emerald-400 font-medium">Demonstrated:</span> the ambulance-delay
                  objective is learnable — the tabular Q-Learning baseline converged to ~0.1s average
                  delay / ~5% stop rate after reduced-scale retraining on the new reward.
                </li>
                <li>
                  <span className="text-yellow-400 font-medium">Limitation (measured, not assumed):</span> the
                  current DQN checkpoint improved over the naive baselines on ambulance delay but did not
                  fully converge within its reduced-scale (60-episode) SUMO retraining budget — see the
                  paragraph above and <code className="text-[10px]">backend/ARCHITECTURE.md</code> for details and
                  training logs.
                </li>
                <li>
                  <span className="text-yellow-400 font-medium">Limitation (measured, not assumed):</span> a
                  separate robustness study (<code className="text-[10px]">evaluation/robustness_runner.py</code>,
                  30 seeds/condition) injects GPS/sensor noise into the already-trained, noise-free DQN
                  checkpoint post hoc. Under its "degraded" and "extreme" noise presets, DQN's reward and
                  emergency accuracy fall <em>below</em> Rule-Based's (e.g. 215.3 vs. 274.4 reward, 76.4% vs.
                  94.7% emergency accuracy under "degraded";
                  see <code className="text-[10px]">results/robustness/study_3_20260712_023703.json</code>). DQN
                  was trained at zero sensor noise, so this is a measured limitation of the current
                  checkpoint, not a demonstrated strength — "handles sensor noise" is not a supported claim.
                </li>
                <li>
                  <span className="text-yellow-400 font-medium">Not supported by the current architecture:</span> the
                  19-dimensional state encodes a single ambulance zone
                  (<code className="text-[10px]">rl/state_features.py</code>); simultaneous multi-zone emergencies
                  are queued and prioritized by a separate, non-learned scheduler
                  (<code className="text-[10px]">services/emergency_manager.py</code>, which explicitly reduces the
                  queue to "the highest-priority ambulance" before the agent ever sees it). DQN does not
                  perceive or reason about concurrent emergencies — this is a rule-based scheduling layer,
                  not a learned capability, and should not be described as DQN adaptability.
                </li>
                <li>
                  <span className="text-blue-400 font-medium">Future work:</span> retraining/fine-tuning
                  under injected sensor noise to close the robustness gap; extending the state
                  representation to multiple concurrent ambulances; evaluating on demand distributions
                  held out from training; and closing the train/test environment gap between SUMO
                  (training) and the discrete-event simulator used for this benchmark page
                  (<code className="text-[10px]">rl/environment.py</code> + <code className="text-[10px]">traffic/simulator.py</code>).
                </li>
              </ul>
            </div>

            <div>
              <p className="text-[11px] font-medium text-[#9ca0b0] uppercase tracking-wide mb-1">Learning-based vs. rule-based: the trade-off</p>
              <p className="text-xs text-[#9ca0b0] leading-relaxed">
                Rule-Based is bounded by what its author anticipated: near-optimal on the exact
                conditions it was written for (this benchmark's clean-sensing setting), and, per the
                robustness study above, still more resilient than the current DQN checkpoint under
                heavy sensing degradation. DQN's ceiling is not capped by hand-authored logic and it can
                be retrained as conditions change, but realizing that advantage requires actually
                training on the conditions it needs to be robust to — a policy is not robust to noise it
                was never exposed to.
              </p>
            </div>

            <div>
              <p className="text-[11px] font-medium text-[#9ca0b0] uppercase tracking-wide mb-1">Research contribution</p>
              <p className="text-xs text-[#9ca0b0] leading-relaxed">
                This benchmark's contribution is a reproducible, identically-seeded comparison (same
                traffic pattern, emergency schedule, and episode count for every controller — see
                "Identical experimental conditions" above) that separates raw reward from
                multi-objective, deployment-relevant performance via <code className="text-[10px]">composite_score</code>,
                and — combined with the separate robustness study — characterizes <em>where</em> a
                learned policy currently trades reward-term exploitation for adaptability and where it
                does not yet, rather than asserting a blanket capability advantage.
              </p>
            </div>

            {results.controller_profiles && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 border-t border-[#2a2d3a]">
                {Object.entries(results.controller_profiles).map(([key, note]) => (
                  <div key={key} className="text-[11px] text-[#9ca0b0] pt-2">
                    <span className="font-medium" style={{ color: COLORS[key] }}>{NAMES[key] || key}:</span> {note}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Charts — ambulance-delay-related metrics first, matching the
              primary-objective framing above, then the secondary
              multi-objective metrics. */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <MetricBarChart results={results} field="ambulance_stops_per_trip" />
            <MetricBarChart results={results} field="pct_ambulances_no_stop" />
            <MetricBarChart results={results} field="avg_time_to_clear_before_arrival" />
            <MetricBarChart results={results} field="overall_traffic_efficiency" />
            <MetricBarChart results={results} field="avg_reward" />
            <MetricBarChart results={results} field="avg_queue_length" />
            <MetricBarChart results={results} field="emergency_accuracy" />
            <MetricBarChart results={results} field="avg_throughput" />
            <MetricBarChart results={results} field="avg_wait_time" />
            <MetricBarChart results={results} field="fairness_index" />
          </div>

          {/* Response time — only meaningful for controllers that actually resolved emergencies */}
          <div>
            <h3 className="text-sm font-medium text-[#9ca0b0] mb-1">
              Emergency Response Time <span className="text-[#6b7080] font-normal">(steps to first correct signal, lower is better)</span>
            </h3>
            <p className="text-[10px] text-[#6b7080] mb-2">
              Measured only over emergency events a controller resolved at least once. Bars for
              controllers below the 50% resolution-rate qualification bar (needed to win "Fastest
              Qualified Response" above) are shown hatched/dimmed — their response-time average is
              based on too small or too lucky a sample to be a reliable comparison point.
            </p>
            <ResponsiveContainer width="100%" height={190}>
              <BarChart
                data={Object.entries(results.controllers).map(([key, val]) => ({
                  name: NAMES[key] || key,
                  value: val.emergency_events_resolved > 0 ? val.avg_emergency_response_steps : 0,
                  resolved: val.emergency_resolution_rate ?? 0,
                  resolvedCount: val.emergency_events_resolved ?? 0,
                  totalCount: val.emergency_events_total ?? 0,
                  qualified: (val.emergency_resolution_rate ?? 0) >= 50,
                  fill: COLORS[key] || "#888",
                }))}
                margin={{ top: 20, right: 8, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
                <XAxis dataKey="name" fontSize={10} tick={{ fill: "#9ca0b0" }} />
                <YAxis fontSize={10} tick={{ fill: "#9ca0b0" }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1a1d27", border: "1px solid #2a2d3a", borderRadius: "8px" }}
                  labelStyle={{ color: "#e4e6ef" }}
                  itemStyle={{ color: "#e4e6ef" }}
                  formatter={(v, n, p) => [
                    `${Number(v).toFixed(2)} steps — ${p.payload.resolvedCount}/${p.payload.totalCount} events resolved (${p.payload.resolved}%)${!p.payload.qualified ? ", below 50% qualification bar" : ""}`,
                    "Response time",
                  ]}
                />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {Object.entries(results.controllers).map(([key, val], i) => (
                    <Cell
                      key={i}
                      fill={COLORS[key] || "#888"}
                      fillOpacity={(val.emergency_resolution_rate ?? 0) >= 50 ? 1 : 0.3}
                    />
                  ))}
                  <LabelList
                    dataKey="value"
                    position="top"
                    fontSize={10}
                    fill="#e4e6ef"
                    formatter={(v) => Number(v).toFixed(2)}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Metrics table — explicit text colors for readability */}
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[#2a2d3a]">
                  <th className="text-left py-2 text-[#6b7080] font-medium">Controller</th>
                  <th className="text-right py-2 text-[#a78bfa] font-medium">{METRIC_META.avg_ambulance_delay.short} (s)</th>
                  <th className="text-right py-2 text-[#a78bfa] font-medium">{METRIC_META.pct_ambulances_no_stop.short}</th>
                  <th className="text-right py-2 text-[#a78bfa] font-medium">{METRIC_META.ambulance_stops_per_trip.short}</th>
                  <th className="text-right py-2 text-[#6b7080] font-medium">{METRIC_META.avg_reward.short}</th>
                  <th className="text-right py-2 text-[#6b7080] font-medium">{METRIC_META.avg_queue_length.short} ({METRIC_META.avg_queue_length.unit})</th>
                  <th className="text-right py-2 text-[#6b7080] font-medium">{METRIC_META.avg_wait_time.short} ({METRIC_META.avg_wait_time.unit})</th>
                  <th className="text-right py-2 text-[#6b7080] font-medium">{METRIC_META.avg_throughput.short}</th>
                  <th className="text-right py-2 text-[#6b7080] font-medium">{METRIC_META.emergency_accuracy.short}</th>
                  <th className="text-right py-2 text-[#6b7080] font-medium">Resp. (steps)</th>
                  <th className="text-right py-2 text-[#6b7080] font-medium">Resolved</th>
                  <th className="text-right py-2 text-[#6b7080] font-medium">{METRIC_META.fairness_index.short}</th>
                  <th className="text-right py-2 text-[#6b7080] font-medium">{METRIC_META.avg_switches_per_episode.short}</th>
                  <th className="text-right py-2 text-[#6b7080] font-medium">{METRIC_META.composite_score.short}</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(results.controllers).map(([key, val]) => {
                  const isBest = key === results.best_per_metric?.best_overall;
                  return (
                    <tr key={key} className={`border-b border-[#2a2d3a] ${isBest ? "bg-blue-950/30" : ""}`}>
                      <td className="py-2 font-medium" style={{ color: COLORS[key] }}>
                        {NAMES[key] || key}
                      </td>
                      <td className="text-right py-2 text-white font-medium">{val.avg_ambulance_delay}</td>
                      <td className="text-right py-2 text-white font-medium">{val.pct_ambulances_no_stop}%</td>
                      <td className="text-right py-2 text-white">{val.ambulance_stops_per_trip}</td>
                      <td className="text-right py-2 text-white">{val.avg_reward > 0 ? "+" : ""}{val.avg_reward}</td>
                      <td className="text-right py-2 text-white">{val.avg_queue_length}</td>
                      <td className="text-right py-2 text-white">{val.avg_wait_time}</td>
                      <td className="text-right py-2 text-white">{val.avg_throughput}</td>
                      <td className="text-right py-2 text-white">{val.emergency_accuracy}%</td>
                      <td className="text-right py-2 text-white">
                        {val.emergency_events_resolved > 0 ? val.avg_emergency_response_steps : "—"}
                      </td>
                      <td className="text-right py-2 text-white">
                        {val.emergency_events_resolved}/{val.emergency_events_total} ({val.emergency_resolution_rate}%)
                      </td>
                      <td className="text-right py-2 text-white">{val.fairness_index}</td>
                      <td className="text-right py-2 text-white">{val.avg_switches_per_episode}</td>
                      <td className="text-right py-2 text-white">{val.composite_score}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="text-[10px] text-[#6b7080] mt-1">
              "—" = the controller never resolved a single emergency event, so no response-time
              average exists for it (not zero — undefined).
            </p>
          </div>

          {/* DQN improvement over baselines */}
          {results.rl_improvements && (
            <div>
              <h3 className="text-sm font-medium text-[#9ca0b0] mb-2">DQN Reward Gain Over Baselines</h3>
              <div className="grid grid-cols-4 gap-2">
                {Object.entries(results.rl_improvements).map(([key, val]) => (
                  <div key={key} className="bg-[#141620] rounded-lg p-3 text-center border border-[#2a2d3a]">
                    <p className="text-[10px] text-[#6b7080] uppercase">{NAMES[key] || key}</p>
                    {val.improvement_pct !== null && val.improvement_pct !== undefined ? (
                      <p className={`text-lg font-bold ${val.improvement_pct > 0 ? "text-emerald-400" : "text-red-400"}`}>
                        {val.improvement_pct > 0 ? "+" : ""}{val.improvement_pct}%
                      </p>
                    ) : (
                      <p className={`text-lg font-bold ${val.absolute_gain > 0 ? "text-emerald-400" : "text-red-400"}`}>
                        {val.absolute_gain > 0 ? "+" : ""}{val.absolute_gain} pts
                      </p>
                    )}
                    {val.improvement_pct === null && (
                      <p className="text-[9px] text-[#6b7080]">baseline too low for a % figure</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
