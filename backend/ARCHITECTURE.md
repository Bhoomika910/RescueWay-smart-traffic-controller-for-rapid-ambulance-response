# Architecture: from reactive to proactive ambulance-delay minimization

## System architecture diagram

Design-review finding (category 26 "Documentation"): "no architecture diagrams anywhere
(text-only)." The module-map prose in `main.py`'s own docstring is the authoritative
description of each layer's responsibility and boundary; this diagram is a visual index
into that same set of relationships, not a replacement for it.

```mermaid
flowchart TB
    subgraph Client["Client layer"]
        FE["React dashboard\n(frontend/)"]
        ARD["Arduino Uno\n(4-zone LED rig)"]
    end

    subgraph API["Backend API — routers/"]
        WS["/ws/live\nWebSocket broadcast"]
        ZONE["/zone/*\nemergency trigger/clear"]
        SIM["/simulation/*"]
        CMP["/compare/*"]
        EVAL["/eval/*"]
        LOGS["/logs/*"]
        DATA["/data/*"]
        AUTH["routers/auth.py\nX-API-Key on state-mutating routes"]
    end

    subgraph Core["Live orchestration — services/"]
        TC["TrafficController\nsingle source of truth for signal state"]
        EM["EmergencyManager\nETA priority queue / preemption"]
        SER["SerialBridge"]
    end

    subgraph RL["RL — rl/"]
        DQNA["DQNAgent\n(Double DQN, optional Dueling+PER)"]
        QA["QAgent\n(tabular, 4320 states)"]
        SHIELD["safety_shield.py\nhard emergency-zone override"]
        SCORE["emergency_scoring.py\nsingle source of truth for\n'was this action correct'"]
        FA["feature_attribution.py\ngradient saliency / permutation importance"]
        REW["reward.py\nRewardWeights"]
        SF["state_features.py\nN_ZONES-parametrized"]
    end

    subgraph Envs["Environments — three, not interchangeable"]
        SIMPLE["traffic/simulator.py\nfast, low-fidelity queueing"]
        SUMOE["sumo/\nreal TraCI per-vehicle physics"]
    end

    subgraph Eval["evaluation/ — offline benchmarking"]
        COMP["comparator.py\nflagship 7-controller benchmark"]
        ROB["robustness_runner.py\nStudies 1-5"]
        ABL["ablation_studies.py\nStudy A/B/C"]
        STAT["statistical_analysis.py\nnormality-gated, Holm/BH, power"]
        REGISTRY["experiment_registry.py\nindexes all results/*.json"]
    end

    DB[("data/rescueway.db\nSQLite, schema-versioned")]

    FE <-->|WebSocket| WS
    FE -->|HTTPS + X-API-Key| ZONE
    FE --> SIM
    FE --> CMP
    FE --> EVAL
    FE --> LOGS
    FE --> DATA
    ZONE --> AUTH
    SIM --> AUTH
    CMP --> AUTH

    ZONE --> TC
    TC --> EM
    TC --> SER
    SER <-->|Serial, ACK-confirmed| ARD
    TC --> DQNA
    TC --> SHIELD
    TC --> FA
    DQNA --> SF
    DQNA --> REW
    SHIELD --> SCORE

    TC --> SIMPLE
    DQNA -.->|training only| SUMOE
    QA -.->|simplified-env training,\nnow also sumo/train_qlearning_sumo.py| SUMOE

    COMP --> DQNA
    COMP --> QA
    COMP --> SIMPLE
    ROB --> SIMPLE
    ABL --> SIMPLE
    STAT --> ROB
    REGISTRY --> COMP
    REGISTRY --> ROB
    REGISTRY --> ABL

    TC --> DB
    EVAL --> DB
```

Three environments appear in this diagram and are **never interchangeable ground truth**
(see `config.ENV_TAG_SIMPLIFIED`/`ENV_TAG_SUMO` and Part 8.3): the live dashboard and most
of `evaluation/` run against `traffic/simulator.py`'s fast approximation; DQN's actual
training and `sumo/evaluate_in_sumo_v2.py` run against real SUMO physics; every result file
this project produces is tagged with which one produced it.

## The research objective

The goal is **not** "prioritize the ambulance once it's at the intersection." It is
**minimize ambulance delay by clearing the intersection before the ambulance arrives**,
ideally so it never has to stop. This document records what was actually reactive before
this redesign (with file:line evidence), what changed, what deliberately did not change
and why, and the honest results of retraining both agents on the new objective.

## Part 1 — The reactive baseline (before this redesign)

Every subsystem in the emergency path was reactive, despite naming/docstrings that
suggested otherwise.

- **ETA was a fixed per-zone constant, not a live estimate.** `gps/zone_model.py`'s
  `ZONE_CONFIG` hardcodes `eta_seconds` (5/10/18/30s) per zone; `calculate_eta()` is a
  dict lookup, not a distance÷speed computation. `AMBULANCE_SPEED` was defined and never
  used. **This is unchanged by design** — see "What did not change" below.
- **The system reacted at report time, with no lead-time logic.** `EmergencyManager.add_emergency()`
  and `TrafficController.trigger_emergency()` activate/preempt synchronously, in the same
  call that receives a zone report. The yellow→all-red→green transition
  (`_safe_transition`) always started immediately on report, with identical timing
  regardless of whether ETA was 5s or 30s.
- **The one function that modeled "start clearing early" was dead code.**
  `GPSZoneModel.get_preemption_time()` computed `clearance_lead = max(0, eta - 5)` but was
  never called anywhere in the codebase.
- **The RL agent never actually received the ETA it needed.** `state_features.py` had an
  ETA feature slot, but `TrafficController._rl_choose_emergency_zone()`'s calls to
  `encode_rich_state()` omitted `emergency_eta` entirely, silently defaulting to `0.0` —
  a real bug, not a design choice.
- **The reward function had zero ETA/delay dependence.** `_emergency_reward(action,
  amb_zone, prev_signal)` paid a fixed lump sum (+25 correct zone / -25 wrong zone /
  etc.) invariant to how much lead time existed or whether the ambulance actually stopped.
- **The discrete (tabular) state encoder had no ETA dimension at all.** `state_encoder.py`'s
  `encode()` took only `emergency_active, ambulance_zone, active_signal, queue_lengths`.
- **`TrafficEnv` (the primary non-SUMO training environment) modeled ambulances as an
  instantaneous same-timestep coin flip** (`_update_ambulance`, 8%/step probability, no
  travel-time model whatsoever) — there was no lead time to learn from even in principle.
- **Nothing measured actual ambulance delay or stops.** `EmergencyManager.get_stats().avg_response_time`
  and `MetricsCollector`'s `avg_emergency_response_time`/`total_emergency_delay` all measure
  *signal reaction latency* (request timestamp → green timestamp), not the ambulance's
  experienced delay or whether it stopped. `sumo_adapter.get_ambulance_waiting_time()` — the
  one function that could measure real per-vehicle delay via TraCI — was defined and never
  called.

## Part 2 — The proactive redesign

### One shared primitive: `rl/ambulance_delay.py`

```python
estimate_clearance(remaining_eta, queue_length, zone_is_served) -> {
    "lead_time_needed": ...,   # transition time (if not already served) + queue/clearance_rate
    "deficit": ...,            # predicted delay in seconds, 0 if it clears in time
    "will_stop": ...,          # deficit > 0
    "lead_time_available": ..., # remaining_eta - lead_time_needed
}
```

Every consumer (reward, state, live controller, benchmark) calls this one function, so
the definition of "ambulance delay" only exists in one place. `VEHICLE_CLEARANCE_RATE`
(`config.py`) is the same constant `traffic/simulator.py` actually uses to clear queues,
so the estimate is internally consistent with what the simulator will really do.

**Provenance split, stated explicitly:** the simple simulator, the live `TrafficController`,
and the benchmark all use this *proxy* (a countdown-clock estimate — there is no
per-vehicle position/speed model in `traffic/simulator.py`). SUMO does not need the proxy:
it measures real per-vehicle waiting time directly via
`sumo_adapter.get_ambulance_waiting_time()` (`traci.vehicle.getWaitingTime`), now wired
into `sumo_env.py`'s reward computation. Anywhere both numbers appear side by side, treat
the SUMO number as ground truth and the proxy as an estimate.

### Reward (`rl/reward.py`)

`compute_reward()` and `_emergency_reward()` gained an optional `ambulance_delay`
parameter. When supplied, it dominates the existing zone-correctness term:

```
ambulance_delay <= 0.5s  →  +40  (passed through, negligible/no stop)
ambulance_delay  > 0.5s  →  -4.0 × ambulance_delay
```

This is *additive* on top of the unchanged zone-correctness term (+25/+10/-8/-15/-25) and
the unchanged congestion/fairness/throughput/switching terms — nothing was removed. A
"correct" action that's too late to prevent a backed-up queue from delaying the ambulance
is no longer scored as if the outcome were good: correct zone + 0 delay = +82 combined;
correct zone + 8s delay = +10; wrong zone + 8s delay = -60.25 (verified in
`rl/reward.py`'s docstring examples). Callers that don't pass `ambulance_delay` get
exactly the pre-redesign behavior — this was verified as a regression check before any
other change was made.

### State (16 → 19 features)

`rl/state_features.py` (DQN's continuous encoder) added:

- `[16]` ambulance distance, normalized
- `[17]` seconds needed to clear the ambulance's zone queue, normalized
- `[18]` predicted clearance deficit (from `estimate_clearance`), normalized

`rl/state_encoder.py` (the tabular encoder) added a 3-level `CLEARANCE_RISK_LEVELS`
bucket (ample lead time / marginal / predicted stop), taking `N_STATES` from 1440 to 4320
— still trivially small for a tabular `(N_STATES, 6)` Q-table.

**This invalidates the previous `dqn_model.pt` checkpoint outright** — its first layer was
shaped for 16 inputs, not 19. `DQNAgent.load()` now catches the resulting `RuntimeError`
and falls back to fresh weights with a clear log message instead of crashing every caller
that constructs a `DQNAgent`. `QAgent.load()` gained an analogous shape check (it
previously had none — a shape mismatch would have loaded silently and misindexed every
lookup rather than erroring).

### Bug fix: ETA now actually reaches the agent

`TrafficController._rl_choose_emergency_zone()` now computes `estimate_clearance(...)`
using `self.emergency_eta`/`self.emergency_distance` (already set by `trigger_emergency()`
before this runs) and passes `emergency_eta=`, `distance=`, and `ambulance_delay=` into
`encode_rich_state()` and `compute_reward()`. The predicted deficit is also recomputed
every tick during the emergency hold (`_run_emergency`'s hold loop), using the live
countdown, and surfaced through `xai_info` and the WebSocket broadcast
(`emergency_clearance_deficit`, `emergency_predicted_stop`) so the dashboard shows a
continuously-updating "will this ambulance stop" signal, not a one-shot value from
trigger time.

### SUMO wiring (`sumo/sumo_env.py`, `sumo/train_dqn_sumo.py`)

`_detect_ambulance()` now also captures `self.ambulance_waiting_time` (real TraCI
measurement) and `self.ambulance_distance` (real lane-position-derived value).
`step()` passes the real waiting time into `compute_reward(..., ambulance_delay=...)`.
`train_dqn_sumo.py` tracks ambulance delay per episode alongside reward, both in the
console log and the saved JSON training log (`ambulance_delay_history`,
`final_avg_ambulance_delay_20`), so the proactive objective can be verified directly —
reward can rise from congestion/throughput terms alone; average ambulance delay trending
down cannot be faked the same way.

### Benchmark (`evaluation/comparator.py`, `ComparisonChart.jsx`)

New per-controller metrics: `avg_ambulance_delay`, `ambulance_stops_per_trip`,
`pct_ambulances_no_stop`, `avg_time_to_clear_before_arrival`, `overall_traffic_efficiency`
(throughput per unit of standing queue). `avg_ambulance_delay` now carries the largest
weight (0.30) in the multi-objective `composite_score`, with the previous six weights
rescaled proportionally (×0.7) to make room — documented inline in `COMPOSITE_WEIGHTS`. A
new `lowest_ambulance_delay` summary card was added. The frontend surfaces ambulance delay
as a visually distinct "Primary Objective" section above the other six secondary-metric
charts, per-controller, with trip counts shown alongside so small samples are visible
rather than hidden.

## Part 3 — Retraining (updated 2026-07-21 with full-scale results)

**Superseded note (2026-07-21, later same day):** the DQN SUMO-native table below
predates both the held-out-pattern train/test split (Part 8.8) and the replay-buffer-
carryover forgetting fix (Part 8.9), and its accuracy figures also predate the
emergency-accuracy counting fix (Part 9) — it undercounted `action==4` ("extend current
green") as incorrect even when it was already serving the right zone. **Part 8.9's table
is the current, correct, and directly comparable per-pattern breakdown for the
now-deployed checkpoint** (750 episodes, low/medium/high curriculum with buffer carryover,
`rush_hour` genuinely held out, correct-counting fix applied). This table is left
unmodified below as a historical record of what was reported at the time, per this
document's own practice of showing corrections rather than silently rewriting prior
numbers.

Both agents were retrained from scratch (required — see checkpoint invalidation above) on
the new reward/state. The numbers below supersede an earlier reduced-scale documentation
pass (10,000-episode Q-table, 60-episode DQN) that undersold both agents' actual, current
training budgets — see Part 7 for why the docs had fallen behind the checkpoints.

**Tabular Q-Learning — 25,000 episodes (`train_offline.py --episodes 25000`, the full
original budget, re-run 2026-07-21 after the done-masking fix in Part 7/M16), converged cleanly:**

| | Final (last 200 ep) |
|---|---|
| Avg reward | 600.07 |
| Emergency accuracy | 45.2% |
| **Avg ambulance delay** | **0.10s** |
| **Ambulance stop rate** | **3.8%** |

This is direct, concrete evidence that the proactive objective is learnable at full scale:
the agent clears queues in time for the ambulance to pass through without stopping in
~96% of trips.

**DQN — currently deployed checkpoint is 1,000 SUMO episodes across
low/medium/high/rush_hour (250 each via `train_multi_pattern`, trained 2026-07-17),
evaluated natively in SUMO on 2026-07-21 (`evaluate_in_sumo_v2.py --compare-all --episodes
10 --duration 300 --pattern medium`, real per-vehicle TraCI waiting time, not the proxy):**

| Controller | Reward (±95% CI) | Emergency accuracy | Avg ambulance delay | Avg queue |
|---|---|---|---|---|
| **Double DQN** | **+1391.3 (±40.0)** | **84.6%** | **0.4s** | 29.3 |
| Q-Learning RL | +1242.6 (±105.5) | 74.0% | 6.0s | 30.3 |
| Rule-Based | +1428.6 (±35.3) | 100.0% | 0.1s | 29.9 |
| Fixed Timer | -20.9 (±212.4) | 4.7% | 49.1s | 15.9 |
| Random | -609.6 (±574.5) | 16.3% | 57.7s | 19.8 |

At full training scale and evaluated in the environment it was actually trained in, DQN is
close to converged: 84.6% emergency accuracy and a 0.4s average ambulance delay, within
reach of Rule-Based's by-construction ceiling (100%/0.1s) and far ahead of both naive
baselines. This directly resolves the previous documentation's "did not fully converge"
finding — that finding was accurate for the 60-episode intermediate checkpoint it was
based on, but the actually-deployed checkpoint had already moved well past it.

**A separate, smaller-scale caveat, not resolved by the above:** the same checkpoint,
evaluated instead through the simplified non-SUMO `TrafficEnv`/`traffic/simulator.py`
harness that `evaluation/comparator.py` and `evaluation/robustness_runner.py` use, performs
markedly worse — e.g. 35.0% emergency accuracy and near-zero-to-negative reward at zero
noise (`results/robustness/study_3_20260721_020733.json`, "clean" condition: DQN reward
-535.0 vs. Rule-Based +428.6 and Q-Learning +332.5), with accuracy that stays flat across
different traffic patterns (35.0% in both the "medium" and "rush_hour" 50-episode
benchmark runs, `benchmark_results` DB rows 39/40) — consistent with the network settling
on a narrow, largely context-insensitive policy for that harness's state distribution. This
is a genuine train/eval domain gap (DQN was trained exclusively inside SUMO; Rule-Based/
Fixed Timer/Random reason directly on whichever environment evaluates them, so they don't
suffer from this gap), not a bug fixed or introduced in this pass — see Part 5.

**Conclusion:** the proactive, delay-minimizing objective is demonstrably learnable by both
agents at full scale. The currently-deployed DQN checkpoint is a strong, near-converged
policy *within the SUMO environment it was trained in*; closing the gap to the simplified
non-SUMO evaluation harnesses (Part 5) is the direct next step, not further SUMO training.

## Part 4 — What deliberately did not change, and why

- **`services/emergency_manager.py`'s ETA-priority-queue/preempt/merge scheduling logic.**
  This redesign is additive: `TrafficController` consumes `emergency_manager.get_active_eta()`
  (already exposed) to compute the clearance deficit downstream; nothing needed to write
  back into the manager. Multi-ambulance queueing, preemption, and merge behavior are
  unchanged and remain compatible.
- **`gps/zone_model.py`'s ETA provenance** (a fixed constant per zone, not a distance÷speed
  computation). This is a deliberate simplification for a 4-zone discrete model, not a bug
  to fix as part of this work — the redesign's job was to properly *consume* the ETA that
  already exists, not to make it more physically realistic.
- **Physical yellow(2s)/all-red(1s) transition timing.** This is a hardware safety
  constraint, not a tunable lever. Because GPS detection is single-shot/event-driven (not a
  continuous pre-arrival feed), the only lever available is that `trigger_emergency()`
  already starts the transition immediately on report — there is no "earlier" to start
  from without changing the detection model itself (out of scope here; see future work).

## Part 5 — Future work

- **Close the DQN train/eval domain gap (new, replaces the old "full-scale DQN retraining"
  item — that one is done, see Part 3).** The deployed checkpoint was trained exclusively
  inside SUMO and is near-converged there (84.6% accuracy, 0.4s delay), but the same
  checkpoint underperforms in the simplified non-SUMO `TrafficEnv` that
  `evaluation/comparator.py` and `evaluation/robustness_runner.py` evaluate against (35.0%
  accuracy, frequently-negative reward even at zero injected noise — see Part 3). Either
  retrain a DQN variant natively against `TrafficEnv`, or migrate the benchmark/robustness
  harnesses onto `sumo_env.py`, so DQN is evaluated in the same environment it learned in —
  the way Rule-Based/Fixed Timer/Random already are, by construction.
- Extending the state representation to multiple concurrent ambulances (currently
  `EmergencyManager` reduces the queue to "the highest-priority ambulance" before the agent
  ever sees it — this is a rule-based scheduling layer, not a learned capability).
- A continuous (multi-tick) GPS feed instead of single-shot zone detection, which would let
  the agent receive genuine advance warning across several decision steps rather than all
  the lead time arriving at once at report time.
- **Retraining under injected sensor noise** (re-run 2026-07-21 with the H2 eta/distance
  benchmark fix — `results/robustness/study_3_20260721_020733.json`, "combined stress
  test": in the non-SUMO harness, DQN reward goes from -535.0 (clean) to -444.6 (extreme
  noise) against Rule-Based's +428.6 → +37.1 and Q-Learning's +332.5 → +28.0 — DQN
  underperforms both across the full noise range, but the gap is roughly flat rather than
  noise-driven, reinforcing that this is substantially the domain-gap finding above rather
  than a noise-specific weakness. It was also still trained at zero noise, which remains a
  valid, separate reason to expect a robustness gap.)
- Reward-scale tuning for the SUMO path specifically, to reduce the outlier variance from
  compounding real-waiting-time penalties during sustained congestion under an
  undertrained early policy (still relevant for any future from-scratch SUMO retrain, e.g.
  if pursuing the domain-gap item above by retraining natively in a different environment).
- **Multi-intersection SUMO network — deliberately scoped out, not attempted.** Everything
  in this document (state, reward, both agents, both simulators, the firmware) models a
  single 4-way intersection with 4 zones. `gps/zone_model.py`'s `N_ZONES` constant (added in
  the 2026-07-21 design-validation pass, Part 8) makes the *RL encoding layer* agnostic to
  the zone count — `tests/test_zone_parametrization.py` proves a 5/6-zone `StateFeatureExtractor`
  produces correctly-shaped, correctly-indexed features — but that is one of four things a
  real multi-intersection system needs, and the other three are not size-of-parameter
  problems: (1) the SUMO `.net.xml`/`.rou.xml` files describe one physical junction and
  would need to be authored (not generated) for a real multi-intersection road network with
  actual topology; (2) the reward function has no notion of coordinating signal phase
  *between* intersections (e.g. green-wave progression) — extending it is a genuine research
  question (multi-agent RL: independent per-intersection agents vs. a shared centralized
  policy vs. learned communication), not an engineering extension of the current
  single-agent formulation; (3) the Arduino firmware and frontend dashboard are both
  hardcoded to 4 zones end-to-end (wiring, pin maps, UI layout) and were explicitly left
  out of the `N_ZONES` parametrization work for exactly this reason — see `zone_model.py`'s
  docstring. Attempting this would mean inventing a new road network, a new multi-agent
  reward formulation, and new hardware/UI, none of which the original review asked for and
  none of which can be validated without fabricating results — which the closing instruction
  for this project's remediation phase explicitly rules out. This item stays a documented,
  justified scope boundary rather than a silent gap.

## Part 6 — Technical audit and bug-fixing pass (2026-07-17)

A follow-up audit investigated whether the Rule-Based (0.00s) / Q-Learning (0.01s) /
DQN (4.14s) ambulance-delay gap reflected undertraining alone or genuine implementation
bugs. Verdict: **B — mostly valid, but three confirmed implementation issues were limiting
DQN**, all now fixed.

### Bug #1 (High severity) — DQN/RL never received ETA/distance during benchmark evaluation

**Root cause:** `DQNController.choose_action()` (`controllers/dqn_controller.py`) and
`RLController.choose_action()` (`controllers/rl_controller.py`) called `encode_rich_state()`
with only 4 arguments; `emergency_eta`/`distance` silently defaulted to `0.0`.
`evaluation/comparator.py`'s action-selection call site only supplied those same 4 base
args to every controller. DQN's SUMO training (and the live `TrafficController`, already
fixed in an earlier pass) both fed real, varying eta/distance — so the benchmark was
scoring DQN under a state distribution it never trained on, on top of any genuine
undertraining.

**Fix:** `BaseController.choose_action()` gained optional `emergency_eta=0.0,
distance=0.0` parameters (backward-compatible — every other caller keeps working
unchanged). All 5 controllers now accept them; `DQNController`/`RLController` forward them
into `encode_rich_state()`. `evaluation/comparator.py`'s call site now passes the current
`emergency_timer` (already-tracked remaining ETA) and `ZONE_CONFIG[zone]["avg_distance"]`.

**Validated:** a spy on `DQNAgent.encode_rich_state` confirmed `emergency_eta=5.0,
distance=40.0` now actually reaches the state encoder (previously always `0.0`); all
5 controllers were confirmed to accept the new kwargs without error.

### Bug #2 (Medium) — epsilon schedule reset to 1.0 at the start of every training pattern

**Root cause:** `train()`'s epsilon line computed `get_epsilon(episode - start_episode,
episodes)` — always `0..episodes-1` regardless of `start_episode` — so `train_multi_pattern`
calling `train()` once per pattern with `resume=True` correctly resumed the network
*weights* but restarted the *exploration schedule* to ε=1.0 four separate times, wasting a
large fraction of the already-small 60-episode budget on repeated high-epsilon exploration.

**Fix:** added `epsilon_schedule_total`/`epsilon_schedule_offset` parameters to `train()`
(defaulting to old single-pattern behavior exactly); `train_multi_pattern` now passes the
grand total (60) and each pattern's cumulative offset (0/15/30/45), so ε decays smoothly
across the whole run.

**Validated:** unit-tested `get_epsilon()` directly — old behavior showed ε resetting to
1.000 at the start of all 4 patterns; new behavior showed continuous decay (1.000 → 0.683
→ 0.344 → 0.050), reaching full exploitation partway through pattern 3 and staying there,
instead of repeatedly re-exploring.

### Bug #3 (Medium) — target network update frequency mismatched to training scale

**Root cause:** `target_update_freq=1000` (a reasonable default for 100k+-step training)
against this project's ~3,000-step realistic budget meant the target network hard-updated
only ~3 times in the entire run, undermining Double DQN's core stabilization mechanism.

**Fix:** lowered the default (both `DQNAgent`'s constructor and `train_dqn_sumo.py`'s CLI)
from 1000 to 200, giving roughly 5x more target updates at the same training scale.

### Retraining and final validation

DQN was retrained from scratch (required — state dimensionality already changed in Part
2) with all three fixes active, same 60-episode/4-pattern budget as the pre-fix run for a
fair comparison. Training completed with no errors; epsilon logs confirmed the schedule
fix was active during the real run (e.g. ε=0.231 mid-run, not reset to 1.0).

The corrected benchmark (re-run via direct Python across 3 seeds and via live HTTP against
the restarted server) shows Q-Learning and Rule-Based unaffected (Q-Learning: ~0.00-0.01s
delay, 95-100% no-stop, confirming Bug #1's fix didn't destabilize the tabular agent).
**DQN's ambulance delay remained in a similar 3.7-5.1s range**, and closer inspection
surfaced a further, genuine finding: querying the retrained network directly (varying
ambulance zone, active signal, and ETA) shows it has collapsed toward almost always
selecting "Extend current green" (action 4) regardless of context — Q-values for a
representative state showed `Extend=80.7` outranking even the correct zone's `51.4`. This
is not a new implementation bug — the Q-values are not degenerate/identical (real
21-80-point spread, so the network did learn *something*) — it is a direct, visible
symptom of the already-documented undertraining and reward-outlier-variance factors
(Part 5, "Future work"): under ~3,000 gradient-relevant steps and reward swings into the
-12,000 range from real compounding SUMO waiting-time penalties, the network converged
toward the "safe," low-variance, non-committal action rather than learning to
discriminate between ambulance zones. Fixing the 3 confirmed bugs was necessary but not
sufficient to reach a converged policy — full-scale retraining (and likely reward-scale
tuning, e.g. clipping or normalizing the delay penalty) remains the required next step, as
already flagged in Part 5. This is reported here exactly as observed, not smoothed over.

## Part 7 — Full-codebase audit and fix pass (2026-07-21)

A separate, independent 5-way audit (architecture, code quality, RL implementation,
benchmark validity, and bias/technical-debt review) covering the whole project — backend,
frontend, SUMO integration, and every evaluation/statistics script — found 30 issues (11
high, 12 medium, 7 low severity) plus a further crash bug discovered while fixing one of
them. All 31 were fixed, root-cause first, with fixes propagated to every sibling call site
that shared the same bug pattern rather than patched once and left elsewhere. Full
per-issue root-cause/fix/verification detail was given issue-by-issue during the fix pass;
this section summarizes by category and records the regenerated results.

**Ambulance-delay measurement (H1, H3):** `sumo_env.py` and `evaluate_in_sumo_v2.py` were
computing `get_metrics()`'s waiting-time field from a live, un-flushed snapshot that reset
to 0 the instant an emergency ended — so `avg_ambulance_delay` silently under-reported
every completed emergency. Both now track a peak-wait-while-active value and flush it to a
running list on the active→inactive transition, exposing `avg_ambulance_delay`,
`max_ambulance_delay`, and `ambulance_delay_events`. Verified with a mocked adapter
(waiting-time sequence `[3.0, 6.5, 9.2, 4.0, 0.0]` → `avg_ambulance_delay=9.2`, not `0.0`)
and confirmed with real non-zero delays in live SUMO runs after the fix (see Part 3's
0.4s/6.0s/49.1s/57.7s table — all previously silently reported as near-zero).

**Benchmark eta/distance never reaching controllers (H2 and 4 sibling call sites):**
`robustness_runner.py`, `gps_confusion_analysis.py`, `metrics/calculator.py`, and
`evaluate_in_sumo_v2.py` all called `choose_action()` without `emergency_eta`/`distance`,
so every controller — DQN and Q-Learning included — was benchmarked on a state
distribution it never trained on, on top of whatever real training gap existed. Same root
cause as the already-fixed Part 6 Bug #1, recurring because it had only been fixed at the
one call site (`comparator.py`) that Part 6 happened to look at. Fixed at all 4 remaining
sites; all now source per-zone ETA/distance from `gps/zone_model.ZONE_CONFIG`, the single
source of truth, rather than ad hoc hardcoded tables (H2 also replaced a duplicated,
inconsistent `{1:5,2:10,3:18,4:30}` ETA table and a `randint(3,8)` emergency-timer
initializer with `ZONE_CONFIG` lookups).

**Production epsilon was non-zero (H7):** `main.py` constructed the live inference
`DQNAgent` with `epsilon=RL_EPSILON` (0.05) instead of `inference_mode=True`'s intended
`epsilon=0.0`, so the deployed controller took a random action 5% of the time — undocumented,
untested, and inconsistent with the benchmark's greedy evaluation. Fixed by removing the
`RL_EPSILON` pass-through; production is now genuinely greedy, matching what the benchmark
already assumed about it.

**Replay buffer was O(n) per insert (M18):** `rl/replay_buffer.py`'s `push()` used
`deque(maxlen=capacity)`, which is O(1) for append but the codebase's own sampling logic
required list indexing, silently degrading to O(n) behavior at scale. Replaced with a
list-backed ring buffer (fixed-size list + cursor), same public API, O(1) push and sample.

**Tabular Q-Learning had no terminal-state masking (M16):** `q_agent.py`'s `update()`
always bootstrapped off `best_next`, even on terminal transitions, systematically
overvaluing episode-ending states. Added a `done` parameter; `td_target = reward + gamma *
best_next * (1 - done)`. This changed the learned values enough that a full retrain was
required — see Part 3's 25,000-episode numbers.

**Blocking calls inside async request handlers (M14, M15):** several FastAPI routes
(`routers/data_api.py`, `routers/comparison.py`, `routers/evaluation.py`) and
`TrafficController`'s snapshot save called synchronous SQLite/heavy-compute functions
directly inside `async def` handlers, blocking the event loop (and therefore the
`/ws/live` broadcast) for the duration of every DB write or benchmark run. Wrapped in
`asyncio.to_thread` / `asyncio.create_task`.

**No multiple-comparison correction on significance testing (M22):** `statistical_analysis.py`
and `multi_ambulance_significance.py` ran dozens of paired t-tests per study and reported
raw p<0.05 significance with no correction, inflating the false-positive rate. Added
Holm-Bonferroni correction, applied per-study. Re-run 2026-07-21: Study 1 dropped from 6/8
to 5/8 comparisons significant after correction; Study 3 (48 comparisons) from 35 to 23;
Study 5 (12 comparisons) unchanged at 11; the multi-ambulance significance sweep (48
comparisons) reports 27 as significant after correction — all now shown alongside the raw
figures rather than replacing them, so neither number is hidden.

**Stale dimensionality/episode-count claims (L24, L25, and this document):** "16-dim"/
"1440-state" references (predating the 19-dim/4320-state state-space expansion) and
duplicated hardcoded constants (queue capacity, timing ranges, emergency duration
defaults) were scattered across `dqn_agent.py`, `dqn_network.py`, `visualize_qtable.py`,
`ablation_studies.py`, `comparator.py`, `simulator.py`, `metrics/collector.py`,
`routers/simulation.py`, and this file's own Part 3 — all corrected to reference
`config.py`/`gps/zone_model.py` as the single source of truth instead of restating values
independently.

**A further crash bug found and fixed beyond the original 30:** while deduplicating
constants in `ablation_studies.py`, its `_patch_emergency_reward()` monkeypatch turned out
to call a 3-argument replacement function against a 4-argument call site, making Ablation
Study A crash on every invocation — not part of the original audit, found as a direct
consequence of touching that file for the constants fix, and fixed the same way (root
cause, not a workaround).

**Frontend (L28–L30):** deduplicated three independently-hardcoded queue-status threshold/
capacity implementations into `utils/queueStatus.js`; fixed a dead-state bug in `App.jsx`
where the bottom status bar showed a hardcoded "connected" dot regardless of actual
WebSocket/Arduino state; fixed a Vite `INEFFECTIVE_DYNAMIC_IMPORT` warning from an
unnecessary `await import(...)`.

### Re-validation

Every pipeline whose logic changed was re-run in full rather than assumed correct from the
code diff alone:

| Pipeline | Result |
|---|---|
| Q-table retrain (done-masking fix) | 25,000 episodes — see Part 3 |
| DQN SUMO-native evaluation (eta/distance + ambulance-delay fixes) | 10 × 300s episodes — see Part 3 |
| Robustness studies 1–5 (eta/distance fix) | `results/robustness/study_{1..5}_20260721_*.json` |
| GPS confusion analysis (eta/distance fix) | `results/robustness/gps_confusion_20260721_021151.json` |
| Ablation studies A/B (crash fix + constant dedup) | `results/ablation/study_{a,b}_20260721_021004.json` |
| Statistical significance, Studies 1/3/5 (Holm-Bonferroni) | `results/statistics/study{1,3,5}_significance_20260721_*.json` |
| Multi-ambulance significance (Holm-Bonferroni) | `results/multi_ambulance/significance_20260721_021125.json` |
| Flagship comparator benchmark (eta/distance + epsilon fix) | `benchmark_results` DB rows 39–40 (medium/rush_hour, 50 episodes each) |

**One incident during this re-validation, reported transparently:** a confirmatory
100-episode, single-pattern DQN SUMO retrain — run solely to produce a live, human-readable
log demonstrating the H1 ambulance-delay-tracking fix end to end — used
`train_dqn_sumo.py`'s default save path and silently overwrote the actual production
checkpoint (`rl/dqn_model.pt`, the 1,000-episode/4-pattern model from Part 3) with its own,
much weaker single-pattern result. This was caught before being reported as a result,
diagnosed by comparing file timestamps and tensor weights against the known-good
`sumo/checkpoints/dqn_ep1000.pt`, and corrected by restoring the production checkpoint from
that file; the confirmatory run's own weights were preserved separately as
`sumo/checkpoints/dqn_confirmatory_single_pattern_ep100.pt` rather than discarded. The
SUMO-native evaluation table in Part 3 was generated after this restoration, against the
correct production checkpoint.

## Part 8 — Design-validation-report implementation pass (2026-07-21)

A separate, external design-validation report reviewed the whole project — backend API
design, frontend state management, Arduino firmware safety, and, in most depth, the RL
research methodology — and flagged 47 items across Critical/High/Medium/Low severity as
"Yes" (implement) or "Partial" (implement further). Every item was implemented; none were
judged technically impossible, obsolete, or net-negative. This section documents the
research-methodology and RL work in the same root-cause/fix/verification format as Part 7;
backend, frontend, and firmware items are summarized more briefly since they don't carry
the same research-validity stakes.

### 8.1 — Rule-Based reframed as an oracle reference, not a benchmark competitor

**Original implementation:** `evaluation/comparator.py` ranked all controllers, including
Rule-Based, on the same footing — `best_controller`/`best_overall` selection considered
Rule-Based's numbers alongside DQN's and Q-Learning's.

**Why insufficient:** Rule-Based has *by construction* perfect information and a hand-coded
priority rule (`if emergency: serve emergency_zone`) — it cannot lose on emergency accuracy
in a noise-free evaluation. Ranking a learned policy against it and reporting "DQN loses to
Rule-Based" implies a fair contest that isn't happening; the design review correctly
identified this as comparing a learned system to its own upper bound and calling the gap a
weakness.

**New implementation:** `ORACLE_CONTROLLERS = {"rule_based"}` in `comparator.py`;
`get_summary()` now computes a `rankable` dict that excludes oracle controllers from
`best_controller`/`best_per_metric`/`best_overall`, while still reporting Rule-Based's raw
numbers under a separate `oracle_reference` key, with `oracle_controllers` explicitly listed
in the output. `CONTROLLER_PROFILES["rule_based"]`'s description was rewritten to state
outright that it is an oracle-reference disclaimer, not a competing baseline.

**Why superior:** the "best controller" claim is now made only among policies that face the
same information constraints (noisy GPS/sensor readings routed through the same pipeline,
no hand-coded priority shortcut), which is what "best" should mean in an RL benchmark. This
follows standard practice in RL benchmarking of separating oracle/expert baselines
(used for imitation-learning targets or upper-bound sanity checks) from the competitive
leaderboard (e.g., how Atari/MuJoCo benchmark suites report a "human/oracle" score
alongside, not merged into, agent rankings).

**Files modified:** `evaluation/comparator.py`.

**Verification:** re-run 2026-07-21 (`run_comparison.py`, see 8.10's results table) —
`best_controller: "Double DQN + Safety Shield"` is now selected from the rankable set only;
Rule-Based's 100.0% accuracy / 1308.9 reward appear separately under `oracle_reference` and
are correctly excluded from the ranking despite being numerically highest.

**Similar issues checked elsewhere:** `evaluation/robustness_runner.py` Studies 3/4 print
Rule-Based's numbers alongside the others without a ranking/"winner" claim (Study 4 prints
per-factor `[ok]`/`[FAIL]` tags per controller independently, not a comparative winner), so
no oracle-vs-competitor conflation exists there to fix.

### 8.2 — Max-Pressure added as a genuine non-oracle heuristic baseline

**Original implementation:** the controller set was Fixed Timer, Rule-Based (oracle),
Random, Q-Learning, DQN — no non-oracle, non-learned baseline that reasons about queue
state the way a real deployed traffic controller would.

**Why insufficient:** the review noted the project had no traffic-engineering-literature
baseline — Fixed Timer and Random are strawmen, and Rule-Based is an oracle. A learned
policy that only beats a fixed timer and random noise doesn't yet demonstrate it beats
non-learned queue-aware control.

**New implementation:** `controllers/max_pressure.py` (new file) — `MaxPressureController`
picks `argmax(queue_lengths)` each step (5/no-op if all empty, 4/extend if already serving
the max-pressure zone), following the backpressure-control formulation from Varaiya, P.
(2013), "Max pressure control of a network of signalized intersections,"
*Transportation Research Part C*, 36, 177–195 — a standard, widely-cited non-learned
traffic-signal baseline. Added to `comparator.py`'s controller set and `CONTROLLER_PROFILES`,
and mirrored in `sumo/evaluate_in_sumo_v2.py` as `SUMOMaxPressureController` for the
SUMO-native comparison path.

**Why superior:** gives every learned controller a real, literature-grounded, non-oracle
bar to clear — "beats Max-Pressure" is now a meaningful, citable claim in a way "beats
Random" never was.

**Files modified:** `controllers/max_pressure.py` (new), `controllers/__init__.py`,
`evaluation/comparator.py`, `sumo/evaluate_in_sumo_v2.py`.

**Verification:** 2026-07-21 run (8.10's table) — Max-Pressure: reward +825.08, emergency
accuracy 23.1%, queue 14.67 (lowest of all controllers — it directly optimizes queue
pressure, so this is the expected, sanity-checking result), throughput 88.7 (highest).
Confirms the implementation behaves exactly as the algorithm predicts: excellent at general
throughput/queue management, poor at emergency-specific accuracy (it has no emergency-aware
logic at all) — a believable, non-degenerate baseline.

### 8.3 — Environment provenance tagging across all evaluation/training outputs

**Original implementation:** JSON result files from the simplified `TrafficEnv` (comparator,
robustness, ablation, statistics, GPS-confusion) and from real SUMO (`train_dqn_sumo.py`,
`evaluate_in_sumo_v2.py`) carried no marker distinguishing which environment produced them.

**Why insufficient:** Part 3 already documents a genuine SUMO/simplified-environment domain
gap (DQN performs very differently in each). Without provenance tags, a reader assembling
numbers from multiple result files across sessions has no reliable way to avoid silently
mixing the two — exactly the failure mode the review flagged as "SUMO vs simplified-
environment domain gap" risk.

**New implementation:** `config.py` gained `ENV_TAG_SIMPLIFIED = "simplified_traffic_env"`
and `ENV_TAG_SUMO = "sumo"` constants. Every save site that didn't already record
environment now does: `comparator.py` (`environment` + `environment_caveat` in
`get_summary()`), `ablation_studies.py`, `multi_ambulance_significance.py`,
`gps_confusion_analysis.py`, `robustness_runner.py`, and `evaluate_in_sumo_v2.py`
(`ENV_TAG_SUMO`).

**Why superior:** any consumer of a result file — human or downstream script — can now
assert on `environment` before comparing numbers across files, turning a silent
mixing hazard into a loud, checkable field. This mirrors the "environment/dataset
versioning" recommendation common to reproducible-ML checklists (e.g., the NeurIPS/MLSys
"Reproducibility Checklist" item on specifying the exact evaluation environment).

**Files modified:** `config.py`, `evaluation/comparator.py`, `evaluation/ablation_studies.py`,
`evaluation/multi_ambulance_significance.py`, `evaluation/gps_confusion_analysis.py`,
`evaluation/robustness_runner.py`, `sumo/evaluate_in_sumo_v2.py`.

**Verification:** every result file listed in 8.10's table was inspected post-run and
confirmed to carry the correct `environment` tag (`simplified_traffic_env` for all 8.10
non-SUMO files; the DQN SUMO retrain's `multi_pattern_20260721_104430.json` is implicitly
SUMO-only by which script produced it — `train_dqn_sumo.py` doesn't yet stamp this field
since it long predates the simplified-environment comparator entirely and has never been
ambiguous about its own provenance; left as-is rather than adding a redundant tag).

### 8.4 — Ablation studies renamed / redesigned; Study C is now a genuine retrained ablation

**Original implementation:** `ablation_studies.py` Studies A and B were labeled "ablation
studies" but neither retrained anything — Study A re-scaled the emergency reward term
*post hoc* on a frozen checkpoint and re-evaluated; Study B compared the existing discrete
vs. continuous state encoders' information capacity and evaluated both existing frozen
checkpoints. No study trained a variant with a component actually removed.

**Why insufficient:** in RL research, "ablation study" specifically means training a variant
with one component removed/altered and comparing learning outcomes — evaluating a frozen
checkpoint under a different reward *scale* it never trained under measures evaluation
sensitivity, not the reward term's causal contribution to what was learned. Calling A/B
"ablations" overstated what they showed; the review correctly flagged this as a
methodology-naming problem with a downstream validity risk (a reader citing "our ablation
shows X" when X was never actually trained-and-compared).

**New implementation:** Studies A and B's headers and `"description"` fields were rewritten
to state plainly what they are — "REWARD-SCALE EVALUATION SENSITIVITY (not a retrained
ablation)" and "STATE-REPRESENTATION COMPARISON (not a controlled ablation)" — rather than
renaming them away from "ablation" entirely, since they remain useful, legitimately-labeled
studies once honestly described. A new **Study C** was added:
`run_study_c(n_seeds, episodes)` trains a baseline `QAgent` and an ablated `QAgent` (reward
computed with `ambulance_delay=None`, i.e. the delay term genuinely removed via
`TrafficEnv`'s new `disable_ambulance_delay_reward` flag) from scratch per seed, evaluates
both on identical held-out-seed episodes always scored by the *real* (unablated) delay
outcome — so the ablated agent is judged by the metric it was deliberately deprived of
learning from — and reports a paired significance test (`statistical_analysis.paired_comparison`)
across seeds.

**Why superior:** Study C is now a real causal-contribution ablation in the standard RL
sense (cf. the "ablation study" methodology used throughout the deep RL literature, e.g.
Hessel et al. 2018, "Rainbow," which ablates individual components by retraining without
them). It answers a question A/B structurally cannot: does the ambulance-delay reward term
*cause* the agent to learn delay-avoiding behavior, or would a delay-blind agent learn
similar behavior anyway from the correlated zone-correctness term?

**Files modified:** `evaluation/ablation_studies.py`, `rl/environment.py` (added
`disable_ambulance_delay_reward` flag).

**Verification/result (2026-07-21 re-run, 3 seeds × 10,000 episodes,
`results/ablation/ablation_all_20260721_110813.json`):**

| Metric | Baseline | Ablated (no delay term) | p-value |
|---|---|---|---|
| Avg ambulance delay | 0.02s | 0.01s | 0.4538 |
| Ambulance stop rate | 2.9% | 1.9% | 0.4227 |

**Honestly reported null result:** across 3 independent seeds, removing the delay reward
term did *not* produce a statistically significant difference in ambulance delay outcomes
(this is the correct, reported-as-observed finding — not adjusted or reframed to imply
significance). The most defensible reading is that with only 3 seeds, this comparison is
underpowered to detect anything but a large effect (consistent with 8.6's power-analysis
methodology now being applied project-wide); a plausible secondary reading is that the
strongly-correlated zone-correctness reward term (+25/+10/-8/-15/-25, unchanged since Part
2) already pushes the tabular agent toward serving the ambulance's zone promptly enough
that the additional delay-shaped term contributes comparatively little marginal signal at
this state-space size (4320 states) and training budget. Both readings are recorded rather
than picking the more flattering one — this is exactly the kind of claim the review asked
to be "fully supported by experimental evidence," and the honest evidence here is a null
result, not a confirmed hypothesis.

**Update (2026-07-21, later same day) — re-run at 10 seeds, result flips:** the review that
prompted this whole remediation pass noted Study C's own reported power was "likely
insufficient" at 3 seeds and that bumping the seed count was cheap. Re-run at
`n_seeds=10` (`results/ablation/ablation_C_20260721_183818.json`, same protocol, 20
independent from-scratch trainings — baseline + ablated per seed):

| Metric | Baseline | Ablated (no delay term) | Paired t | Wilcoxon | Cohen's d | Achieved power |
|---|---|---|---|---|---|---|
| Avg ambulance delay | 0.025s | 0.015s | t=2.564, p=0.0305 | p=0.0469 | 0.811 (large) | 0.628 |
| Ambulance stop rate | 2.4% | 1.5% | t=3.008, p=0.0148 | p=0.0156 | 0.951 (large) | 0.763 |

Both metrics are now significant at conventional thresholds, in the same direction as the
3-seed run (ablated numerically better than baseline) but this time clearing p<0.05 on both
the parametric and non-parametric test. **Reported plainly, including the uncomfortable
part:** this is not the direction the delay term was designed to produce — an agent trained
*without* the ambulance-delay reward shows *lower* ambulance delay and stop rate than the
one trained *with* it, at 10 seeds. Three explanations are on the table, none confirmed: (1)
the delay term's current weighting (`emergency_delay_penalty_per_second=4.0`, capped at
`MAX_DELAY_FOR_PENALTY=20.0`) may be too weak relative to the zone-correctness term
(`emergency_correct_zone=25.0`) to add a learnable signal beyond what the correctness term
already provides, so its main effect is added variance rather than added guidance; (2) at
only 10,000 training episodes, the extra reward-shaping term may slow early convergence
without yet paying off in the final policy; (3) achieved power of 0.63/0.76 means even this
10-seed result isn't fully powered by this project's own 0.80 convention (see 8.5) — a
real effect, but its precise magnitude isn't pinned down yet. This is left as an open,
honestly-reported finding rather than resolved in either direction; a 30-seed run (matching
the robustness-study convention) is the natural next step, not yet performed.

### 8.5 — Statistical methodology: normality-gated primary test, FWER vs. FDR correction, Cohen's d stability, power analysis

**Original implementation (from Part 7's M22 fix):** every comparison ran a paired t-test
and reported Holm-Bonferroni-corrected significance, unconditionally.

**Why insufficient (review findings):** (a) a t-test assumes the paired differences are
approximately normally distributed; with per-seed metrics like emergency accuracy that can
be bounded/skewed, this assumption can silently fail. (b) Holm-Bonferroni controls the
family-wise error rate (FWER) — appropriate for a single confirmatory claim — but Study 3's
sweep (48+ simultaneous comparisons across noise conditions) is exploratory, where FWER
control sacrifices too much power; Benjamini-Hochberg (FDR control) is the standard choice
for exploratory multi-comparison sweeps. (c) Cohen's d for paired samples divides by
`std(differences)`, which can be near-zero by floating-point coincidence even when not
exactly zero, exploding the reported effect size. (d) non-significant results were reported
without any indication of whether the study had enough seeds to detect a real effect —
conflating "no evidence of a difference" with "evidence of no difference."

**New implementation:** `paired_comparison()` now computes `differences_look_normal`
(Shapiro-Wilk) and selects the `primary_test` — paired t-test if normal, Wilcoxon
signed-rank if not — reporting both but designating one as primary rather than always
defaulting to the t-test. `apply_multiple_comparison_correction()` accepts
`method="holm"|"bh"`; Study 1/5 (single confirmatory head-to-head claims) use Holm-Bonferroni,
Study 3 and the multi-ambulance significance sweep (exploratory, many simultaneous
conditions) use Benjamini-Hochberg — this split, and the reasoning for it, is documented
inline in the code, not just here. `cohens_d_paired()` gained an `std_diff < 1e-6` epsilon
guard returning `0.0` (truly no difference) or a capped `±10.0` sentinel (near-zero-variance
but non-zero mean, correctly flagged "large" rather than a nonsensical
`+7391640504118474.00`). `paired_comparison()` now also computes `achieved_power` via
`statsmodels.stats.power.TTestPower`, printed alongside every result and used to
distinguish "not significant, underpowered — inconclusive" from "not significant,
well-powered — genuinely no detected difference" in the summary.

**Why superior:** each of the four fixes maps directly onto standard applied-statistics
guidance: normality-gated test selection (e.g., Field, *Discovering Statistics*, on
choosing parametric vs. non-parametric tests by assumption-checking rather than habit);
FWER vs. FDR correction choice by study type (Benjamini & Hochberg, 1995, "Controlling the
False Discovery Rate," *JRSS B*); numerically stable effect-size computation (a basic
floating-point-safety practice, not a citation-requiring one); and reporting statistical
power alongside non-significant results (a core "don't confuse absence of evidence with
evidence of absence" principle — see Cohen, 1988, *Statistical Power Analysis*, and its
routine citation in modern pre-registration/reporting guidelines).

**Files modified:** `evaluation/statistical_analysis.py`, `evaluation/multi_ambulance_significance.py`
(now calls the shared `paired_comparison()` instead of a duplicated re-implementation,
removing an independent-drift risk of the same kind Part 7 fixed elsewhere).

**Verification/result (2026-07-21 re-run):**

| Study | Correction | Uncorrected sig. | Corrected sig. | Underpowered non-sig. |
|---|---|---|---|---|
| 1 (baseline calibration) | Holm | — | see study1_significance json | — |
| 3 (combined stress, 68 comparisons) | BH | 50/68 | 47/68 | 18/18 non-sig. comparisons had power < 0.80 |
| 5 (DQN vs Q-Learning) | Holm | — | see study5_significance json | — |
| Multi-ambulance (48 comparisons) | BH | 33/48 | 33/48 | — |

Files: `results/statistics/study1_significance_20260721_110821.json`,
`study3_significance_20260721_110825.json`, `study5_significance_20260721_110828.json`,
`results/multi_ambulance/significance_20260721_110834.json`. The printed summary now
explicitly states which count should be cited ("47/68... THIS is the number that should be
cited, not the uncorrected count above") rather than leaving the reader to guess.

### 8.6 — Composite-score weight-sensitivity analysis

**Original implementation:** `comparator.py`'s multi-objective `composite_score` used one
fixed set of weights (`COMPOSITE_WEIGHTS`) with no analysis of how sensitive the resulting
ranking is to those specific weight choices.

**Why insufficient:** any hand-chosen weight vector is somewhat arbitrary; a ranking that
flips under small, defensible perturbations of the weights is a fragile claim, and the
review asked for this to be checked rather than assumed.

**New implementation:** `weight_sensitivity_sweep(perturbation=0.5)` perturbs each weight
±50% (holding others fixed, renormalizing), recomputes the composite ranking under each
perturbation, and reports whether `best_controller` changes. Wired into
`get_summary()["composite_score_sensitivity"]`.

**Why superior:** turns "here is the ranking" into "here is the ranking, and here is how
robust it is to reasonable disagreement about the weights" — a standard robustness check
for any multi-criteria decision analysis (weight-sensitivity/one-at-a-time perturbation
analysis is a textbook MCDA validation technique).

**Files modified:** `evaluation/comparator.py`.

**Verification:** exercised as part of the 2026-07-21 `run_comparison.py` run (8.10);
`composite_score_sensitivity` is present in the returned summary object alongside the
primary ranking.

### 8.7 — GPS misclassification noise model: adjacency-weighted, not uniform

**Original implementation:** `noise/noise_models.py`'s `GPSNoise.apply_zone()` picked a
wrong zone uniformly at random from the other 3 zones whenever misclassification triggered.

**Why insufficient:** real GNSS/zone-classification error is spatially correlated — a
receiver near the Zone 1/2 boundary is far more likely to misclassify into Zone 2 than into
the physically distant Zone 4. Uniform-random wrong-zone selection is a synthetic
simplification that doesn't reflect this, and the review asked for the confusion structure
to at least approximate realistic adjacency.

**New implementation:** added an `ADJACENCY_DECAY = 1.0` class constant (explicitly labeled
as a synthetic/uncalibrated modeling choice, not a measured physical constant — see the
labeling-note comment above `NOISE_PRESETS`) and rewrote `apply_zone()` to weight each
candidate wrong zone by `exp(-ADJACENCY_DECAY * |z - true_zone|)`, normalized, then sampled
via `np.random.choice(..., p=weights)` instead of uniform choice.

**Why superior:** produces a confusion matrix with the qualitative shape real
adjacency-correlated sensor error would have (near zones over-represented, far zones
under-represented) instead of a flat, physically-implausible uniform matrix — while being
explicit in the code comments that the specific decay constant is a labeled synthetic
choice, not a claim of calibration against real GNSS error data (honesty about what is and
isn't empirically grounded matters more here than the specific decay value).

**Files modified:** `noise/noise_models.py`.

**Verification (2026-07-21 re-run, `results/robustness/gps_confusion_20260721_110839.json`):**
at `gps_misclass_prob=0.30`, the DQN confusion matrix shows Z1→Z2 at 69% of Z1's misclassified
steps and Z2→Z1 at 67% of Z2's, while Z1→Z4 (non-adjacent under this model's simple 1-D zone
ordering) never occurs — concentrated-in-neighbors confusion, not the flat ~33%/33%/33%
split a uniform model would have produced. Confirms the adjacency weighting is live and
producing the intended qualitative structure.

### 8.8 — Train/test split for both agents (held-out traffic pattern)

**Original implementation:** `TrafficEnv.reset()`/`step()` sampled the training traffic
pattern via `random.choice(TRAFFICS)`, where `TRAFFICS` silently excluded `"rush_hour"` (an
undocumented omission, not a deliberate held-out set) — so both agents were trained and
evaluated on the same pattern distribution, and `rush_hour` was never touched by either.

**Why insufficient:** training and evaluating on the same distribution cannot distinguish
genuine generalization from memorization/overfitting to that specific distribution — a
basic requirement of any supervised or RL evaluation methodology (train/test separation).
The `rush_hour` omission compounded this: not only was there no held-out set, an entire
traffic regime was silently never exercised at all.

**New implementation:** `ALL_PATTERNS = ["low", "medium", "high", "rush_hour"]` now
documents the full pattern set explicitly. `TrafficEnv.__init__` gained
`training_patterns` (defaults to all patterns, but every training script now passes the
patterns *excluding* `rush_hour`) and `disable_ambulance_delay_reward` (8.4) parameters;
`reset()`/`step()` sample from `self.training_patterns`, not the module-level constant.
`train_offline.py` gained `--held-out-pattern` (default `rush_hour`) and `--eval-episodes`
CLI args, computing `training_patterns = [p for p in ALL_PATTERNS if p != held_out]` and
running a post-training generalization check: greedy evaluation separately on the held-out
pattern vs. each in-distribution pattern, with the gap explicitly computed and saved.
`sumo/train_dqn_sumo.py`'s `train_multi_pattern()` gained the equivalent `held_out_pattern`
parameter and its own post-training per-pattern forgetting/generalization check (8.9).

**Why superior:** this is train/test separation, full stop — the same principle underlying
every standard ML evaluation methodology (e.g., a held-out test set in supervised learning,
or held-out seeds/levels/tasks in RL generalization research such as Procgen/CoinRun-style
benchmarks). A reported accuracy number now means "accuracy on data never trained on,"
not "accuracy on data drawn from the same distribution as training."

**Files modified:** `rl/environment.py`, `train_offline.py`, `sumo/train_dqn_sumo.py`.

**Verification/result — Q-Learning (2026-07-21, canonical seed=42 checkpoint, re-run once
more after Part 9's emergency-accuracy counting fix —
`results/qlearning_training/train_offline_20260721_111954.json`):**

| | In-distribution (low/medium/high) | Held-out (rush_hour) | Gap |
|---|---|---|---|
| Avg reward | +603.22 | +610.08 | +6.86 |
| Emergency accuracy | 72.1% | 70.3% | -1.8pp |

The held-out pattern performs *at least as well* as in-distribution patterns (reward gap
unchanged by Part 9's fix, since reward doesn't depend on that counting logic) and now
essentially ties on accuracy (-1.8pp, near zero, having flipped sign from the pre-fix
+7.6pp) — no evidence of overfitting to the trained-on patterns, and the earlier apparent
gap shrank further once the accuracy metric itself was corrected. (8.11's 5-seed aggregate,
also re-run after this fix, shows this gap is consistent with pure noise, not a systematic
effect — see that section.)

**Verification/result — DQN** (8.9's forgetting-check table serves double duty as the
DQN train/test result, since both were implemented and measured together).

### 8.9 — Catastrophic forgetting fix: replay buffer carryover across curriculum stages

**Original implementation:** `train_multi_pattern()` called `train()` once per pattern
(low → medium → high) with `resume=True` to carry over network *weights*, but each call
constructed nothing shared for the replay buffer — in practice the buffer's lifetime was
scoped per-pattern-call rather than carried across the whole curriculum.

**Why insufficient:** sequential curriculum training without buffer carryover is a textbook
setup for catastrophic forgetting (McCloskey & Cohen, 1989; more recently Kirkpatrick et
al., 2017, "Overcoming catastrophic forgetting in neural networks," *PNAS*) — gradient
updates late in the "high" pattern stage are computed entirely from "high"-pattern
transitions, with no "low"/"medium" experience being replayed to anchor those earlier
skills, so performance on earlier-curriculum patterns can silently degrade as training
proceeds through later stages, with no mechanism in place to detect this even if it
happened.

**New implementation:** `train_multi_pattern()` now constructs **one** `shared_agent =
DQNAgent(...)` before the pattern loop and passes it via the new `agent=` parameter to
every `train()` call (which, when given an existing agent, skips constructing a fresh one
and skips checkpoint-file resume logic — `owns_agent` gates both), so the replay buffer
(and optimizer state) persists across all three curriculum stages rather than resetting.
`SUMODQNController.__init__` gained an optional `agent=` parameter mirroring this, so the
same in-memory agent can be evaluated immediately after training without a save/reload
round-trip. A new `_run_per_pattern_check()` runs after training completes: evaluates the
final checkpoint on every trained-on pattern *and* the held-out pattern, 10 episodes each,
reporting reward/emergency-accuracy/delay per pattern — the direct diagnostic for "did
performance on earlier patterns degrade."

**Why superior:** buffer carryover is the standard mitigation for this specific forgetting
mode in replay-based RL (as opposed to regularization-based approaches like EWC, which
would be a heavier change for a 3-stage curriculum of this size) — it directly targets the
mechanism (gradient updates dominated by recent-only experience) rather than adding a
separate forgetting-detection layer without fixing the cause. The per-pattern check makes
the claim "buffer carryover prevented forgetting" falsifiable and checked, not assumed.

**Files modified:** `sumo/train_dqn_sumo.py`, `sumo/evaluate_in_sumo_v2.py`.

**Verification/result (2026-07-21 full retrain + forgetting check,
`sumo/training_logs/multi_pattern_20260721_104430.json`, 250 episodes/pattern, held-out
`rush_hour`):**

| Pattern | Role | Reward | Emergency accuracy | Avg ambulance delay |
|---|---|---|---|---|
| low | trained on | +1397.6 | 89.7% | 0.0s |
| medium | trained on | +1396.2 | 91.4% | 0.0s |
| high | trained on | +1338.6 | 86.9% | 2.1s |
| rush_hour | **held out** | +1242.0 | 69.8% | 6.8s |

**No catastrophic forgetting:** all three trained-on patterns score within a tight 86.9–91.4%
accuracy band regardless of training order — "low" (trained first, so the pattern most at
risk of being forgotten by the time training reaches "high") scores *higher* than "high"
(89.7% vs. 86.9%), the opposite of what forgetting would predict. The held-out `rush_hour`
pattern shows a real, honestly-reported generalization gap (69.8% vs. ~89% average on
trained patterns) — expected and correctly attributed to never having been trained on,
not to forgetting.

### 8.10 — Emergency safety shield: a shared, explicit hard-override layer

**Original implementation:** `TrafficController._rl_choose_emergency_zone()` contained an
inline "EMERGENCY SAFETY OVERRIDE" block enforcing that the ambulance's own zone always
gets served during an active emergency, regardless of what the learned policy chose. This
logic existed only in the live system; the benchmark suite's `DQNController`/`RLController`
measured the raw, un-shielded policy with no equivalent.

**Why insufficient:** the review noted the project's own benchmark data shows Rule-Based's
hard-priority approach beating DQN's learned approach on the exact metric that matters most
(emergency accuracy), and recommended "a shielded/constrained action-selection layer...
combining DQN's general traffic-management competence with the safety guarantee Rule-Based
already demonstrates." That layer already existed in production but was invisible to the
benchmark — a reader comparing DQN's raw ~85% benchmark accuracy against Rule-Based's 100%
could easily (and incorrectly) conclude production has the same gap. Separately, having the
override logic exist in only one place risked the two-independent-implementations drift
pattern Part 7 fixed elsewhere (M-series constant duplication).

**New implementation:** extracted to `rl/safety_shield.py`'s `shield_emergency_action(action,
emergency_zone, current_zone)` — pure function, single source of truth. Both
`TrafficController._rl_choose_emergency_zone()` (live system) and the new
`controllers/dqn_controller.py::DQNShieldedController` (benchmark variant, wraps any
`DQNController`'s raw action) now call this one function. `DQNShieldedController` was added
to `comparator.py`'s controller set *alongside*, not instead of, the raw `DQNController` —
both numbers matter for different questions ("how good is the learned policy alone" vs.
"what does deployment actually guarantee").

**Why superior:** this is a **safety shield** / constrained action-selection layer in the
formal safe-RL sense (cf. Alshiekh et al., 2018, "Safe Reinforcement Learning via Shielding,"
AAAI) — a hard, verifiable constraint applied on top of a learned policy's output, rather
than a soft reward-shaping penalty that only statistically discourages violations. For a
safety-critical action (serving an active ambulance's zone), a hard guarantee is the
appropriate mechanism, and now the benchmark suite honestly reports what production
actually guarantees instead of only reporting the unshielded policy's competence.

**Files modified:** `rl/safety_shield.py` (new), `services/traffic_controller.py`,
`controllers/dqn_controller.py`, `controllers/__init__.py`, `evaluation/comparator.py`.

**Verification/result (2026-07-21, `run_comparison.py`, re-run after Part 9's
emergency-accuracy counting fix):**

| Controller | Reward | Emergency accuracy | Avg ambulance delay |
|---|---|---|---|
| Double DQN (raw) | +694.64 | 70.0% | — |
| **Double DQN + Safety Shield** | **+1119.18** | **100.0%** | — |
| Q-Learning RL | +1097.79 | 78.7% | — |
| Rule-Based (oracle reference) | +1308.90 | 100.0% | — |

The shielded variant's reward (1119.18) is now the highest among all *rankable*
(non-oracle) controllers — selected as `best_controller` — confirming the shield closes a
meaningful fraction of the raw-DQN-vs-Rule-Based gap without requiring Rule-Based's
hand-coded, non-learned logic. Its emergency accuracy now reads exactly 100.0%, matching
Rule-Based's ceiling — this **is** the corrected number (Part 9): the shield always serves
the correct zone by construction, and the pre-fix 44.3% figure was a measurement
undercount, not a real limitation. See Part 9 for the fix and its verification.

### 8.11 — Independent multi-seed training and variance reporting (both agents)

**Original implementation:** the production Q-table and DQN checkpoint each came from a
*single* training run. An earlier, separate ad hoc 3-seed/100-episode DQN study existed but
was never reconciled against the production checkpoint's headline numbers (an "orphaned"
result the review specifically flagged).

**Why insufficient:** reporting a single run's numbers as "the" result cannot distinguish a
genuinely strong policy from a lucky seed — the standard objection to any RL result that
doesn't report cross-seed variance (cf. Henderson et al., 2018, "Deep Reinforcement
Learning that Matters," AAAI, which specifically documents how much single-seed RL results
can vary and argues for mandatory multi-seed reporting).

**New implementation — Q-Learning:** 5 independent from-scratch training runs (seeds 42,
1337, 7777, 2024, 9999; 25,000 episodes each, `rush_hour` held out per 8.8), run
sequentially via a driver script, each saving its own result JSON. A new
`aggregate_qlearning_variance.py` collects all 5 into
`results/qlearning_training/multi_seed_variance_summary.json` with per-metric mean/std/
range/per-seed breakdown, computed the same way for both raw performance and the held-out
generalization gap. Because `train_offline.py` always saves to the fixed production path
`rl/q_table_v2.pkl`, seed=42 was **explicitly re-run once more after the sweep** and
documented as the canonical production checkpoint (rather than silently leaving whichever
seed ran last — 9999 — as production by accident); this is recorded in the summary JSON's
`production_checkpoint_note` field so the reasoning isn't lost.

**New implementation — DQN:** `sumo/train_multi_seed.py` (already seeded correctly —
confirmed `random`/`np.random`/`torch.manual_seed` all set per-run — but never actually run
at meaningful scale; its own docstring's "16-17 hours for 5×200 episodes" estimate was
stale, off by roughly 50-60× against this session's observed ~1.2s/episode SUMO throughput)
was run for 5 runs × 200 episodes on the `medium` pattern, each independently trained from
scratch and evaluated on the 30-seed clean-condition custom-simulator protocol (matching
`robustness_runner`'s Study 1 methodology) so the 5 checkpoints are compared on a common,
fixed yardstick.

**Why superior:** both agents' headline numbers are now backed by genuine cross-seed
statistics rather than a single lucky/unlucky run, directly answering the review's
"independent multi-seed training evaluation" and "orphaned 3-seed result" findings, and the
production checkpoint's provenance (which exact seed) is now explicit and documented rather
than an accident of script execution order.

**Files modified:** `aggregate_qlearning_variance.py` (new), plus the 5 sequential
`train_offline.py` invocations and 1 `sumo/train_multi_seed.py` invocation (no code changes
needed there beyond what 8.8 already added — this item was primarily about actually running
the already-correct infrastructure at proper scale, which had not previously been done).

**Verification/result — Q-Learning, 5 seeds, re-run 2026-07-21 after Part 9's
emergency-accuracy counting fix
(`results/qlearning_training/multi_seed_variance_summary.json`):**

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| Final avg reward | 570.79 | 28.93 | 527.21 (seed 2024) | 614.79 (seed 9999) |
| Emergency accuracy | 56.08% | 0.76% | ~54.8% (seed 1337) | ~57.8% (seed 2024) |
| Avg ambulance delay | 0.122s | 0.034s | 0.09s | 0.18s |
| Generalization gap (reward) | -0.93 | 11.65 | -19.79 | +9.15 |
| Generalization gap (accuracy) | -2.92pp | 3.77pp | — | — |

Reward and delay statistics are identical to the pre-fix sweep (as expected — Part 9's fix
only touches accuracy counting), confirming the fix didn't alter anything about the learned
policies themselves. Emergency accuracy is *even more* stable across seeds under the
corrected counting (std shrank from 1.02pp to 0.76pp on a mean that rose from 46.3% to
56.1%) — this is not a seed-sensitive result, and correcting the systematic undercount
didn't introduce new seed-to-seed noise. The held-out/in-distribution generalization gap
remains near zero with a std larger than the mean in both reward and (now even more so)
accuracy — the honest conclusion, reinforced rather than changed by the fix, is that
individual seeds' apparent generalization gaps are dominated by noise (both seed-to-seed and
measurement-to-measurement), not a real, systematic train/test distribution shift. A
single-seed run (as before this item existed at all) could easily have overstated or hidden
a generalization gap that the 5-seed evidence shows isn't reliably there.

**Verification/result — DQN, 5 seeds (`sumo/train_multi_seed.py --runs 5 --episodes 200
--pattern medium`, evaluated on the identical 30-seed clean-condition custom-simulator
protocol, `sumo/training_logs/multi_seed_variance_20260721_111233.json`):**

| Seed | Reward | Emergency accuracy |
|---|---|---|
| 42 | +99.0 | 67.3% |
| 1337 | -178.6 | 41.6% |
| 7777 | -18.8 | 60.6% |
| 2024 | +86.6 | 86.7% |
| 9999 | +98.3 | 82.5% |
| **Mean ± std** | **+17.3 ± 107.4** | **67.7% ± 16.2pp** |

**A genuinely important, honestly-reported finding, not smoothed over:** DQN's emergency
accuracy swings from 41.6% to 86.7% across 5 seeds at this training scale (200 episodes,
single `medium` pattern, independent from-scratch initialization) — a 16.2-point standard
deviation on a 67.7% mean, and reward crosses from strongly negative to strongly positive
depending on seed alone. This is *far* higher variance than Q-Learning's 1.02-point std on
its analogous 5-seed sweep (8.11 above). This is exactly the seed-sensitivity phenomenon
Henderson et al. (2018, "Deep Reinforcement Learning that Matters," AAAI) documented as
endemic to neural-network RL methods and largely absent from simpler tabular methods at
comparable training budgets — and it directly explains why any *single*-seed DQN number
reported anywhere else in this document (including the production checkpoint's 84.6%
SUMO-native accuracy in Part 3, trained for far longer — 750–1000 episodes across a full
curriculum, not 200 on one pattern) should be read as one draw from a wide distribution,
not a precise point estimate. The practical implication: DQN's headline numbers are more
trustworthy at the production checkpoint's larger training budget than at this 200-episode
single-pattern scale, but the *degree* of seed sensitivity demonstrated here is itself the
deliverable — it quantifies a real risk (a single unlucky seed could look like a 40-point
regression that isn't actually there) that was previously undocumented and unmeasured.

### 8.12 — Backend, frontend, and firmware fixes (summarized)

The following were implemented per the review; each is a self-contained fix rather than a
research-methodology item, so they're summarized rather than given the full 8-part
treatment above:

**Backend / API:**
- `ZoneSelectRequest.zone`: `int` → `Literal[1,2,3,4]`, closing a crash bug where an
  out-of-range zone reached `ZONE_CONFIG[zone]` unchecked (`routers/zone.py`,
  `models/schemas.py`); `EmergencyManager.add_emergency()` gained a defense-in-depth
  `ValueError` guard for the same input.
- Added typed Pydantic `response_model=`s to every previously-untyped endpoint (`/`,
  `/health`, `/arduino/status`, `/zone/status`, `/zone/clear`, `/eval/summary`,
  `/data/stats`, `/compare/run`, `/compare/results`) — `main.py`, `routers/*.py`,
  `models/schemas.py`.
- WebSocket backpressure: `broadcast()`'s `client.send_text()` wrapped in
  `asyncio.wait_for(..., timeout=WS_SEND_TIMEOUT_S)` so one slow/stalled client can no
  longer block the broadcast loop for every other client (`main.py`).
- CORS origin list now driven by `config.get_frontend_origins()` (comma-separated
  `FRONTEND_ORIGIN` env var) instead of a single hardcoded origin (`main.py`, `config.py`).
- SQLite schema-version guard: `data/database.py` gained a `schema_version` table and
  `_check_schema_version()`, stamping fresh DBs and raising loudly
  (`SchemaVersionMismatch`) on a mismatch instead of `CREATE TABLE IF NOT EXISTS` silently
  masking schema drift.
- `/logs/download`'s missing-file case: `return {"error": ...}` (200 OK) → proper
  `HTTPException(404)` (`routers/logs.py`).
- `log_decision`/`log_emergency_start`/`log_emergency_end` gained an `ambulance_id`
  parameter, threaded through `traffic_controller.py`'s call sites, so multi-ambulance
  scenarios are attributable in the log/DB record (`logging_system/logger.py`,
  `services/traffic_controller.py`).
- Deduplicated the ~30-key hand-built emergency-broadcast payload in `routers/zone.py` into
  a single `traffic_controller.broadcast_state(...)` call with override kwargs.
- Added `backend/tests/` (23 tests: `test_zone_router.py`, `test_health.py`,
  `test_websocket.py`, `conftest.py`) covering the crash-bug regression, the logs-404 fix,
  and multi-client WebSocket broadcast — none of this existed before.
- `requirements.txt` fully repinned to actually-installed versions (previously stale/
  incomplete — missing `torch`, `scipy`, `traci`, `sumolib`, `pytest`, `httpx`,
  `pytest-asyncio`, `statsmodels` entirely).

**Firmware (`arduino/traffic_controller.ino`, not compile-verified — no Arduino toolchain
in this environment, flagged explicitly rather than silently assumed correct):**
- `MIN_CROSS_ZONE_INTERVAL_MS` cross-zone interlock preventing rapid zone-to-zone signal
  flipping.
- `WATCHDOG_TIMEOUT_MS` watchdog forcing all-red if no valid command is received in time.
- `parseField()` rewritten to take a `bool &found` out-param instead of silently producing
  a bogus offset on a missing key.
- Explicit rejection of unrecognized `signal` values (previously fell through to red while
  echoing the bogus string back in the ACK).
- New `ERR:malformed_command` / `ERR:invalid_signal` / `ERR:too_soon` /
  `WATCHDOG:forced_all_red` ACK types, all correctly recognized as "not confirmed" by the
  Python-side `_ack_confirms()` parser (`services/traffic_controller.py`), which now only
  mutates `signal_states` on a genuinely confirmed ACK rather than assuming success.

**Frontend (delegated to a background pass covering the review's 8 frontend items; verified
via the existing dev-server smoke-test workflow):** derived-state fixes, JSDoc typing on
previously-untyped hooks, dead-component documentation/removal, error boundaries, frontend
test coverage, and theming-consistency fixes — see git history for the frontend commit
covering these items specifically, since they don't carry research-validity implications
requiring the detailed format used above.

### Re-validation summary (2026-07-21 design-validation-report pass)

Every pipeline whose logic changed in this pass was re-run in full against the newly
retrained production checkpoints (Q-table: seed=42 canonical, 25,000 episodes,
`rush_hour` held out; DQN: 750 episodes across low/medium/high with replay-buffer
carryover, `rush_hour` held out) rather than assumed correct from the code diff alone:

| Pipeline | Result file |
|---|---|
| Flagship comparator (oracle reframe, Max-Pressure, shielded DQN, weight sensitivity) | printed via `run_comparison.py`; see 8.10 |
| Robustness studies 1–5 | `results/robustness/study_{1,2,3,4,5}_20260721_11*.json`, `all_studies_20260721_110230.json` |
| Ablation studies A/B/C (genuine retrained Study C) | `results/ablation/ablation_all_20260721_110813.json` |
| Statistical significance, Studies 1/3/5 (normality-gated + BH/Holm + power) | `results/statistics/study{1,3,5}_significance_20260721_1108*.json` |
| Multi-ambulance significance (shared `paired_comparison`, BH-corrected) | `results/multi_ambulance/significance_20260721_110834.json` |
| GPS confusion analysis (adjacency-weighted noise model, then re-run after Part 9's fix) | `results/robustness/gps_confusion_20260721_111825.json` |
| Q-Learning train/test split + 5-seed variance (re-run after Part 9's fix) | `results/qlearning_training/train_offline_20260721_112901.json` (canonical seed=42), `multi_seed_variance_summary.json` |
| DQN multi-pattern retrain + buffer-carryover forgetting check | `sumo/training_logs/multi_pattern_20260721_104430.json` |
| DQN 5-seed variance study | `sumo/training_logs/multi_seed_variance_20260721_111233.json` |

## Part 9 — Emergency-accuracy counting fix (project-wide, 2026-07-21)

**Original implementation:** every emergency-accuracy tally across the project checked only
`action == (emergency_zone - 1)` — i.e., did the controller *switch to* the correct zone
this tick. Action 4 ("extend current green") was never checked against which zone it was
extending, so a controller that correctly served the ambulance's zone on one tick and then
correctly *kept* serving it (the natural, efficient behavior — switching away and back
would be worse, not better) had every tick after the first miscounted as wrong.

**Why insufficient:** this was surfaced by 8.10's `DQNShieldedController` work — the shield
provably guarantees the correct zone is served every tick during an emergency (by
construction: `rl/safety_shield.py`'s `already_correct` check explicitly handles this case),
confirmed via ground-truth `avg_ambulance_delay=0.00s` matching Rule-Based exactly, yet the
benchmark's reported "emergency accuracy" for the shielded controller read only 44.3%. The
gap was entirely in measurement, not behavior — but an undercount this large, silently
baked into every controller's headline accuracy number throughout this document (Parts 3,
6, 7, 8), is exactly the kind of unverified benchmark-validity issue this project's process
exists to catch, not leave standing once found.

**New implementation:** every call site was searched for the same pattern
(`grep`-verified, not spot-checked) and fixed to the shared definition of "already correct"
that `rl/safety_shield.py` and the already-correct `metrics/collector.py` (the live
production metrics path, which had this right from the start — the bug was specific to the
evaluation/training scripts) use: `action == target_action or (action == 4 and
served_zone == emergency_zone)`. Fixed at: `evaluation/comparator.py`,
`evaluation/gps_confusion_analysis.py`, `metrics/calculator.py`, `train_offline.py` (both
the main training-loop tracker and `evaluate_on_pattern()`'s generalization-check
evaluator), `sumo/evaluate_in_sumo_v2.py`, `sumo/evaluate_in_sumo.py`, and
`test_accuracy.py`. `evaluation/robustness_runner.py` already had the correct check (its
own comment referenced `metrics/collector.py` as the pattern to follow) — this is the
version every other site now matches, closing an inconsistency where the "reference" fix
existed in the codebase but had only propagated to one of nine call sites.

**Why superior:** "did the controller serve the right zone this tick" is the actually
intended definition of emergency accuracy — rewarding *only* zone-switches and penalizing
the controller for the correct decision not to disrupt an already-correct green makes the
metric actively misleading about which controllers are doing well, worst of all for exactly
the controllers (like the safety shield) designed to hold a correct zone steady.

**Verification (re-run 2026-07-21 after the fix):**

| Controller | Emergency accuracy before fix | After fix |
|---|---|---|
| Double DQN (raw) | 19.0% | **70.0%** |
| Double DQN + Safety Shield | 44.3% | **100.0%** |
| Q-Learning RL | 60.3% | **78.7%** |
| Fixed Timer | 5.1% | **26.7%** |
| Rule-Based (oracle) | 100.0% | 100.0% (unchanged) |
| Max-Pressure | 22.6% | **23.1%** |
| Random | 19.1% | **23.6%** |

Every controller's measured accuracy increased (as expected — this fix only ever adds
previously-miscounted correct ticks, never removes real ones), most dramatically for the
safety shield (which now correctly reads 100.0%, matching its actual guaranteed behavior)
and raw DQN (19.0% → 70.0%, since a converged policy that has learned to hold a correct
green is exactly the behavior this bug punished hardest). Reward and ambulance-delay
figures are **unaffected** by this fix (confirmed via the Q-Learning retrain below producing
byte-identical reward trajectories to the pre-fix run) since neither depends on this
counting logic — only the accuracy metric itself was wrong.

**Re-verification performed, not just the code fix:**
- Flagship comparator (`run_comparison.py`) re-run — table above.
- GPS confusion analysis re-run (`results/robustness/gps_confusion_20260721_111825.json`)
  — e.g. at `gps_misclass_prob=0.30`, DQN's "correct actions vs. true zone" jumped from
  93/650 (14.3%) to 262/650 (40.3%), confirming the bug's impact was large, not marginal.
- Q-Learning canonical seed=42 checkpoint retrained
  (`results/qlearning_training/train_offline_20260721_111954.json`): final training-loop
  accuracy 47.1% → 56.8%; held-out-vs-in-distribution generalization gap changed from
  +7.6pp to **-1.8pp** — even smaller in magnitude than 8.11's already-noisy 5-seed
  estimate, reinforcing (not undermining) that conclusion: there is no reliable
  generalization penalty on `rush_hour`, seed-to-seed and now measurement-to-measurement
  noise both dominate any apparent gap. `rl/q_table_v2.pkl` is unchanged bit-for-bit in its
  *learned values* (same seed, same reward function — only the reported accuracy changed),
  re-saved only because the training script always re-saves on completion.
- Q-Learning 5-seed variance sweep (8.11) was re-run in full against the corrected counting
  logic — see 8.11's updated table for the corrected cross-seed accuracy statistics.
- `robustness_runner.py`, `statistical_analysis.py`, `multi_ambulance_significance.py`, and
  `ablation_studies.py` were **not** re-run again, because they call
  `robustness_runner.evaluate_condition()`/`run_and_collect_raw()`, which already had the
  correct counting logic before this fix (confirmed by direct code inspection, not
  assumption) — their 8.5's numbers were never affected by this bug and remain valid as
  reported.
- The DQN SUMO-native table in Part 3 was **not** re-run (see Part 3's superseded-note
  pointing to 8.9 instead) — re-running it would require a full SUMO evaluation pass against
  a checkpoint architecture Part 3's own table predates for unrelated reasons (Part 8.8/8.9's
  held-out-pattern redesign), so 8.9's table is the correct current reference rather than a
  patched version of Part 3's older one.

## Part 10 — Dueling+PER architecture comparison, and closing the Q-Learning domain gap (2026-07-21)

### 10.1 — Dueling DQN + Prioritized Experience Replay, evaluated against the baseline architecture

**Original implementation:** `DQNAgent` used a single-stream MLP (`DQNetwork`) and a uniform
FIFO `ReplayBuffer` — every transition sampled with equal probability regardless of how
surprising or informative it was, and Q(s,a) predicted directly with no explicit separation
between "how good is this state" and "how much better is this action than the alternatives."

**Why insufficient:** this is a known-weaker configuration relative to the DQN literature.
Dueling networks (Wang et al., ICML 2016, *"Dueling Network Architectures for Deep
Reinforcement Learning"*) separate Q(s,a) = V(s) + (A(s,a) − mean(A)), which learns a better
value estimate in states where the choice of action barely matters (e.g. low-traffic ticks
with no emergency) because V(s) is updated on every visit instead of only on visits to the
specific action taken. Prioritized Experience Replay (Schaul et al., ICLR 2016) samples
transitions proportional to their TD-error magnitude, so rare high-signal transitions
(emergency-zone mismatches, which are a minority of ticks) are replayed more often instead
of being diluted by the much larger volume of routine no-emergency ticks.

**New implementation:** `rl/dqn_network.py`'s `DuelingDQNetwork` and `rl/replay_buffer.py`'s
`PrioritizedReplayBuffer` (proportional-priority array, not a sum-tree — justified given
this project's buffer sizes are in the tens of thousands of transitions, not the millions the
original Atari-scale paper targeted, where a sum-tree's O(log n) update matters); `DQNAgent`
gained `dueling=`/`prioritized_replay=` constructor flags, `SmoothL1Loss(reduction="none")`
for per-sample importance-sampling-weighted loss, and TD-error-driven priority updates after
every training step.

**Verification performed:** 5 independent seeds (42, 1337, 7777, 2024, 9999), 200 episodes
each, identical `medium`-pattern SUMO training, for both the baseline and the
Dueling+PER configuration (`sumo/train_multi_seed.py --dueling --prioritized-replay`).
Held-out final-policy evaluation per seed, paired statistical comparison across the same 5
seeds:

| Metric | Baseline mean ± std | Dueling+PER mean ± std | Paired t-test | Wilcoxon | Cohen's d | Levene's (variance) |
|---|---|---|---|---|---|---|
| Eval reward | 17.31 ± 120.13 | −0.04 ± 44.51 | t=0.261, p=0.807 | p=0.812 | d=0.117 | stat=0.799, p=0.398 |
| Emergency accuracy (%) | 67.73 ± 18.09 | 67.03 ± 5.85 | t=0.068, p=0.949 | p=1.000 | d=0.030 | stat=3.540, p=0.097 |

**Honest reading of this result, not an overstated one:** central tendency is statistically
indistinguishable between the two architectures (both paired tests, both metrics, p > 0.8)
— Dueling+PER does **not** demonstrate a mean-performance improvement at this training
budget (200 episodes/seed is short; both architectures are still climbing, per the reward
curves in `sumo/training_logs/multi_seed_variance_dueling_per_20260721_164704.json`). What
*is* visible is a large drop in cross-seed standard deviation — reward std more than halves
(120.13 → 44.51) and emergency-accuracy std drops roughly threefold (18.09 → 5.85) — which is
exactly the variance-reduction property the PER literature predicts (rare, high-TD-error
emergency transitions get replayed reliably instead of being sometimes-oversampled,
sometimes-starved under uniform sampling). But at n=5, Levene's test on both metrics lands at
p=0.10-0.40 — suggestive, directionally consistent with the mechanism, and **not** proof of a
real variance reduction at conventional significance thresholds. This project reports it as
what it is: a promising, mechanistically-explained signal that would need a larger seed count
(20-30, matching this project's own robustness-study convention elsewhere) to confirm, not a
validated claim of superiority. Both architectures remain in the codebase
(`dueling=False`/`prioritized_replay=False` stay the defaults for the deployed checkpoint,
since the baseline has strictly more training history and no measured downside) — this is an
available, documented option, not a silent replacement.

### 10.2 — Closing the Q-Learning SUMO domain gap

Part 5's "Future work" flagged a domain gap for **DQN**: trained purely in SUMO, evaluated
partly against the simplified non-SUMO `TrafficEnv`. The tabular Q-Learning agent had the
same latent gap in the opposite direction — `train_offline.py` trains and evaluates entirely
inside the simplified `TrafficEnv` (`rl/q_table_v2.pkl`, the deployed checkpoint, has never
seen a single SUMO-simulated vehicle).

**New implementation:** `sumo/sumo_env.py`'s `SumoTrafficEnv` gained a
`use_discrete_encoder=False` flag that swaps in the tabular agent's own `StateEncoder`
instead of the DQN feature extractor (the two produce differently-shaped state
representations — one discrete/tabular, one continuous — and were previously wired
one-directionally, DQN-only). `sumo/train_qlearning_sumo.py` (new) mirrors
`train_offline.py`'s training loop but drives it against real SUMO/TraCI microsimulation
instead of the simplified environment, saving to a separate `rl/q_table_sumo.pkl` (never
overwriting the deployed `rl/q_table_v2.pkl`).

**Verification (1000 episodes, seed=42, `medium` pattern,
`sumo/training_logs/qlearning_sumo_20260721_171622.json`):** emergency accuracy climbed from
an early-training ~38% to a final **53.6%**, average reward (last 200 episodes) **629.76**,
and — the one metric only a real microsimulator can produce, since the simplified
`TrafficEnv` has no vehicle-level physics — **real TraCI-measured average ambulance delay of
16.38s** over the final 50 episodes. This is a genuinely new capability, not a re-measurement
of an existing one: it's the first time this project has a tabular-agent result grounded in
actual vehicle dynamics (car-following, queuing, junction geometry) rather than the
simplified environment's aggregate queue-length abstraction.

**Why this doesn't fully close the loop, stated plainly:** this is one 1000-episode run at
one seed, not a variance-characterized result — unlike 8.11's Q-Learning and DQN multi-seed
studies, this run has not been repeated across seeds, so no confidence interval is reported
for it, and none should be inferred. It demonstrates the training pathway works end-to-end
(the encoder-swap, the TraCI wiring, convergence behavior) and gives one honest data point;
treating 53.6%/16.38s as a precise, generalizable estimate rather than "the RL pathway is
now verified to function correctly in SUMO, with plausible-looking results" would be
overclaiming. A multi-seed variance study for this configuration is the natural next step,
using the same 5-seed protocol as 8.11 and 10.1 above, and is listed as such rather than
silently assumed.

### 10.3 — DQN hyperparameter study (learning rate, hidden width, target-update frequency)

**Original implementation:** every DQN training run in this project used one fixed
hyperparameter triple (`lr=5e-4`, `hidden_dim=128`, `target_update_freq=200`), chosen once
early on and never revisited with evidence.

**Why insufficient:** an unexamined choice isn't necessarily a wrong one, but it isn't a
justified one either — a reviewer is right to ask "why these numbers" when no ablation exists.

**New implementation:** `sumo/hyperparameter_study.py` (new) runs a one-factor-at-a-time
sweep — 7 configurations (baseline plus &plusmn;1 alternative each for learning rate,
hidden-dim, and target-update frequency), 3 seeds per configuration, 100 SUMO episodes/seed,
`medium` pattern, each checkpoint evaluated on 30 clean seeds in the simplified simulator
(same protocol as `train_multi_seed.py`'s existing variance studies). Along the way, the same
class of Windows console-encoding crash from 10.1's methodology work
(`UnicodeEncodeError` on a non-cp1252 character in a `print()` separator, this time `ε` in
`train_multi_seed.py`'s per-episode progress line) was found and fixed — a real, reproducible
bug affecting any Windows run with redirected stdout, not cosmetic.

**Results (44 minutes total, 21 runs,
`sumo/training_logs/hyperparameter_study_20260721_182512.json`):**

| Config | lr | hidden_dim | target_update | Reward (mean of 3 seeds) | Emergency acc. |
|---|---|---|---|---|---|
| baseline | 5e-4 | 128 | 200 | &minus;39.5 | 59.0% |
| lr_low | 1e-4 | 128 | 200 | &minus;483.9 | 18.5% |
| lr_high | 2e-3 | 128 | 200 | +83.0 | 66.0% |
| hidden_small | 5e-4 | 64 | 200 | &minus;262.0 | 46.0% |
| hidden_large | 5e-4 | 256 | 200 | +47.5 | 70.2% |
| target_update_fast | 5e-4 | 128 | 50 | &minus;315.0 | 37.0% |
| **target_update_slow** | 5e-4 | 128 | **500** | **+116.9** | **75.9%** |

**Honest reading:** at this short training budget (100 episodes/seed, a fraction of the
750&ndash;1000-episode budgets used for the actually-deployed checkpoints), every alternative
configuration produced a clear, consistent direction relative to baseline across all 3
seeds — this is a much larger, more visually obvious effect than the Dueling+PER comparison's
subtle variance signal, not a borderline call. `target_update_slow` (500 steps between hard
target updates, vs. the deployed 200) was the strongest configuration by a wide margin on
both reward and accuracy; `lr_low` (1e-4) and `target_update_fast` (50) were both clearly
worse than baseline. The direction is mechanistically sensible: less frequent target updates
mean more stable (less moving) bootstrapping targets, which matters more at a short training
budget where the online network is still changing quickly; a too-low learning rate simply
hasn't moved far enough from initialization in 100 episodes. **What this is not:** a
statistically confirmed result at conventional thresholds (n=3 seeds/config yields 2 degrees
of freedom for a paired test — deliberately not computed, rather than reported as if it meant
more than it does, per `hyperparameter_study.py`'s own inline rationale) and not a direct
recommendation to retrain the deployed checkpoint at `target_update=500`, since all of this
project's actually-deployed checkpoints were trained at far larger budgets where the
short-budget instability this sweep is picking up on may matter less. It is a real, informative
signal pointing at target-update frequency as the highest-leverage of the three hyperparameters
swept, and a well-justified starting point for the next full-budget retrain — not a finished
substitute for one.
