# sumo/sumo_basepaper_comparison.py
"""
Base-paper limitation study: REACTIVE next-phase priority (Veeraiah et
al., ICIRCA 2025, Sec. III-E) vs RescueWay's PROACTIVE shielded control,
with and without perception error — all in the same SUMO harness, same
routes, same seeds, same metric definitions.

Controllers compared (identical route file + noise realisation per seed):
  reactive : base paper's mechanism (controllers/reactive_priority.py) —
             ambulance's lane becomes the NEXT phase after the current
             25 s phase completes.
  shield   : RescueWay naive shield over Max-Pressure — serves the
             REPORTED approach immediately (proactive preemption).
  pas      : RescueWay Perception-Aware Shield — serves the Bayes-MAP
             approach immediately.

Metrics (paper definitions):
  amb_delay : peak TraCI per-vehicle waiting time of the ambulance (s)
  stop      : did the ambulance stop at all (waiting time > 0)
  acc       : % of emergency decision ticks whose green approach == TRUE
              approach
  queue     : mean total halting vehicles per decision tick (congestion
              cost of each priority mechanism)

Run from backend/sumo:
  python sumo_basepaper_comparison.py --seeds 30 --probs 0.0 0.317
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
for p in (_BACKEND, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import libsumo
    sys.modules["traci"] = libsumo
except ImportError:
    pass
import traci  # noqa: E402
from sumolib import checkBinary  # noqa: E402

from generate_routes import generate_routes  # noqa: E402
from sumo_adapter import SUMOAdapter  # noqa: E402
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("reactive_priority", os.path.join(_BACKEND, "controllers", "reactive_priority.py"))
_rp = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_rp)
ReactivePriorityController = _rp.ReactivePriorityController  # noqa: E402
from rl.safety_shield import shield_emergency_action  # noqa: E402
from rl.perception_aware_shield import PerceptionAwareShield  # noqa: E402
from noise.noise_models import GPSNoise, NoiseConfig  # noqa: E402

DURATION = 600
DECISION = 5
YELLOW = 3


def max_pressure_zone(adapter):
    q = adapter.get_queue_lengths()
    return max(q, key=q.get)


# All three controllers share the SAME 25 s round-robin base cycle so
# that queue/congestion differences reflect ONLY the emergency mechanism
# (reactive next-phase vs proactive immediate), not the normal-traffic
# policy. shield/pas wrap this cycle instead of Max-Pressure here.


def switch_with_yellow(adapter, current_zone, target):
    if target != current_zone:
        adapter.set_yellow()
        for _ in range(YELLOW):
            adapter.step()
        adapter.set_signal(target)
    return target


def run_episode(kind, seed, p_misclass, sumo_bin, route_file):
    np.random.seed(seed)
    noise = GPSNoise(NoiseConfig(gps_misclass_prob=p_misclass))
    traci.start([sumo_bin, "-n", os.path.join(_HERE, "intersection.net.xml"),
                 "-r", route_file, "--no-warnings", "--no-step-log",
                 "--time-to-teleport", "300", "--seed", str(seed)])
    adapter = SUMOAdapter(verify_links=True)

    reactive = ReactivePriorityController(phase_len=25)
    base_cycle = ReactivePriorityController(phase_len=25)  # no emergencies fed
    pas = PerceptionAwareShield(assumed_misclass_prob=max(p_misclass, 0.05),
                                advance_rate=0.0)
    current_zone = 1
    adapter.set_signal(current_zone)

    ticks_ok = ticks_total = 0
    peak_wait = 0.0
    queue_sum = queue_n = 0
    emergency_was_active = False

    for t in range(DURATION):
        amb_present, true_zone = adapter.detect_ambulance()
        if amb_present:
            reported = noise.apply_zone(true_zone)
            if not emergency_was_active:
                pas.notify_emergency(True, first_report=reported)
                emergency_was_active = True
            else:
                pas.filter.update(reported)
            reactive.notify_emergency(reported)
            peak_wait = max(peak_wait, adapter.get_ambulance_waiting_time())
        else:
            if emergency_was_active:
                pas.notify_emergency(False)
                emergency_was_active = False
            reactive.clear_emergency()

        if kind == "reactive":
            target = reactive.decide(dt=1)
            if target != current_zone:
                current_zone = switch_with_yellow(adapter, current_zone, target)
            if amb_present and t % DECISION == 0:
                ticks_total += 1
                ticks_ok += (current_zone == true_zone)
        else:
            base_zone = base_cycle.decide(dt=1)
            if t % DECISION == 0:
                raw = base_zone - 1
                if amb_present:
                    zone_for_shield = (reported if kind == "shield"
                                       else pas.filter.map_zone()[0])
                    enacted, _ = shield_emergency_action(
                        raw, zone_for_shield, current_zone)
                else:
                    enacted = raw
                target = enacted + 1 if enacted < 4 else current_zone
                current_zone = switch_with_yellow(adapter, current_zone, target)
                if amb_present:
                    ticks_total += 1
                    ticks_ok += (current_zone == true_zone)

        if t % DECISION == 0:
            queue_sum += sum(adapter.get_queue_lengths().values())
            queue_n += 1
        adapter.step()

    traci.close()
    acc = 100.0 * ticks_ok / ticks_total if ticks_total else float("nan")
    return {"acc": acc, "amb_delay": peak_wait,
            "stop": 1.0 if peak_wait > 0 else 0.0,
            "queue": queue_sum / max(queue_n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--probs", type=float, nargs="+", default=[0.0, 0.317])
    ap.add_argument("--kinds", nargs="+",
                    default=["reactive", "shield", "pas"])
    args = ap.parse_args()
    sumo_bin = checkBinary("sumo")

    all_res = {}
    for p in args.probs:
        all_res[p] = {k: {m: [] for m in ("acc", "amb_delay", "stop", "queue")}
                      for k in args.kinds}
        for s in range(args.seeds):
            rf = os.path.join(_HERE, f"_bp_routes_{s}.rou.xml")
            generate_routes(pattern="medium", duration=DURATION,
                            ambulance_zone=(s % 4) + 1,
                            ambulance_time=120 + 10 * (s % 20),
                            seed=s, output_file=rf)
            for kind in args.kinds:
                r = run_episode(kind, s, p, sumo_bin, rf)
                for m, v in r.items():
                    all_res[p][kind][m].append(v)
            os.remove(rf)

        print(f"\n=== p = {p} ===")
        print(f"{'controller':<10} {'acc %':>10} {'amb delay s':>12} "
              f"{'stop %':>8} {'mean queue':>11}")
        for kind in args.kinds:
            r = all_res[p][kind]
            print(f"{kind:<10} "
                  f"{np.nanmean(r['acc']):>6.1f}±{np.nanstd(r['acc']):<4.1f}"
                  f"{np.mean(r['amb_delay']):>9.2f} "
                  f"{100*np.mean(r['stop']):>8.1f} "
                  f"{np.mean(r['queue']):>11.1f}")

    try:
        from scipy import stats
        pairs = [("reactive", "shield"), ("reactive", "pas"),
                 ("shield", "pas")]
        for p in args.probs:
            for a, b in pairs:
                if a not in args.kinds or b not in args.kinds:
                    continue
                for metric in ("amb_delay", "acc"):
                    x = np.array(all_res[p][a][metric], dtype=float)
                    y = np.array(all_res[p][b][metric], dtype=float)
                    m = ~(np.isnan(x) | np.isnan(y))
                    x, y = x[m], y[m]
                    if len(x) < 5 or not np.any(x != y):
                        continue
                    w = stats.wilcoxon(x, y)
                    print(f"p={p} {metric}: {a} {x.mean():.2f} vs "
                          f"{b} {y.mean():.2f} (Wilcoxon p={w.pvalue:.2e})")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
