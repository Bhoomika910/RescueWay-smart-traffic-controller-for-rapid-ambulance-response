# sumo/sumo_pas_study.py
"""
SUMO-native evaluation of the Perception-Aware Shield (PAS).

Extends Section V-C from the simplified environment to the microscopic
SUMO environment used for every headline result. Design decisions, each
anchored to the paper:

* Wrapped policy = Max-Pressure (independent implementation already in
  controllers/max_pressure.py). Table II established that the shield
  EQUALISES emergency outcomes across wrapped policies (override rate
  differs, emergency metrics do not), so the choice of wrapped policy
  cannot flatter PAS; Max-Pressure is the strongest external baseline
  and needs no torch checkpoint.
* Noise model = the repository's own GPSNoise.apply_zone (adjacency-
  weighted misclassification), applied to the zone the adapter DETECTS,
  producing the REPORTED zone stream both shields consume. Fixes arrive
  at 1 Hz (every simulation step); control decisions every DECISION s.
* Naive shield = rl.safety_shield.shield_emergency_action on the LATEST
  reported zone (the paper's V-C behaviour).
* PAS = rl.perception_aware_shield on the same report stream. Motion
  prior for SUMO: an ambulance's approach is CONSTANT for the whole
  emergency (it is driving down one road), so advance_rate=0.0 — the
  physically correct prior in this environment, and stated as such.
* Metrics = the paper's own: emergency accuracy scored against the TRUE
  approach, and TraCI-measured peak per-vehicle ambulance waiting time.
* Pairing = for each (seed, p), naive and PAS run on the IDENTICAL route
  file and identical noise realisation (same numpy seed), so differences
  are attributable to the shield alone. Wilcoxon signed-rank across
  seeds, matching the paper's protocol.

Run from backend/sumo:  python sumo_pas_study.py --seeds 30
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

import traci  # noqa: E402
from sumolib import checkBinary  # noqa: E402

from generate_routes import generate_routes  # noqa: E402
from sumo_adapter import SUMOAdapter, ZONE_TO_EDGE  # noqa: E402
from rl.safety_shield import shield_emergency_action  # noqa: E402
from rl.perception_aware_shield import PerceptionAwareShield  # noqa: E402
from noise.noise_models import GPSNoise, NoiseConfig  # noqa: E402

DURATION = 600
DECISION = 5          # seconds between control decisions
YELLOW = 3            # paper's mandatory transition
FIX_HZ = 1            # GPS fixes per simulation second


def max_pressure_zone(adapter):
    """Serve the approach with the largest halting queue (Varaiya-style
    single-intersection pressure with sink lanes uncongested)."""
    q = adapter.get_queue_lengths()
    return max(q, key=q.get)


def run_episode(seed, p_misclass, shield_kind, sumo_bin, route_file):
    rng = np.random.default_rng(seed)          # noise realisation
    np.random.seed(seed)                        # GPSNoise uses global np
    noise = GPSNoise(NoiseConfig(gps_misclass_prob=p_misclass))

    traci.start([sumo_bin, "-n", os.path.join(_HERE, "intersection.net.xml"),
                 "-r", route_file, "--no-warnings", "--no-step-log",
                 "--time-to-teleport", "300", "--seed", str(seed)])
    adapter = SUMOAdapter(verify_links=True)
    pas = PerceptionAwareShield(assumed_misclass_prob=max(p_misclass, 0.05),
                                advance_rate=0.0)

    current_zone = 1
    adapter.set_signal(current_zone)
    ticks_ok = ticks_total = 0
    peak_wait = 0.0
    emergency_was_active = False

    for t in range(DURATION):
        amb_present, true_zone = adapter.detect_ambulance()

        if amb_present:
            reported = noise.apply_zone(true_zone)
            if not emergency_was_active:
                pas.notify_emergency(True, first_report=reported)
                emergency_was_active = True
            else:
                # feed every 1 Hz fix into the filter regardless of
                # whether this second is a decision second
                pas.filter.update(reported)
            peak_wait = max(peak_wait, adapter.get_ambulance_waiting_time())
        else:
            if emergency_was_active:
                pas.notify_emergency(False)
                emergency_was_active = False

        if t % DECISION == 0:
            raw_action = max_pressure_zone(adapter) - 1  # wrapped policy
            if amb_present:
                if shield_kind == "naive":
                    enacted, _ = shield_emergency_action(
                        raw_action, reported, current_zone)
                else:  # pas — MAP zone from the full fix history
                    map_zone, _conf = pas.filter.map_zone()
                    enacted, _ = shield_emergency_action(
                        raw_action, map_zone, current_zone)
            else:
                enacted = raw_action

            target = enacted + 1 if enacted < 4 else current_zone
            if target != current_zone:
                adapter.set_yellow()
                for _ in range(YELLOW):
                    adapter.step()
                adapter.set_signal(target)
                current_zone = target

            if amb_present:
                ticks_total += 1
                ticks_ok += (current_zone == true_zone)

        adapter.step()

    traci.close()
    acc = 100.0 * ticks_ok / ticks_total if ticks_total else float("nan")
    return acc, peak_wait


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--probs", type=float, nargs="+",
                    default=[0.0, 0.317, 0.867])
    args = ap.parse_args()

    sumo_bin = checkBinary("sumo")
    rows = {}
    for p in args.probs:
        res = {"naive": {"acc": [], "delay": []},
               "pas": {"acc": [], "delay": []}}
        for s in range(args.seeds):
            rf = os.path.join(_HERE, f"_pas_routes_{s}.rou.xml")
            zone = (s % 4) + 1
            generate_routes(pattern="medium", duration=DURATION,
                            ambulance_zone=zone,
                            ambulance_time=120 + 10 * (s % 20),
                            seed=s, output_file=rf)
            for kind in ("naive", "pas"):
                acc, dly = run_episode(s, p, kind, sumo_bin, rf)
                res[kind]["acc"].append(acc)
                res[kind]["delay"].append(dly)
            os.remove(rf)
        rows[p] = res
        na, pa = np.array(res["naive"]["acc"]), np.array(res["pas"]["acc"])
        nd, pd_ = np.array(res["naive"]["delay"]), np.array(res["pas"]["delay"])
        print(f"p={p:.3f}  acc: naive {np.nanmean(na):5.1f}±{np.nanstd(na):4.1f}"
              f" | PAS {np.nanmean(pa):5.1f}±{np.nanstd(pa):4.1f}   "
              f"delay(s): naive {nd.mean():5.2f} | PAS {pd_.mean():5.2f}")

    try:
        from scipy import stats
        for p in args.probs:
            if p == 0:
                continue
            na = np.array(rows[p]["naive"]["acc"])
            pa = np.array(rows[p]["pas"]["acc"])
            w = stats.wilcoxon(pa, na)
            d = pa - na
            print(f"p={p}: PAS-naive = +{d.mean():.1f} pts "
                  f"(Wilcoxon p={w.pvalue:.2e}, d={d.mean()/max(d.std(),1e-9):.2f})")
            nd = np.array(rows[p]["naive"]["delay"])
            pd_ = np.array(rows[p]["pas"]["delay"])
            wd = stats.wilcoxon(nd, pd_) if np.any(nd != pd_) else None
            print(f"      ambulance delay: naive {nd.mean():.2f}s -> PAS "
                  f"{pd_.mean():.2f}s"
                  + (f" (p={wd.pvalue:.2e})" if wd else " (identical)"))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
