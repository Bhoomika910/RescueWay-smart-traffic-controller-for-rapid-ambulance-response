# NEW IN V4 — Perception-Aware Shield + Base-Paper Comparison

This version adds three modules and one study on top of the v3 codebase.
Nothing existing was modified; every published v3 result remains
reproducible unchanged.

## New files

1. `backend/rl/perception_aware_shield.py`
   The Perception-Aware Shield (PAS): a discrete Bayes filter over the
   ambulance's approach (sensor model mirrors `noise/noise_models.py`'s
   GPSNoise exactly; inbound motion prior; MAP zone + posterior
   confidence + optional latency-ahead prediction), composed with the
   existing `rl/safety_shield.py` rather than replacing it.
   Solves the Section V-C finding: the naive shield's guarantee
   dissolves under GNSS zone misclassification.

2. `backend/sumo/sumo_pas_study.py`
   SUMO-native replication of the Section V-C sweep with PAS.
   30 seeds, paired episodes (identical routes + noise per seed),
   Wilcoxon tests. Measured here (medium demand, Max-Pressure wrapped):

     p=0.000  acc: naive 100.0 | PAS 100.0   delay: 0.00 | 0.00
     p=0.317  acc: naive  72.4 | PAS  94.4   delay: 6.97 | 0.00  (p=1.7e-5)
     p=0.867  acc: naive  15.6 | PAS  87.8   delay: 116.9 | 1.60 (p=1.7e-6)

3. `backend/controllers/reactive_priority.py`
   Independent implementation of the BASE PAPER's emergency mechanism
   (Veeraiah et al., ICIRCA 2025, Sec. III-E): the ambulance's lane is
   granted the NEXT phase after the current phase completes its cycle —
   reactive by construction.

4. `backend/sumo/sumo_basepaper_comparison.py`
   Head-to-head: base-paper reactive vs RescueWay proactive shield vs
   PAS. All three share the SAME 25 s round-robin base cycle, so
   differences reflect only the emergency mechanism. Same routes, same
   seeds, same noise per seed. Measured (30 seeds):

     p = 0.0 (perfect detection, base paper's own assumption)
       controller   acc %        delay s   stop %   queue
       reactive     80.6±19.5    0.77      10.0     23.8
       shield      100.0±0.0     0.00       0.0     25.3
       pas         100.0±0.0     0.00       0.0     25.3

     p = 0.317 (dense-urban GNSS error)
       reactive     60.8±23.8   14.23      53.3     23.9
       shield       71.2±13.6    7.07      73.3     25.5
       pas          95.4±8.5     0.10       6.7     25.3

## How to run

    pip install eclipse-sumo traci sumolib libsumo scipy numpy
    export SUMO_HOME=$(python -c "import sumo,os;print(os.path.dirname(sumo.__file__))")
    cd backend/sumo
    python sumo_pas_study.py --seeds 30
    python sumo_basepaper_comparison.py --seeds 30 --probs 0.0 0.317

Note: `sumo_basepaper_comparison.py` loads reactive_priority.py directly
by path to avoid `controllers/__init__.py`'s torch import; no torch is
required for either study.

## Paper mapping

* New subsection V-C2 "Restoring the guarantee under perception error"
  <- sumo_pas_study.py results.
* New subsection (or expanded V-A) "Comparison with reactive next-phase
  preemption" <- sumo_basepaper_comparison.py results; cite Veeraiah et
  al. (ICIRCA 2025) as the base paper whose reactive-priority limitation
  this work solves, DNLight (IEEE IoT-J 2025) and CCPVLight (MT-ITS
  2025) as the closest reward-weighted alternatives.

5. `backend/sumo/generate_routes_calibrated.py`
   Poisson-arrival route generation (exponential inter-arrival gaps)
   replacing the per-second Bernoulli draw, plus a 'hangzhou' regime
   calibrated to the real-intersection counts published in Xu et al.
   (MT-ITS 2025), Table II: 240/275/267/180 veh/h with measured
   left-turn rates 0.16/0.14/0.11/0.20. Closes the "artificial traffic
   demand" reviewer issue; turning-split assumption documented in the
   module docstring. Existing synthetic regimes kept at identical mean
   rates for comparability.
