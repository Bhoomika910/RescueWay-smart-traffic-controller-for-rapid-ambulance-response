# services/traffic_controller.py
"""
RescueWay Traffic Controller — Single Source of Truth Architecture

ALL signal changes go through _apply_signal_state().
This guarantees:
  1. Arduino command is sent and confirmed
  2. Internal state is updated ONLY after confirmation
  3. Frontend broadcast happens ONLY after state is confirmed
  4. No component ever sees a state that hasn't been confirmed

Signal Change Flow:
  Controller Decision
       ↓
  _apply_signal_state(zone, color)
       ↓
  Arduino Serial Command (non-blocking via thread pool)
       ↓
  Arduino ACK Received
       ↓
  Internal signal_states Updated
       ↓
  WebSocket Broadcast to Frontend
       ↓
  Frontend Animation Changes

Both frontend and Arduino always show the SAME signal.
"""
import asyncio
import random
import time

from config import (
    DEFAULT_ALL_RED_DURATION,
    DEFAULT_EMERGENCY_DURATION,
    DEFAULT_GREEN_DURATION,
    DEFAULT_YELLOW_DURATION,
    DEMO_MODE,
    DEMO_INTERVAL_MIN,
    DEMO_INTERVAL_MAX,
)
from data import db
from logging_system import app_logger
from rl.ambulance_delay import estimate_clearance
from rl.reward import compute_reward
from rl.safety_shield import shield_emergency_action
from services.emergency_manager import EmergencyManager
from traffic.simulator import TrafficSimulator


class TrafficController:
    def __init__(self):
        self.mode = "normal"
        self.current_zone = 1
        self.green_duration = DEFAULT_GREEN_DURATION
        self.yellow_duration = DEFAULT_YELLOW_DURATION
        self.all_red_duration = DEFAULT_ALL_RED_DURATION
        self.emergency_zone = None
        self.emergency_time = 0
        self.step_count = 0
        self.running = False
        self.paused = False
        # Public demo emergency scheduler
        self._next_demo_emergency_at = None

        # Traffic simulator
        self.simulator = TrafficSimulator()
        self.simulator.set_traffic_pattern("medium")

        # Metrics collector
        from metrics.collector import MetricsCollector

        self.metrics_collector = MetricsCollector()

        # Emergency Manager (handles multiple ambulances)
        self.emergency_manager = EmergencyManager()
        self.gps_model = self.emergency_manager.gps_model
        # Must be set after self.emergency_manager exists — the setter
        # below propagates into it. Real, active emergencies always use
        # emergency_green_adaptive (GPS/ZONE_CONFIG-driven, unaffected by
        # this), so this only reaches the manager's *fallback* default.
        self.emergency_duration = DEFAULT_EMERGENCY_DURATION
        self.emergency_eta = 0
        self.emergency_distance = 0
        self.emergency_green_adaptive = DEFAULT_EMERGENCY_DURATION
        self.emergency_preemption_active = False
        # Proactive clearance prediction — recomputed every time an
        # emergency decision is made (trigger + each _run_emergency hold
        # tick), from rl/ambulance_delay.estimate_clearance(). Lets the
        # dashboard/XAI/reward answer "will this ambulance actually stop"
        # instead of only "is the right zone green right now."
        self.emergency_clearance_deficit = 0.0
        self.emergency_predicted_stop = False

        # Dependencies (set later)
        self._serial_bridge = None
        self._broadcast = None
        self._rl_agent = None

        # Tracking
        self.last_action = -1
        self.last_reward = 0
        self.total_reward = 0
        self.decision_count = 0

        # Current signal states for all zones — SINGLE SOURCE OF TRUTH
        # Only _apply_signal_state() may modify this dict
        self.signal_states = {1: "red", 2: "red", 3: "red", 4: "red"}

        # Starvation protection: track how many cycles since each zone was served
        # Prevents DQN from ping-ponging between 2 zones under asymmetric traffic
        self._zone_last_served = {1: 0, 2: 0, 3: 0, 4: 0}
        self._starvation_limit = 4  # Force-serve a zone if not served for N cycles

        # ─── Decision-transparency state (field/dict names kept as
        # "xai_info"/"xai" for backward compatibility with the frontend
        # and any existing consumers of the WebSocket payload — see the
        # naming note on _build_xai_explanation below for why the content
        # is described as "decision transparency," not "Explainable AI"
        # in the rigorous technical sense) ────────────────────
        # Populated by _rl_choose_next_zone / _rl_choose_emergency_zone.
        # Every field here is derived from real computed values —
        # nothing here is fabricated or hardcoded for display purposes.
        self.previous_zone = 1
        self.xai_info = {
            "reason": "System initializing",
            "reward_breakdown": {},
            "q_values": {},
            "confidence": None,
            "feature_importance": {},
            "inference_time_ms": None,
            "previous_zone": 1,
            "selected_zone": 1,
            "overridden": False,
            "dqn_original_zone": None,
            "mode": "normal",
            "clearance_deficit": 0.0,
            "predicted_stop": False,
        }

    @property
    def emergency_duration(self):
        return self._emergency_duration

    @emergency_duration.setter
    def emergency_duration(self, value):
        # Propagate to EmergencyManager's fallback default (see its
        # default_green_duration docstring) so this API-settable field
        # (POST /simulation/timings) actually has an observable effect —
        # previously it was stored and echoed back by the API but never
        # read by _run_emergency(), which always uses
        # emergency_green_adaptive (GPS/ZONE_CONFIG-driven) instead.
        self._emergency_duration = value
        self.emergency_manager.default_green_duration = value

    def set_dependencies(self, serial_bridge, broadcast_fn, rl_agent):
        self._serial_bridge = serial_bridge
        self._broadcast = broadcast_fn
        self._rl_agent = rl_agent

    def set_traffic_pattern(self, pattern):
        self.simulator.set_traffic_pattern(pattern)
        app_logger.log_traffic_pattern_change(pattern)

    def _get_queue_lengths(self):
        return {z: self.simulator.zones[z].get_length() for z in [1, 2, 3, 4]}

    # ══════════════════════════════════════════════════════
    # DECISION TRANSPARENCY — built from real decision data only.
    #
    # Naming note (design-review finding, not a functional change): the
    # variable/field names below (xai_info, "xai", _build_xai_explanation)
    # are kept as-is for backward compatibility with the frontend and any
    # existing WebSocket consumers, but the *content* here is intentionally
    # a shallow, fast form of transparency appropriate for a real-time
    # embedded controller — raw Q-values, a softmax-derived confidence
    # score, and a templated reason string — not "Explainable AI" in the
    # rigorous technical sense (attribution methods like SHAP/saliency
    # maps, or counterfactual explanations). In particular, the `reason`
    # text below is built by inspecting the *reward function's* known
    # structure (which component of get_reward_breakdown dominated), not
    # by attributing the *trained network's* actual forward pass — those
    # two can diverge whenever the policy is imperfectly trained (a
    # network that has collapsed toward one dominant action, for example,
    # would still produce a plausible-sounding reason string here, because
    # the text describes what the reward function rewards, not what the
    # network actually weighted to reach its output). Treat this as
    # operator-facing decision transparency, not a verified attribution of
    # the model's internal decision process.
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _softmax_confidence(q_values: dict, action: int):
        """
        Derive a confidence percentage from Q-values via softmax.
        Standard technique for converting value estimates into a
        probability-like confidence measure — computed from the DQN's
        actual output, not fabricated. Returns None if unavailable.
        """
        if not q_values or action is None or action < 0 or action >= len(q_values):
            return None
        import numpy as np

        values = np.array(list(q_values.values()), dtype=np.float64)
        values = values - np.max(values)
        exp_values = np.exp(values)
        probs = exp_values / np.sum(exp_values)
        return round(float(probs[action]) * 100, 1)

    def _compute_feature_importance(self, state):
        """
        Real gradient-based feature attribution (rl/feature_attribution.
        gradient_saliency) for this decision — which input features the
        network's chosen-action Q-value is actually most sensitive to,
        not a self-reported description. Design-review finding (category
        14 "Explainability"): the rest of xai_info is decision
        TRANSPARENCY (real Q-values, a real reward breakdown) but was not
        attribution in the ML-XAI sense until this method existed — see
        rl/feature_attribution.py's module docstring for the two methods
        implemented and why this cheap one (a single backward pass, ~1ms)
        is the one wired into the live per-decision path, versus the
        more expensive permutation_importance() used for offline analysis.

        Returns {} (not an exception) whenever unavailable — a tabular
        QAgent has no gradients to take, and any unexpected failure here
        must never break the live control loop over an instrumentation
        feature. Callers should treat an empty dict as "not available for
        this agent," not as an error.
        """
        if not hasattr(self._rl_agent, "online_net"):
            return {}  # tabular QAgent — no gradients; see module docstring
        try:
            from rl.feature_attribution import gradient_saliency

            return gradient_saliency(self._rl_agent, state)
        except Exception:
            return {}

    def _build_xai_explanation(
        self, action, reward, queue_lengths, emergency_active, emergency_zone, ambulance_delay=None
    ):
        """
        Build a human-readable explanation from real values already
        computed for this decision — no invented text.
        """
        from rl.reward import get_reward_breakdown

        breakdown = get_reward_breakdown(
            action=action,
            amb_zone=emergency_zone if emergency_active else 0,
            prev_signal=self.current_zone,
            traffic=self.simulator.traffic_pattern,
            queue_lengths=queue_lengths,
            ambulance_delay=ambulance_delay,
        )

        chosen_zone = action + 1 if action < 4 else None

        if emergency_active:
            stop_note = ""
            if ambulance_delay is not None:
                stop_note = (
                    " — predicted to pass through without stopping"
                    if ambulance_delay <= 0.5
                    else f" — predicted {ambulance_delay:.1f}s delay at current queue/ETA"
                )
            if chosen_zone == emergency_zone:
                reason = (
                    f"Emergency priority: ambulance detected in Zone "
                    f"{emergency_zone}, signal assigned to clear its path{stop_note}"
                )
            elif action == 4:
                reason = (
                    f"Emergency priority: extending current green while "
                    f"ambulance approaches Zone {emergency_zone}{stop_note}"
                )
            else:
                reason = (
                    f"Emergency active in Zone {emergency_zone}, but "
                    f"selected action does not directly serve it{stop_note}"
                )
        else:
            components = {k: v for k, v in breakdown.items() if k != "total"}
            dominant = max(components, key=lambda k: components[k]) if components else None
            max_zone = max(queue_lengths, key=queue_lengths.get) if queue_lengths else None
            max_queue = queue_lengths.get(max_zone, 0) if max_zone else 0

            if action == 4:
                reason = "Extending current green — no zone change needed"
            elif action == 5:
                reason = "Maintaining normal cycle"
            elif dominant == "congestion" and chosen_zone == max_zone:
                reason = (
                    f"Zone {chosen_zone} selected: highest queue length "
                    f"({max_queue} vehicles) among all zones"
                )
            elif dominant == "fairness":
                reason = f"Zone {chosen_zone} selected to correct queue " f"imbalance across zones"
            elif dominant == "throughput":
                reason = (
                    f"Zone {chosen_zone} selected for vehicle clearance "
                    f"({queue_lengths.get(chosen_zone, 0)} vehicles waiting)"
                )
            else:
                reason = (
                    f"Zone {chosen_zone} selected by DQN "
                    f"(queue={queue_lengths.get(chosen_zone, 0)} vehicles)"
                )

        return reason, breakdown

    # ══════════════════════════════════════════════════════
    # SINGLE SOURCE OF TRUTH — ALL SIGNAL CHANGES GO HERE
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _ack_confirms(ack, zone, color):
        """
        True only if the Arduino's ACK actually confirms the requested
        zone/color change took effect — either a real "OK:zone=N,signal=X"
        reply from the firmware, or the bridge's documented simulation-mode
        echo ("SIM:zone=N,signal=X"), with N/X cross-checked against what
        was actually requested. An empty ack, an "ERR:..." ack, or a
        malformed/mismatched ack all return False.

        Previously the ACK was captured but never inspected here — any of
        those failure cases still caused signal_states to be updated and
        broadcast as if the physical light had changed, silently breaking
        this module's own documented guarantee ("Internal state is updated
        ONLY after Arduino ACK Received").
        """
        if not ack or not ack.startswith(("OK:", "SIM:")):
            return False
        try:
            payload = ack.split(":", 1)[1]
            fields = dict(part.split("=", 1) for part in payload.split(","))
            return int(fields.get("zone", -1)) == zone and fields.get("signal") == color
        except (ValueError, IndexError):
            return False

    async def _apply_signal_state(
        self, zone, color, broadcast_mode="normal", phase="green", phase_description="", extra=None
    ):
        """
        THE ONLY WAY to change a traffic signal.

        This function guarantees synchronization:
        1. Send command to Arduino (non-blocking)
        2. Wait for Arduino ACK
        3. Update internal signal_states ONLY if the ACK actually confirms
           the requested zone/color (see _ack_confirms) — a failed/garbled
           command leaves signal_states at its last confirmed value instead
           of being silently advanced.
        4. Broadcast to frontend ONLY after state is confirmed, and only
           ever describing the zone's true (confirmed) color — never the
           attempted-but-unconfirmed one.

        Both Arduino and frontend always show the SAME signal — enforced
        by this check, not merely assumed from the hardware call succeeding.
        """
        # Step 1: Send to Arduino and WAIT for confirmation
        ack = await self._send_signal_to_hardware(zone, color)
        confirmed = self._ack_confirms(ack, zone, color)

        # Step 2: Update internal state ONLY after hardware actually confirmed.
        # Arduino firmware calls allRed() on every command, so when we set a
        # zone to green/yellow, all others become red — but only once we know
        # that command really landed.
        if confirmed:
            if color in ("green", "yellow"):
                for z in [1, 2, 3, 4]:
                    self.signal_states[z] = "red"
                self.signal_states[zone] = color
            else:
                self.signal_states[zone] = "red"
            broadcast_zone, broadcast_color = zone, color
        else:
            app_logger.log_error(
                f"Arduino did not confirm zone={zone} color={color} "
                f"(ack={ack!r}) — signal_states left at last confirmed value"
            )
            broadcast_zone = zone
            broadcast_color = self.signal_states.get(zone, "red")

        # Step 3: Broadcast to frontend, always reflecting the true
        # (confirmed) state, with an explicit hardware_confirmed flag so a
        # dashboard/RL consumer can tell a real change from a failed one.
        broadcast_extra = dict(extra) if extra else {}
        broadcast_extra["hardware_confirmed"] = confirmed
        await self._broadcast_full(
            mode=broadcast_mode,
            zone=broadcast_zone,
            color=broadcast_color,
            reward=self.last_reward,
            phase=phase,
            phase_description=phase_description,
            extra=broadcast_extra,
        )

        return ack

    async def _apply_all_red(
        self, broadcast_mode="normal", phase="all_red", phase_description="Intersection clearing"
    ):
        """
        Set ALL zones to red. Only sends ONE command to Arduino
        since Arduino firmware calls allRed() on every command anyway.

        Same ACK-validation discipline as _apply_signal_state: internal
        state only advances to "all red" once the hardware actually
        confirms it, not merely because the send call returned.
        """
        # Send ONE red command — Arduino's allRed() handles all zones
        ack = await self._send_signal_to_hardware(1, "red")
        confirmed = self._ack_confirms(ack, 1, "red")

        if confirmed:
            for z in [1, 2, 3, 4]:
                self.signal_states[z] = "red"
        else:
            app_logger.log_error(
                f"Arduino did not confirm all-red (ack={ack!r}) — "
                f"signal_states left at last confirmed value"
            )

        # Broadcast
        await self._broadcast_full(
            mode=broadcast_mode,
            zone=self.current_zone,
            color="all_red",
            reward=self.last_reward,
            phase=phase,
            phase_description=phase_description,
            extra={"hardware_confirmed": confirmed},
        )

        return ack

    async def _send_signal_to_hardware(self, zone, color):
        """
        Send signal command to Arduino via thread pool (non-blocking).
        Returns the ACK string.
        """
        try:
            ack = await asyncio.to_thread(self._serial_bridge.send_command, zone, color)
            app_logger.log_arduino_command(zone, color, ack)
            return ack
        except Exception as e:
            app_logger.log_arduino_error(str(e))
            return f"ERR:{e}"

    # ══════════════════════════════════════════════════════
    # SAFE TRANSITION — Uses _apply_signal_state exclusively
    # ══════════════════════════════════════════════════════

    async def _safe_transition(self, from_zone, to_zone, mode="normal"):
        """
        Execute a safe traffic signal transition.

        Sequence:
        1. YELLOW phase — current green zone turns yellow
        2. ALL RED phase — all zones red (clearance)
        3. GREEN phase — new zone turns green

        Every phase goes through _apply_signal_state,
        so Arduino, backend, and frontend are always in sync.
        """
        if from_zone == to_zone:
            return

        # ─── PHASE 1: YELLOW ──────────────────────────────
        await self._apply_signal_state(
            zone=from_zone,
            color="yellow",
            broadcast_mode=mode,
            phase="yellow",
            phase_description=f"Zone {from_zone} clearing",
        )

        self.simulator.step(active_signal=from_zone, is_green=False)
        app_logger.log_signal_change(from_zone, "yellow", from_zone)

        # Hold yellow
        for _ in range(self.yellow_duration * 2):
            if self.mode != mode and mode == "normal":
                return  # emergency interrupted
            await asyncio.sleep(0.5)

        # ─── PHASE 2: ALL RED ─────────────────────────────
        await self._apply_all_red(
            broadcast_mode=mode,
            phase="all_red",
            phase_description="Intersection clearing",
        )

        self.simulator.step(active_signal=to_zone, is_green=False)

        # Hold all-red
        for _ in range(self.all_red_duration * 2):
            if self.mode != mode and mode == "normal":
                return
            await asyncio.sleep(0.5)

        # ─── PHASE 3: GREEN ──────────────────────────────
        await self._apply_signal_state(
            zone=to_zone,
            color="green",
            broadcast_mode=mode,
            phase="green",
            phase_description=f"Zone {to_zone} active",
        )

        self.current_zone = to_zone
        app_logger.log_signal_change(to_zone, "green", from_zone)

    # ══════════════════════════════════════════════════════
    # RL DECISION
    # ══════════════════════════════════════════════════════

    def _rl_choose_next_zone(self):
        queue_lengths = self._get_queue_lengths()

        state = self._rl_agent.encode_rich_state(
            emergency_active=False,
            ambulance_zone=0,
            active_signal=self.current_zone,
            queue_lengths=queue_lengths,
        )

        inference_start = time.perf_counter()
        action = self._rl_agent.choose(state)
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        q_values = {}
        confidence = None
        if hasattr(self._rl_agent, "get_q_values"):
            try:
                q_values = self._rl_agent.get_q_values(state)
                confidence = self._softmax_confidence(q_values, action)
            except Exception:
                q_values = {}
                confidence = None

        reward = compute_reward(
            action=action,
            amb_zone=0,
            prev_signal=self.current_zone,
            traffic=self.simulator.traffic_pattern,
            queue_lengths=queue_lengths,
        )

        new_signal = action + 1 if action < 4 else self.current_zone

        reason, breakdown = self._build_xai_explanation(
            action=action,
            reward=reward,
            queue_lengths=queue_lengths,
            emergency_active=False,
            emergency_zone=0,
        )

        self.previous_zone = self.current_zone
        self.xai_info = {
            "reason": reason,
            "reward_breakdown": breakdown,
            "q_values": q_values,
            "confidence": confidence,
            "feature_importance": self._compute_feature_importance(state),
            "inference_time_ms": round(inference_ms, 3),
            "previous_zone": self.current_zone,
            "selected_zone": new_signal,
            "overridden": False,
            "dqn_original_zone": None,
            "mode": "normal",
            "clearance_deficit": 0.0,
            "predicted_stop": False,
        }

        next_state = self._rl_agent.encode_rich_state(
            emergency_active=False,
            ambulance_zone=0,
            active_signal=new_signal,
            queue_lengths=queue_lengths,
        )
        self._rl_agent.update(state, action, reward, next_state)

        self.last_action = action
        self.last_reward = reward
        self.total_reward += reward
        self.decision_count += 1

        return action, reward, new_signal

    def _rl_choose_emergency_zone(self, emergency_zone):
        """Make RL decision specifically for an emergency zone."""
        queue_lengths = self._get_queue_lengths()

        # Predict whether this ambulance will actually clear the
        # intersection before it arrives — self.emergency_eta/_distance are
        # already set by trigger_emergency() before this is called. This is
        # what makes the decision proactive rather than purely reactive:
        # the agent's state AND reward both reflect whether the current
        # queue can be cleared in the time actually available, not just
        # whether the flag is on.
        clearance = estimate_clearance(
            remaining_eta=self.emergency_eta,
            queue_length=queue_lengths.get(emergency_zone, 0),
            zone_is_served=(self.current_zone == emergency_zone),
        )
        self.emergency_clearance_deficit = clearance["deficit"]
        self.emergency_predicted_stop = clearance["will_stop"]

        state = self._rl_agent.encode_rich_state(
            emergency_active=True,
            ambulance_zone=emergency_zone,
            active_signal=self.current_zone,
            queue_lengths=queue_lengths,
            emergency_eta=self.emergency_eta,
            distance=self.emergency_distance,
        )

        inference_start = time.perf_counter()
        action = self._rl_agent.choose(state)
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        q_values = {}
        confidence = None
        if hasattr(self._rl_agent, "get_q_values"):
            try:
                q_values = self._rl_agent.get_q_values(state)
                confidence = self._softmax_confidence(q_values, action)
            except Exception:
                q_values = {}
                confidence = None

        reward = compute_reward(
            action=action,
            amb_zone=emergency_zone,
            prev_signal=self.current_zone,
            traffic=self.simulator.traffic_pattern,
            queue_lengths=queue_lengths,
            ambulance_delay=clearance["deficit"],
        )

        new_signal = action + 1 if action < 4 else self.current_zone

        reason, breakdown = self._build_xai_explanation(
            action=action,
            reward=reward,
            queue_lengths=queue_lengths,
            emergency_active=True,
            emergency_zone=emergency_zone,
            ambulance_delay=clearance["deficit"],
        )

        self.previous_zone = self.current_zone
        self.xai_info = {
            "reason": reason,
            "reward_breakdown": breakdown,
            "q_values": q_values,
            "confidence": confidence,
            "feature_importance": self._compute_feature_importance(state),
            "inference_time_ms": round(inference_ms, 3),
            "previous_zone": self.current_zone,
            "selected_zone": new_signal,
            "overridden": False,
            "dqn_original_zone": None,
            "mode": "emergency",
            "clearance_deficit": clearance["deficit"],
            "predicted_stop": clearance["will_stop"],
        }

        # ─── EMERGENCY SAFETY OVERRIDE ────────────────────
        # Unlike normal-mode cycling (which already has a starvation
        # override — see _run_rl_cycle), the emergency path used to trust
        # the DQN's raw action unconditionally for the FINAL signal
        # assignment. Reproduced case: ambulance_zone=1 with
        # active_signal=1 gets action=1 ("Green Zone 2") instead of the
        # ambulance's own zone. During an active emergency the ambulance's
        # zone must win regardless of whether the network has perfectly
        # generalized to this exact state — this is the safety-critical
        # path. The reward/training update below still uses the DQN's
        # ORIGINAL action (Q-learning must associate the reward with the
        # action actually evaluated, not a substituted one); only the
        # ENACTED zone (self.last_action, which _run_emergency reads to
        # drive the real signal) is overridden.
        #
        # shield_emergency_action (rl/safety_shield.py) is the single
        # shared implementation of this override — previously this logic
        # lived only here, with no equivalent in the benchmark controllers
        # (controllers/dqn_controller.py), so the benchmark's ~85%
        # DQN emergency-accuracy numbers could be misread as "what
        # production does," when production has actually guaranteed 100%
        # via this exact override the whole time. See
        # controllers/dqn_controller.py's DQNShieldedController for the
        # benchmark-side equivalent, now sharing this one function.
        enacted_action, overridden = shield_emergency_action(
            action, emergency_zone, self.current_zone
        )
        if overridden:
            override_msg = (
                f"Emergency safety override: DQN selected "
                f"'{self._rl_agent.get_best_action_name(action)}' for ambulance in "
                f"Zone {emergency_zone} — forcing Zone {emergency_zone} to guarantee "
                f"the correct signal during an active emergency"
            )
            app_logger.log_system_event(override_msg)
            self.xai_info["reason"] = override_msg
            self.xai_info["overridden"] = True
            self.xai_info["dqn_original_zone"] = new_signal
            self.xai_info["selected_zone"] = emergency_zone
            new_signal = emergency_zone

        next_state = self._rl_agent.encode_rich_state(
            emergency_active=True,
            ambulance_zone=emergency_zone,
            active_signal=new_signal,
            queue_lengths=queue_lengths,
            emergency_eta=self.emergency_eta,
            distance=self.emergency_distance,
        )
        self._rl_agent.update(state, action, reward, next_state)

        self.last_action = enacted_action
        self.last_reward = reward
        self.total_reward += reward
        self.decision_count += 1

        return enacted_action, reward, new_signal, queue_lengths

    # ══════════════════════════════════════════════════════
    # EMERGENCY TRIGGER
    # ══════════════════════════════════════════════════════

    async def trigger_emergency(self, zone, traffic, source="GPS"):
        """
        Called when ambulance GPS or the demo scheduler reports a location.
        """
        source = str(source).upper()
        if source not in ("GPS", "DEMO"):
            raise ValueError("Source must be GPS or DEMO")
        decision = self.emergency_manager.add_emergency(zone, traffic,source=source)
        decision_type = decision["decision"]
        request = decision["request"]
        # Captured once, up front, and threaded through every log call for
        # this emergency below — previously each call independently decided
        # whether/how to mention the ambulance, so reconstructing one
        # ambulance's full timeline from the log file meant manually
        # correlating by timestamp instead of filtering by id.
        amb_id = request.id if request else None

        app_logger.log_emergency_start(zone, traffic, ambulance_id=amb_id)
        app_logger.log_system_event(decision["message"], {"ambulance_id": amb_id,"source": source},)

        if decision_type == "merge":
            self.emergency_green_adaptive = self.emergency_manager.get_active_green_duration()
            app_logger.log_system_event(
                f"Green extended to {self.emergency_green_adaptive}s for Zone {zone}",
                {"ambulance_id": amb_id,"source": source},
            )
            return self.last_action, self.last_reward, self.current_zone

        # For activate, preempt, or queue — run RL agent
        active_zone = self.emergency_manager.get_active_zone()
        self.emergency_zone = active_zone
        self.emergency_eta = self.emergency_manager.get_active_eta()
        self.emergency_distance = self.emergency_manager.get_active_distance()
        self.emergency_green_adaptive = self.emergency_manager.get_active_green_duration()

        if decision_type in ["activate", "preempt"]:
            if decision_type == "preempt":
                # Signal any in-progress emergency hold (_run_emergency) to stop
                # immediately instead of finishing out the old zone's timer.
                self.emergency_preemption_active = True

            self.mode = "emergency"
            self.emergency_time = time.time()

            action, reward, new_signal, queue_lengths = self._rl_choose_emergency_zone(active_zone)

            app_logger.log_decision(
                action,
                self._rl_agent.get_best_action_name(action),
                reward,
                0,
                queue_lengths,
                mode="emergency",
                ambulance_id=amb_id,
            )

            try:
                await asyncio.to_thread(
                    db.save_emergency,
                    zone=active_zone,
                    traffic=traffic,
                    distance=self.emergency_distance,
                    eta=self.emergency_eta,
                    green_duration=self.emergency_green_adaptive,
                    action=action,
                    action_name=self._rl_agent.get_best_action_name(action),
                    reward=reward,
                    queue_lengths=queue_lengths,
                    ambulance_id=amb_id,
                    status=decision_type,
                )
            except Exception as e:
                app_logger.log_error(f"Database save_emergency failed: {e}")

            if decision_type == "preempt" and "preempted" in decision:
                try:
                    preempted_amb = decision["preempted"]
                    await asyncio.to_thread(
                        db.mark_preempted,
                        preempted_amb.id,
                        amb_id,
                    )
                except Exception as e:
                    app_logger.log_error(f"Database mark_preempted failed: {e}")

            return action, reward, new_signal

        else:
            # Queued
            app_logger.log_system_event(
                f"Ambulance queued: Zone {zone} (position {decision.get('queue_position', '?')})",
                {"ambulance_id": amb_id,"source": source},
            )
            return self.last_action, self.last_reward, self.current_zone

    # ══════════════════════════════════════════════════════
    # CLEAR EMERGENCY — Now async for proper transitions
    # ══════════════════════════════════════════════════════

    async def _clear_emergency(self):
        """
        Complete current emergency. Check queue for next ambulance.
        If queue has more, do FULL transition to next emergency zone.
        """
        # Defensive: a stray preemption flag (e.g. from a manual /clear
        # racing a just-arrived ambulance) must not abort the NEXT
        # emergency's hold before it even starts.
        self.emergency_preemption_active = False

        duration = time.time() - self.emergency_time if self.emergency_time else 0
        # Captured before complete_active() clears emergency_manager.active,
        # so the "cleared" log line can still be tied back to which
        # ambulance it was.
        completing_amb_id = (
            self.emergency_manager.active.id if self.emergency_manager.active else None
        )

        app_logger.log_emergency_end(self.emergency_zone, duration, ambulance_id=completing_amb_id)

        try:
            asyncio.create_task(
                asyncio.to_thread(db.resolve_emergency, self.emergency_zone, duration)
            )
        except Exception as e:
            app_logger.log_error(f"Database resolve_emergency failed: {e}")

        # Check for next ambulance in queue
        next_request = self.emergency_manager.complete_active()

        if next_request:
            # ─── TRANSITION TO NEXT EMERGENCY ─────────────
            app_logger.log_system_event(
                f"Next ambulance: {next_request.id} in Zone {next_request.zone} "
                f"(ETA {next_request.eta_seconds}s, Distance {next_request.distance_m}m)",
                {"ambulance_id": next_request.id},
            )
            self.emergency_zone = next_request.zone
            self.emergency_time = time.time()
            self.emergency_eta = next_request.eta_seconds
            self.emergency_distance = next_request.distance_m
            self.emergency_green_adaptive = next_request.green_duration

            # Make NEW RL decision for THIS emergency
            action, reward, new_signal, queue_lengths = self._rl_choose_emergency_zone(
                next_request.zone
            )

            app_logger.log_decision(
                action,
                self._rl_agent.get_best_action_name(action),
                reward,
                0,
                queue_lengths,
                mode="emergency",
                ambulance_id=next_request.id,
            )

            # Do FULL transition to new emergency zone
            target_zone = action + 1 if action < 4 else next_request.zone

            if target_zone != self.current_zone:
                await self._safe_transition(self.current_zone, target_zone, mode="emergency")

            # mode stays "emergency" — _run_emergency will handle the green hold

        else:
            # No more ambulances — return to normal
            app_logger.log_system_event("All emergencies resolved. Returning to normal mode.")
            self.mode = "normal"
            self.emergency_zone = None
            self.emergency_eta = 0
            self.emergency_distance = 0

    def clear_emergency(self):
        """Synchronous wrapper for backward compatibility (zone router uses this)."""
        self.emergency_preemption_active = False

        duration = time.time() - self.emergency_time if self.emergency_time else 0
        completing_amb_id = (
            self.emergency_manager.active.id if self.emergency_manager.active else None
        )
        app_logger.log_emergency_end(self.emergency_zone, duration, ambulance_id=completing_amb_id)

        try:
            asyncio.create_task(
                asyncio.to_thread(db.resolve_emergency, self.emergency_zone, duration)
            )
        except Exception as e:
            app_logger.log_error(f"Database resolve_emergency failed: {e}")

        next_request = self.emergency_manager.complete_active()

        if next_request:
            self.emergency_zone = next_request.zone
            self.emergency_time = time.time()
            self.emergency_eta = next_request.eta_seconds
            self.emergency_distance = next_request.distance_m
            self.emergency_green_adaptive = next_request.green_duration
        else:
            self.mode = "normal"
            self.emergency_zone = None
            self.emergency_eta = 0
            self.emergency_distance = 0

    # ══════════════════════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════════════════════

    async def run(self):
        self.running = True
        print("Traffic controller started - RL-driven continuous mode")

        # Set initial state through the single source of truth
        await self._apply_signal_state(
            zone=1,
            color="green",
            broadcast_mode="normal",
            phase="green",
            phase_description="Zone 1 active (startup)",
        )
        self.current_zone = 1

        # Schedule the first simulated emergency after a random 1–3 minutes
        if DEMO_MODE:
            self._next_demo_emergency_at = (
                time.monotonic() + random.randint(
                    DEMO_INTERVAL_MIN, DEMO_INTERVAL_MAX
                )
            )


        while self.running:
            try:
                if self.paused:
                    await asyncio.sleep(0.5)
                    continue

                    # Automatically generate simulated ambulance emergencies in demo mode
                if (
                    DEMO_MODE
                    and self._next_demo_emergency_at is not None
                    and time.monotonic() >= self._next_demo_emergency_at
                ):
                    # Schedule the next event before triggering the current one
                    self._next_demo_emergency_at = (
                        time.monotonic() + random.randint(
                            DEMO_INTERVAL_MIN, DEMO_INTERVAL_MAX
                        )
                    )

                    demo_zone = random.randint(1, 4)
                    demo_traffic = getattr(self.simulator, "traffic_pattern", "medium")

                    app_logger.log_system_event(
                        f"DEMO: Simulated ambulance generated for Zone {demo_zone}",
                        {"source": "DEMO", "zone": demo_zone},
                    )

                    await self.trigger_emergency(
                        zone=demo_zone,
                        traffic=demo_traffic,
                        source="DEMO",
                    )
                if self.mode == "normal":
                    await self._run_rl_cycle()
                elif self.mode == "emergency":
                    await self._run_emergency()
            except Exception as e:
                app_logger.log_error(f"Controller loop error: {e}")
                if self.mode == "emergency":
                    try:
                        self.clear_emergency()
                        app_logger.log_system_event("Forced emergency recovery after error")
                    except Exception:
                        self.mode = "normal"
                        self.emergency_zone = None
                await asyncio.sleep(1)

    async def _run_rl_cycle(self):
        """RL agent decides which zone gets green. Controller handles safe transition."""
        if self.mode != "normal":
            return

        action, reward, next_zone = self._rl_choose_next_zone()
        previous_zone = self.current_zone
        self.step_count += 1

        if self.mode != "normal":
            return

        # ─── STARVATION PROTECTION ────────────────────
        # If any zone hasn't been served for too long, override DQN
        # This prevents ping-ponging between 2 zones under asymmetric traffic
        for z in [1, 2, 3, 4]:
            self._zone_last_served[z] = self._zone_last_served.get(z, 0) + 1

        starved_zones = [
            z
            for z in [1, 2, 3, 4]
            if self._zone_last_served[z] >= self._starvation_limit
            and self._get_queue_lengths().get(z, 0) > 0
        ]

        if starved_zones:
            forced_zone = max(starved_zones, key=lambda z: self._zone_last_served[z])
            if forced_zone != next_zone:
                starvation_msg = (
                    f"Starvation override: Zone {forced_zone} not served for "
                    f"{self._zone_last_served[forced_zone]} cycles, forcing service"
                )
                app_logger.log_system_event(starvation_msg)
                self.xai_info["reason"] = starvation_msg
                self.xai_info["overridden"] = True
                self.xai_info["dqn_original_zone"] = next_zone
                self.xai_info["selected_zone"] = forced_zone
                next_zone = forced_zone

        # Reset counter for the zone being served
        self._zone_last_served[next_zone] = 0

        # If RL chose a different zone, execute safe transition
        # _safe_transition handles Arduino + state + broadcast
        if next_zone != previous_zone:
            await self._safe_transition(previous_zone, next_zone, mode="normal")

            if self.mode != "normal":
                return

        # Zone is now green — hold for green_duration
        # NO direct signal_states manipulation here!
        # State was already set correctly by _safe_transition
        # (or is still correct if same zone was chosen)
        self.current_zone = next_zone

        for tick in range(self.green_duration * 2):
            if self.mode != "normal":
                return
            if self.paused:
                await asyncio.sleep(0.5)
                continue

            self.simulator.step(
                active_signal=next_zone, is_green=True, current_time=self.step_count
            )

            # Broadcast current state (state is already correct from _apply_signal_state)
            await self._broadcast_full(
                mode="normal",
                zone=next_zone,
                color="green",
                reward=reward,
                phase="green",
                phase_description=f"Zone {next_zone} active",
            )

            await asyncio.sleep(0.5)

    async def _run_emergency(self):
        """
        Emergency mode with safe signal transition.
        All signal changes go through _apply_signal_state.
        """
        emergency_zone = self.emergency_zone
        previous_zone = self.current_zone

        # Determine target zone from RL decision. Fallback to previous_zone
        # (not emergency_zone) for Extend/Normal — matches
        # _rl_choose_emergency_zone's own definition of "Extend" (stay at
        # the current signal) and the reward function's semantics
        # (_emergency_reward scores action==4 relative to prev_signal, not
        # to the emergency zone). The two functions used to disagree on
        # what action 4/5 means, which could desync the frontend's
        # premature broadcast (routers/zone.py) from what _run_emergency
        # actually enacts.
        target_zone = self.last_action + 1 if 0 <= self.last_action < 4 else previous_zone

        app_logger.log_system_event(
            f"Emergency transition: Zone {previous_zone} -> Zone {target_zone} "
            f"(ambulance in Zone {emergency_zone})"
        )

        try:
            # ─── PHASE 1: SAFE TRANSITION ────────────────
            if target_zone != previous_zone:
                await self._safe_transition(previous_zone, target_zone, mode="emergency")
                if self.mode != "emergency":
                    return

            if self.emergency_preemption_active:
                # A newer, higher-priority ambulance preempted us while this
                # transition was still in flight. Bail out without holding —
                # run() immediately re-invokes _run_emergency(), which reads
                # the fresh self.emergency_zone/self.last_action set by the
                # preempting trigger_emergency() call.
                #
                # MUST reset the flag here (previously didn't): otherwise it
                # stays True forever after the first preemption, and every
                # subsequent _run_emergency() call — even a fully legitimate,
                # non-preempted one — hits this same check and bails out
                # instantly before ever applying a signal, forever. That
                # produced a silent, zero-exception busy loop (reproduced:
                # 180k+ iterations/sec) where the emergency is never served
                # and never cleared.
                self.emergency_preemption_active = False
                return

            # ─── PHASE 2: PRIORITY GREEN HOLD ────────────
            # Send green command through single source of truth
            self.current_zone = target_zone
            await self._apply_signal_state(
                zone=target_zone,
                color="green",
                broadcast_mode="emergency",
                phase="emergency_hold",
                phase_description=f"Ambulance priority - Zone {target_zone}",
            )

            app_logger.log_system_event(
                f"Priority green: Zone {target_zone} for {self.emergency_green_adaptive}s "
                f"(ETA: {self.emergency_eta}s, Distance: {self.emergency_distance}m)"
            )

            elapsed = 0.0
            # Re-read self.emergency_green_adaptive and check the preemption
            # flag every tick, rather than caching them, so that (a) a closer
            # ambulance can interrupt this hold immediately instead of
            # waiting out the old zone's full timer, and (b) a same-zone
            # merge (which extends self.emergency_green_adaptive mid-hold)
            # actually extends the physical green time.
            while (
                self.mode == "emergency"
                and not self.emergency_preemption_active
                and elapsed < self.emergency_green_adaptive
            ):
                self.simulator.step(
                    active_signal=target_zone, is_green=True, current_time=self.step_count
                )

                try:
                    remaining_eta = max(0, self.emergency_eta - elapsed)
                    # Live-updating clearance prediction: the zone already
                    # has green during this hold (zone_is_served=True), so
                    # this tracks whether the REMAINING queue will finish
                    # clearing before the ambulance's ETA counts down to 0
                    # — a continuously-refreshed answer to "will it still
                    # have to stop," not just the one-shot value computed
                    # at trigger time.
                    live_queue = self._get_queue_lengths().get(target_zone, 0)
                    clearance = estimate_clearance(
                        remaining_eta=remaining_eta,
                        queue_length=live_queue,
                        zone_is_served=True,
                    )
                    self.emergency_clearance_deficit = clearance["deficit"]
                    self.emergency_predicted_stop = clearance["will_stop"]

                    await self._broadcast_full(
                        mode="emergency",
                        zone=target_zone,
                        color="green",
                        reward=self.last_reward,
                        phase="emergency_hold",
                        phase_description=f"Ambulance priority - Zone {target_zone}",
                        extra={
                            "emergency_phase": "priority_hold",
                            "emergency_eta": round(remaining_eta, 1),
                            "emergency_distance": self.emergency_distance,
                            "emergency_green_total": self.emergency_green_adaptive,
                            "emergency_green_remaining": round(
                                self.emergency_green_adaptive - elapsed, 1
                            ),
                            "emergency_clearance_deficit": clearance["deficit"],
                            "emergency_predicted_stop": clearance["will_stop"],
                        },
                    )
                except Exception as e:
                    app_logger.log_error(f"Emergency broadcast error: {e}")

                await asyncio.sleep(0.5)
                elapsed += 0.5

            # ─── PHASE 3: RECOVERY ───────────────────────
            if self.emergency_preemption_active:
                # Consume the flag and hand off to the newer emergency.
                # Do NOT call _clear_emergency() here — the interrupted
                # ambulance was already pushed back into the queue by
                # EmergencyManager.add_emergency(), and the preempting
                # request is already self.emergency_manager.active.
                # Calling complete_active() here would wrongly mark the
                # never-served preempting ambulance as completed.
                self.emergency_preemption_active = False
                app_logger.log_system_event(
                    f"Zone {target_zone} PREEMPTED after {elapsed:.1f}s — "
                    f"switching to higher-priority ambulance in Zone {self.emergency_zone}"
                )
                return

            if self.mode == "emergency":
                app_logger.log_system_event(
                    f"Ambulance passed Zone {target_zone} | " f"Green held for {elapsed:.1f}s"
                )
                # Use async version for proper queue transition
                await self._clear_emergency()

        except Exception as e:
            app_logger.log_error(f"Emergency handler error: {e}")
            if self.mode == "emergency":
                self.clear_emergency()

    # ══════════════════════════════════════════════════════
    # BROADCAST — Reads signal_states (never modifies it)
    # ══════════════════════════════════════════════════════

    async def _broadcast_full(
        self, mode, zone, color, reward, phase="green", phase_description="", extra=None
    ):
        """Broadcast complete simulation state. Never raises exceptions."""
        try:
            metrics = self.simulator.get_metrics()
            system = self.simulator.get_system_metrics()
            queue_lengths = {z: metrics[z].queue_length for z in [1, 2, 3, 4]}

            # Record to metrics collector
            self.metrics_collector.record_step(
                active_signal=zone,
                queue_lengths=queue_lengths,
                reward=reward,
                system_metrics=system,
                emergency_active=(mode == "emergency"),
                ambulance_zone=self.emergency_zone or 0,
                action=self.last_action,
            )

            # Periodic DB snapshots — actually offloaded to a worker thread
            # now (asyncio.create_task(asyncio.to_thread(...)), matching
            # _clear_emergency's db.resolve_emergency below). The comment
            # here previously said "non-blocking" but the call was a
            # direct, synchronous db.save_snapshot(...) — genuinely
            # blocking the event loop inline in this hot per-tick
            # broadcast path, unlike this class's other DB writes.
            try:
                if self.step_count % 100 == 0 and self.step_count > 0:
                    asyncio.create_task(
                        asyncio.to_thread(
                            db.save_snapshot,
                            step=self.step_count,
                            mode=mode,
                            pattern=self.simulator.traffic_pattern,
                            signal=zone,
                            queue_lengths=queue_lengths,
                            wait_time=system.get("avg_wait_time_all", 0),
                            congestion=system.get("avg_congestion", 0),
                            cleared=system.get("total_vehicles_cleared", 0),
                            fairness=self.metrics_collector._compute_fairness(),
                        )
                    )
            except Exception:
                pass

            # Build live metrics
            live = self.metrics_collector.get_current_metrics(self.emergency_manager.get_stats())

            action_name = "Initializing"
            if self.last_action >= 0:
                action_name = self._rl_agent.get_best_action_name(self.last_action)

            # Build GPS info safely
            gps_info = {}
            try:
                if self.gps_model.current_zone:
                    gps_info = self.gps_model.get_status()
            except Exception:
                pass

            if self._broadcast:
                broadcast_data = {
                    "step": self.step_count,
                    "mode": mode,
                    "zone": zone,
                    "signal_color": color,
                    "active_signal": zone,
                    "emergency_zone": self.emergency_zone,
                    "phase": phase,
                    "phase_description": phase_description,
                    "signal_states": dict(self.signal_states),
                    "action": self.last_action,
                    "action_name": action_name,
                    "previous_zone": self.previous_zone,
                    "reward": round(float(reward), 2),
                    "xai": dict(self.xai_info),
                    "traffic": str(self.simulator.traffic_pattern),
                    "queue_lengths": queue_lengths,
                    "green_duration": self.green_duration,
                    "yellow_duration": self.yellow_duration,
                    "all_red_duration": self.all_red_duration,
                    "arduino_ack": str(self._serial_bridge.last_ack or ""),
                    "duration_s": (
                        self.emergency_green_adaptive
                        if mode == "emergency"
                        else self.green_duration
                    ),
                    "gps_info": gps_info,
                    "emergency_eta": float(self.emergency_eta or 0),
                    "emergency_distance": float(self.emergency_distance or 0),
                    "emergency_green_adaptive": int(self.emergency_green_adaptive or 15),
                    "emergency_queue": self.emergency_manager.get_status(),
                    "system_metrics": system,
                    "live_metrics": {
                        "avg_wait_time": live.get("avg_wait_time", 0),
                        "avg_queue_length": live.get("avg_queue_length", 0),
                        "total_vehicles_cleared": live.get("total_vehicles_cleared", 0),
                        "clearance_rate": live.get("clearance_rate", 0),
                        "emergency_accuracy": live.get("emergency_accuracy", 0),
                        "fairness_index": live.get("fairness_index", 1),
                        "avg_congestion": live.get("avg_congestion", 0),
                        "total_switches": live.get("total_switches", 0),
                        "emergency_events": live.get("emergency_events", 0),
                        "avg_emergency_response_time": live.get("avg_emergency_response_time", 0),
                    },
                }

                if extra:
                    for k, v in extra.items():
                        broadcast_data[k] = v

                await self._broadcast(broadcast_data)

        except Exception as e:
            app_logger.log_error(f"Broadcast error: {e}")

    async def broadcast_state(
        self, mode, zone, color, reward, phase="green", phase_description="", extra=None
    ):
        """
        Public entry point for broadcasting a full state snapshot from
        outside this class (routers, etc.). Thin wrapper around
        _broadcast_full so external callers have one documented, stable
        way to reuse this class's broadcast payload construction instead
        of hand-building their own copy of it — see routers/zone.py, which
        previously duplicated ~30 keys of this payload independently in
        two places, an easy way for the two copies to silently drift.
        """
        await self._broadcast_full(
            mode=mode,
            zone=zone,
            color=color,
            reward=reward,
            phase=phase,
            phase_description=phase_description,
            extra=extra,
        )
