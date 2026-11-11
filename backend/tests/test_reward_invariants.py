# tests/test_reward_invariants.py
"""
Unit tests for rl/reward.py's documented invariants — design-review
finding (category 27 "Testing"): "zero unit tests for the RL core itself
... no test exists for rl/reward.py." These specifically exercise the
properties rl/reward.py's own module comments claim hold (delay capping,
reward clipping, extend/normal-cycle parity with an explicit re-selection)
rather than just checking "a number came out."
"""
from rl.reward import (
    DEFAULT_WEIGHTS,
    MAX_DELAY_FOR_PENALTY,
    REWARD_CLIP,
    _emergency_reward,
    compute_reward,
)


class TestAmbulanceDelayCapping:
    def test_delay_at_or_under_half_second_gives_full_no_stop_bonus(self):
        r_stopped = _emergency_reward(action=0, amb_zone=1, prev_signal=1, ambulance_delay=0.0)
        r_zero_delay = _emergency_reward(action=0, amb_zone=1, prev_signal=1, ambulance_delay=0.5)
        assert r_stopped == r_zero_delay  # both count as "passed through"

    def test_penalty_scales_with_delay_up_to_the_cap(self):
        r_5s = _emergency_reward(action=0, amb_zone=1, prev_signal=1, ambulance_delay=5.0)
        r_10s = _emergency_reward(action=0, amb_zone=1, prev_signal=1, ambulance_delay=10.0)
        assert r_10s < r_5s  # more delay must never score better

    def test_delay_beyond_cap_does_not_worsen_penalty_further(self):
        # MAX_DELAY_FOR_PENALTY exists specifically so one SUMO-measured
        # 60-100s gridlock spike doesn't dwarf every other reward term and
        # destabilize the TD target — see rl/reward.py's module comment.
        r_at_cap = _emergency_reward(
            action=0, amb_zone=1, prev_signal=1, ambulance_delay=MAX_DELAY_FOR_PENALTY
        )
        r_way_over_cap = _emergency_reward(
            action=0, amb_zone=1, prev_signal=1, ambulance_delay=MAX_DELAY_FOR_PENALTY * 20
        )
        assert r_at_cap == r_way_over_cap

    def test_no_delay_argument_preserves_original_pre_delay_behavior(self):
        # ambulance_delay=None (the default) must be a strict no-op vs.
        # never having added the delay term at all — this is the
        # regression check the proactive-delay redesign (ARCHITECTURE.md
        # Part 2) was verified against before any other change was made.
        with_none = _emergency_reward(action=0, amb_zone=1, prev_signal=1, ambulance_delay=None)
        without_arg = _emergency_reward(action=0, amb_zone=1, prev_signal=1)
        assert with_none == without_arg == 25.0  # base correct-zone reward, unmodified


class TestExtendNormalCycleParity:
    """Actions 4 (Extend) and 5 (Normal cycle) both leave the currently-
    green zone unchanged and must be scored identically to an explicit
    re-selection of that same zone (rl/reward.py's module-level comment) —
    otherwise they're a risk-free, zero-variance fallback relative to
    genuinely committing to a zone, which is the exact reward-hacking
    loophole this project's own history already found and fixed once."""

    def test_extend_scores_identically_to_explicit_reselection_of_same_zone(self):
        queues = {1: 20, 2: 5, 3: 3, 4: 8}
        r_extend = compute_reward(
            action=4, amb_zone=0, prev_signal=1, traffic="high", queue_lengths=queues
        )
        r_explicit_reselect = compute_reward(
            action=0, amb_zone=0, prev_signal=1, traffic="high", queue_lengths=queues
        )
        assert r_extend == r_explicit_reselect

    def test_normal_cycle_scores_identically_to_explicit_reselection_of_same_zone(self):
        queues = {1: 20, 2: 5, 3: 3, 4: 8}
        r_normal = compute_reward(
            action=5, amb_zone=0, prev_signal=1, traffic="high", queue_lengths=queues
        )
        r_explicit_reselect = compute_reward(
            action=0, amb_zone=0, prev_signal=1, traffic="high", queue_lengths=queues
        )
        assert r_normal == r_explicit_reselect


class TestRewardClipping:
    def test_total_reward_never_exceeds_the_clip_bounds(self):
        # A deliberately extreme, physically implausible input (huge queue
        # imbalance + large delay) to try to blow past REWARD_CLIP.
        r = compute_reward(
            action=0,
            amb_zone=1,
            prev_signal=4,
            traffic="high",
            queue_lengths={1: 50, 2: 0, 3: 0, 4: 0},
            ambulance_delay=1000.0,
        )
        assert -REWARD_CLIP <= r <= REWARD_CLIP


class TestEmergencyZoneCorrectness:
    def test_correct_zone_beats_ignoring_beats_actively_choosing_wrong_zone(self):
        # Ordering per rl/reward.py's actual weights: correctly serving
        # the ambulance (+25) > passively "normal cycling" without
        # engaging the emergency (-15) > actively putting a DIFFERENT
        # wrong zone green while the ambulance waits (-25) — deliberately
        # steering away from the ambulance is worse than simply not
        # reacting.
        correct = _emergency_reward(action=0, amb_zone=1, prev_signal=3)
        ignored = _emergency_reward(action=5, amb_zone=1, prev_signal=3)
        actively_wrong = _emergency_reward(action=2, amb_zone=1, prev_signal=3)
        assert correct > ignored > actively_wrong

    def test_no_ambulance_gives_zero_emergency_reward(self):
        assert _emergency_reward(action=0, amb_zone=0, prev_signal=1) == 0.0

    def test_action_5_during_emergency_is_penalized_even_when_zone_already_correct(self):
        """Characterization test, not a correctness assertion: documents a
        real asymmetry found while writing this suite. Action 4 ("extend")
        gets +10 when prev_signal already equals the emergency zone, but
        action 5 ("normal cycle") always scores -15 during an active
        emergency regardless of prev_signal — even though both actions
        produce the IDENTICAL resulting signal state everywhere else in
        the codebase (chosen_zone = prev_signal for both; see
        rl/emergency_scoring.py, which treats them identically for
        accuracy-measurement and safety-shield purposes specifically
        because outcome, not action label, is what the ambulance
        experiences).

        Plausible reading: this is intentional reward shaping — action 5
        represents "not actively engaging with the emergency" (falling
        back to default cycling) as opposed to action 4's affirmative
        "I am choosing to keep serving this zone," and the reward function
        wants to penalize passivity regardless of a lucky coincidental
        outcome. Equally plausible reading: this is a second instance of
        the same asymmetric-action-4-handling bug already found and fixed
        in nine other places (ARCHITECTURE.md Part 9) that simply wasn't
        caught because it lives in the reward function rather than a
        scoring/counting function.

        Deliberately NOT changed as part of this pass: rl/reward.py drives
        actual gradient updates, so altering it would change the trained
        policy and require full retraining to validate — too invasive to
        change on an inference from adjacent (but distinct) code without
        stronger evidence of intent either way. This test locks in current
        behavior so a future change here is a deliberate, reviewed
        decision, not an accidental side effect of an unrelated refactor.
        """
        already_correct_via_normal_cycle = _emergency_reward(action=5, amb_zone=2, prev_signal=2)
        already_correct_via_extend = _emergency_reward(action=4, amb_zone=2, prev_signal=2)
        assert already_correct_via_normal_cycle == -15.0
        assert already_correct_via_extend == 10.0
        assert already_correct_via_normal_cycle != already_correct_via_extend


class TestRewardWeights:
    """Tests for the RewardWeights refactor itself (rl/reward.py) — the
    weights-as-dataclass extraction that makes evaluation/
    reward_sensitivity.py possible. Behavior-preservation for existing
    callers is the primary requirement here."""

    def test_default_weights_match_original_hardcoded_values(self):
        assert DEFAULT_WEIGHTS.emergency_correct_zone == 25.0
        assert DEFAULT_WEIGHTS.emergency_wrong_zone == -25.0
        assert DEFAULT_WEIGHTS.switching_penalty == -1.5

    def test_compute_reward_without_weights_arg_uses_defaults(self):
        # Every existing call site in the codebase calls compute_reward()
        # without `weights=` — this must be indistinguishable from
        # passing DEFAULT_WEIGHTS explicitly.
        queues = {1: 10, 2: 5, 3: 3, 4: 8}
        implicit = compute_reward(
            action=0, amb_zone=1, prev_signal=3, traffic="high", queue_lengths=queues
        )
        explicit = compute_reward(
            action=0,
            amb_zone=1,
            prev_signal=3,
            traffic="high",
            queue_lengths=queues,
            weights=DEFAULT_WEIGHTS,
        )
        assert implicit == explicit

    def test_perturbed_changes_only_the_named_weight(self):
        perturbed = DEFAULT_WEIGHTS.perturbed("emergency_correct_zone", 1.5)
        assert perturbed.emergency_correct_zone == DEFAULT_WEIGHTS.emergency_correct_zone * 1.5
        assert perturbed.switching_penalty == DEFAULT_WEIGHTS.switching_penalty  # unchanged

    def test_perturbed_weights_actually_change_computed_reward(self):
        baseline = _emergency_reward(action=0, amb_zone=1, prev_signal=3, weights=DEFAULT_WEIGHTS)
        perturbed_weights = DEFAULT_WEIGHTS.perturbed("emergency_correct_zone", 2.0)
        perturbed = _emergency_reward(
            action=0, amb_zone=1, prev_signal=3, weights=perturbed_weights
        )
        assert perturbed == baseline * 2.0

    def test_ablation_studies_monkeypatch_stays_compatible_with_weights_parameter(self):
        """Regression guard for the exact bug class ARCHITECTURE.md
        already documents fixing once (evaluation/ablation_studies.py's
        _patch_emergency_reward): adding a new parameter to the real
        _emergency_reward must not silently break the monkeypatch that
        stands in for it during Study A. Import-time only — doesn't run
        the full study, just confirms the patched function accepts the
        real function's current call signature."""
        import rl.reward as reward_module
        from evaluation.ablation_studies import _patch_emergency_reward

        original_fn = reward_module._emergency_reward
        try:
            _patch_emergency_reward(scale=1.5)
            # This is exactly how compute_reward() calls _emergency_reward
            # internally — must not raise.
            result = reward_module.compute_reward(
                action=0, amb_zone=1, prev_signal=3, traffic="high", ambulance_delay=2.0
            )
            assert isinstance(result, float)
        finally:
            reward_module._emergency_reward = original_fn
