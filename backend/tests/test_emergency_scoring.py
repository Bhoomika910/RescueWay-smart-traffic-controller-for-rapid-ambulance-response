# tests/test_emergency_scoring.py
"""
Regression tests for rl/emergency_scoring.py — the shared function that
replaced nine independently-duplicated (and, in most cases, buggy) inline
implementations of "was this action correct during an emergency" (see
ARCHITECTURE.md Part 9 and rl/emergency_scoring.py's module docstring for
the full incident this module exists to prevent from recurring).

These are exactly the tests that, had they existed before, would have
caught the original bug immediately: it undercounted action==4 ("extend
current green") as incorrect even when the extended zone already matched
the emergency zone.
"""
import pytest

from rl.emergency_scoring import is_emergency_action_correct, resolved_zone


class TestIsEmergencyActionCorrect:
    @pytest.mark.parametrize("action,emergency_zone", [(0, 1), (1, 2), (2, 3), (3, 4)])
    def test_direct_switch_to_correct_zone_is_correct(self, action, emergency_zone):
        # served_zone is irrelevant for actions 0-3 — the switch itself
        # determines the served zone, regardless of what was green before.
        assert is_emergency_action_correct(action, emergency_zone, served_zone=99) is True

    @pytest.mark.parametrize("action,emergency_zone", [(0, 2), (1, 4), (2, 1), (3, 3)])
    def test_direct_switch_to_wrong_zone_is_incorrect(self, action, emergency_zone):
        assert is_emergency_action_correct(action, emergency_zone, served_zone=99) is False

    def test_extend_action_already_serving_emergency_zone_is_correct(self):
        # This is the exact case the original bug got wrong: action==4
        # while already serving the emergency zone must count as correct.
        assert is_emergency_action_correct(action=4, emergency_zone=2, served_zone=2) is True

    def test_extend_action_serving_wrong_zone_is_incorrect(self):
        assert is_emergency_action_correct(action=4, emergency_zone=2, served_zone=3) is False

    def test_normal_cycle_action_already_serving_emergency_zone_is_correct(self):
        # action==5 ("normal cycle") also leaves the current zone
        # unchanged (see rl/reward.py's module comment) and must be
        # scored identically to action==4 in this respect — the original
        # nine-site bug didn't even special-case action==4 consistently,
        # let alone action==5.
        assert is_emergency_action_correct(action=5, emergency_zone=3, served_zone=3) is True

    def test_normal_cycle_action_serving_wrong_zone_is_incorrect(self):
        assert is_emergency_action_correct(action=5, emergency_zone=3, served_zone=1) is False

    def test_agrees_with_safety_shield_definition_of_already_correct(self):
        # rl/safety_shield.py's shield_emergency_action() independently
        # encodes the same "already correct" concept for a different
        # purpose (deciding whether to override). The two must never
        # disagree, or the live system and the benchmark could credit
        # different actions as correct for the same scenario.
        from rl.safety_shield import shield_emergency_action

        for action in range(6):
            for emergency_zone in range(1, 5):
                for served_zone in range(1, 5):
                    _, overridden = shield_emergency_action(action, emergency_zone, served_zone)
                    shield_says_already_correct = not overridden
                    scoring_says_correct = is_emergency_action_correct(
                        action, emergency_zone, served_zone
                    )
                    assert shield_says_already_correct == scoring_says_correct, (
                        f"disagreement at action={action}, emergency_zone={emergency_zone}, "
                        f"served_zone={served_zone}"
                    )


class TestResolvedZone:
    @pytest.mark.parametrize("action", [0, 1, 2, 3])
    def test_switch_actions_resolve_to_action_plus_one(self, action):
        assert resolved_zone(action, current_zone=99) == action + 1

    @pytest.mark.parametrize("action", [4, 5])
    def test_extend_and_normal_resolve_to_current_zone(self, action):
        assert resolved_zone(action, current_zone=3) == 3
