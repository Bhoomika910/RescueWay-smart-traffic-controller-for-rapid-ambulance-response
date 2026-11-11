# rl/emergency_scoring.py
"""
Single source of truth for "was this tick's action correct during an
active emergency" — the exact question every training loop, evaluation
script, and benchmark in this project needs to answer, and the exact
question nine independent, hand-copied implementations answered
*differently* until this module existed.

Why this module exists (design-review finding, category 16 "Software
architecture" / category 23 "Maintainability"): `rl/safety_shield.py`
already defined the correct notion of "already correct" for the live
safety-shield override (action == target zone, OR action == 4/"extend"
while already serving the target zone). Nine other call sites — the
training loops, evaluation harnesses, and benchmark scripts — each
independently re-implemented a *simpler, wrong* version that only checked
`action == target_action`, silently never crediting a correct "extend"
decision. That bug shipped in every headline accuracy number this project
reported for an unknown period, through at least two prior audit passes,
before being caught (see ARCHITECTURE.md Part 9). `evaluation/
robustness_runner.py` happened to have the correct check independently
implemented (and its own comment pointed later readers at
`metrics/collector.py` as "the version to match") — proof that the
correct logic existed in the codebase the whole time, just not in one
place all nine call sites could share.

This is the fix for the systemic cause, not just the nine individual
symptoms: one function, imported everywhere. A future tenth call site
gets this by construction instead of by copying it correctly from memory.
"""
from __future__ import annotations

EXTEND_ACTION = 4


def is_emergency_action_correct(action: int, emergency_zone: int, served_zone: int) -> bool:
    """
    Did `action` result in the emergency zone actually being served?

    Args:
        action: the controller's chosen action (0-5). 0-3 = switch to
            zone 1-4, 4 = extend current green, 5 = normal cycle.
        emergency_zone: 1-4, the zone the active ambulance is in/reported
            in. Caller must guard `emergency_zone > 0` before calling —
            this function assumes an emergency is actually active, since
            "correct" is meaningless otherwise.
        served_zone: which zone actually ends up green as a result of this
            tick's action. For action 0-3 this is always `action + 1`; for
            action 4/5 it's whatever zone was already green (the caller's
            `current_zone`/`active_signal`/`env.active_signal`, read at
            the appropriate point in the step — see the two call-timing
            notes below). Passed in rather than recomputed here because
            call sites disagree on when in their step they still have
            access to the *pre-action* current zone vs. having already
            mutated it — the caller is in the best position to know which
            one is correct for its own control flow.

    Returns:
        True iff the zone actually being served this tick equals the
        emergency zone — the only definition of "correct" that matches
        what the ambulance experiences, as opposed to "did the action
        literally spell out a zone-switch instruction."

    Call-timing note for action 4/5 (extend/normal): `served_zone` must be
    the zone that STAYS green as a result of this action, i.e. the
    pre-action current zone. Get this wrong (e.g. pass a post-action zone
    that's already been reset to something else) and every "extend"
    decision will be silently miscounted — exactly the bug this module
    replaces nine copies of.
    """
    if action != EXTEND_ACTION and action < 4:
        return (action + 1) == emergency_zone
    return served_zone == emergency_zone


def resolved_zone(action: int, current_zone: int) -> int:
    """
    Which zone ends up green as a result of `action`, given the zone that
    was green before this tick. Shared helper so callers that need both
    `resolved_zone()` (for logging/state) and `is_emergency_action_correct()`
    (for scoring) compute the "what zone is this" question identically —
    a second, smaller instance of the same duplication risk this module
    exists to close.
    """
    return action + 1 if action < 4 else current_zone
