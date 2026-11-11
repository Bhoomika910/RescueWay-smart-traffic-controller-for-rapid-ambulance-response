# rl/safety_shield.py
"""
Emergency safety shield — a hard override on top of any learned policy's
chosen action, guaranteeing an active emergency's zone always gets served.

Why this exists (design-review finding): the review's reward-function
section recommended adding "a shielded/constrained action-selection layer
... [combining] DQN's general traffic-management competence with the
safety guarantee the Rule-Based controller already demonstrates" — noting
that the project's own benchmark data shows Rule-Based's hard-priority
approach beating DQN's learned-and-scalarized approach on the exact metric
that matters most (emergency accuracy). That layer already existed, but
only inline inside services/traffic_controller.py's
_rl_choose_emergency_zone ("EMERGENCY SAFETY OVERRIDE"), with no equivalent
applied anywhere in the benchmark/evaluation controllers
(controllers/dqn_controller.py, controllers/rl_controller.py). That meant:

  1. The live, deployed system ALREADY guarantees 100% emergency
     accuracy via this override — but a reader of the benchmark tables
     (which measure DQN's RAW, un-shielded action) would have no way to
     know that, and could easily mistake "DQN benchmarks at ~85%
     accuracy" for "production also gets it wrong ~15% of the time,"
     which is false.
  2. The override logic was duplicated risk waiting to happen — a second,
     independent implementation in the benchmark path would have drifted
     from this one silently over time, the same class of bug the design
     review found and fixed elsewhere (e.g. ablation_studies.py's reward
     constants).

Extracted here as the single shared implementation. Both
services/traffic_controller.py (the live system) and
controllers/dqn_controller.py's DQNShieldedController (an explicit,
honestly-labeled benchmark variant — NOT a replacement for the raw
DQNController, which remains the correct way to measure the learned
policy's own competence) now call this one function.

The "is this action already correct" question this function answers is
now delegated to rl/emergency_scoring.is_emergency_action_correct() rather
than reimplemented inline — this module used to have its own copy of that
logic, and a regression test written against this module (see
tests/test_emergency_scoring.py's cross-check) caught a real discrepancy:
this function special-cased action==4 ("extend") but not action==5
("normal cycle"), even though rl/reward.py's own module comment states
both actions leave the current zone unchanged and must be scored
identically. That meant the shield would report `overridden=True` for an
action==5 that was already serving the correct zone — mislabeling
`overridden` in the live xai_info/telemetry, even though the actually
enacted zone was correct either way (the shield's `target_action` for that
case resolves to the same zone action==5 was already maintaining). Fixed
by sharing one definition instead of two that could silently diverge.
"""
from rl.emergency_scoring import is_emergency_action_correct


def shield_emergency_action(
    action: int, emergency_zone: int, current_zone: int
) -> tuple[int, bool]:
    """
    Given a learned policy's raw chosen action, returns the ENACTED action
    after applying the emergency safety shield: during an active
    emergency, the ambulance's own zone always wins, regardless of what
    the policy chose — the same hard guarantee Rule-Based has by
    construction, applied as a wrapper around any learned policy instead
    of requiring a separate hand-coded controller.

    Args:
        action: the policy's raw chosen action (0-5)
        emergency_zone: 1-4 if an emergency is active, 0/falsy otherwise
        current_zone: the zone currently green (needed to recognize a
            valid "extend current green"/"normal cycle" action, i.e.
            action==4 or 5, that already happens to serve the emergency
            zone — that should NOT count as an override)

    Returns:
        (enacted_action, overridden): the action to actually apply, and
        whether it differs from the policy's raw chosen action.
    """
    if not emergency_zone or emergency_zone <= 0:
        return action, False

    target_action = emergency_zone - 1
    if is_emergency_action_correct(action, emergency_zone, current_zone):
        return action, False
    return target_action, True
