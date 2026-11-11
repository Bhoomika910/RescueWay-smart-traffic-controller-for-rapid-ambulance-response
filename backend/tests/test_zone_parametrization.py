# tests/test_zone_parametrization.py
"""
Proves the RL state/action encoding scheme (rl/state_features.py) is
zone-count-agnostic, not just at the actual deployment's N_ZONES=4 —
design-review finding (category 24 "Scalability"): "the number 4 (zones)
is hard-coded pervasively ... adding a 5th zone ... would require code
changes in many places, not a config change."

Scope (see gps/zone_model.py's N_ZONES docstring and rl/state_features.py's
module docstring for the full reasoning): this tests the RL ENCODING layer
specifically — state vector construction and action-space sizing — by
constructing a StateFeatureExtractor at N_ZONES=6 (a zone count that has
never existed in this project) and verifying the resulting state vector
is correctly shaped and populated. It does NOT claim the physical system
(Arduino firmware, SUMO network, frontend) works at N_ZONES=6 — those
remain built for the real 4-zone deployment, a separate scope.
"""
import numpy as np
import pytest

from gps.zone_model import N_ZONES
from rl.state_features import StateFeatureExtractor


class TestActualDeploymentUnaffected:
    """The instance-level n_zones override must not change anything for
    every real caller in the codebase, which never passes it."""

    def test_default_extractor_uses_the_real_deployment_zone_count(self):
        extractor = StateFeatureExtractor()
        assert extractor.n_zones == N_ZONES == 4
        assert extractor.state_dim == 19

    def test_module_level_constants_match_default_extractor(self):
        from rl.state_features import N_ACTIONS, STATE_DIM

        assert STATE_DIM == 19
        assert N_ACTIONS == 6


class TestHypotheticalSixZoneIntersection:
    """A concrete existence proof: N_ZONES=6 (never used anywhere else in
    this project) produces a correctly-shaped, correctly-populated state
    vector using the exact same StateFeatureExtractor class and extract()
    method the real 4-zone deployment uses."""

    def test_state_dim_formula_holds_at_six_zones(self):
        extractor = StateFeatureExtractor(n_zones=6)
        assert extractor.state_dim == 3 * 6 + 7 == 25

    def test_extracted_vector_has_the_expected_shape(self):
        extractor = StateFeatureExtractor(n_zones=6)
        state = extractor.extract(
            emergency_active=False,
            ambulance_zone=0,
            active_signal=1,
            queue_lengths={i: 5 for i in range(1, 7)},
        )
        assert state.shape == (25,)
        assert state.dtype == np.float32

    def test_queue_features_land_in_the_first_six_slots(self):
        extractor = StateFeatureExtractor(n_zones=6)
        queues = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 0}
        state = extractor.extract(
            emergency_active=False, ambulance_zone=0, active_signal=1, queue_lengths=queues
        )
        for i in range(6):
            expected = min(1.0, queues[i + 1] / 50)
            assert state[i] == pytest.approx(expected)

    def test_active_signal_one_hot_lands_in_slots_six_through_eleven(self):
        extractor = StateFeatureExtractor(n_zones=6)
        state = extractor.extract(
            emergency_active=False,
            ambulance_zone=0,
            active_signal=5,  # zone 5 of 6 — only valid because n_zones=6
            queue_lengths={i: 0 for i in range(1, 7)},
        )
        signal_one_hot = state[6:12]
        assert signal_one_hot.sum() == 1.0
        assert signal_one_hot[4] == 1.0  # zone 5 -> index 4

    def test_ambulance_zone_one_hot_lands_after_emergency_flag(self):
        extractor = StateFeatureExtractor(n_zones=6)
        state = extractor.extract(
            emergency_active=True,
            ambulance_zone=6,  # the 6th zone — doesn't exist at n_zones=4
            active_signal=1,
            queue_lengths={i: 0 for i in range(1, 7)},
        )
        assert state[12] == 1.0  # emergency flag at index 2*6=12
        amb_one_hot = state[13:19]  # 2*6+1 : 3*6+1
        assert amb_one_hot.sum() == 1.0
        assert amb_one_hot[5] == 1.0  # zone 6 -> index 5

    def test_action_space_has_eight_actions(self):
        extractor = StateFeatureExtractor(n_zones=6)
        assert extractor._layout.n_actions == 8  # 6 zones + Extend + Normal
        assert extractor._layout.action_names[6] == "Extend current green"
        assert extractor._layout.action_names[7] == "Normal cycle"
        assert extractor._layout.action_names[0] == "Green Zone 1"
        assert extractor._layout.action_names[5] == "Green Zone 6"

    def test_scaling_from_four_to_six_zones_only_changes_zone_dependent_dims(self):
        four = StateFeatureExtractor(n_zones=4)
        six = StateFeatureExtractor(n_zones=6)
        # state_dim = 3*n + 7: going from 4 to 6 zones adds exactly 3*(6-4)=6 dims
        assert six.state_dim - four.state_dim == 6
        # action space = n + 2: going from 4 to 6 zones adds exactly 2 actions
        assert six._layout.n_actions - four._layout.n_actions == 2
