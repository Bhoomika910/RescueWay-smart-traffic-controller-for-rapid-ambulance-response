# tests/test_experiment_registry.py
"""
Unit tests for evaluation/experiment_registry.py's classification/
summarization logic — design-review finding (category 9 "Training
pipeline"): "no experiment tracker ... results live only as timestamped
JSON files a human must locate and read."
"""
import os

from evaluation.experiment_registry import _infer_category, _summarize


class TestInferCategory:
    def test_subdirectory_file_uses_the_subdirectory_as_category(self):
        path = os.path.join("results", "robustness", "study_1_20260101_000000.json")
        assert _infer_category(path) == "robustness"

    def test_root_level_results_file_does_not_use_its_own_filename_as_category(self):
        # Regression guard: results/some_file.json (no subdirectory) was
        # briefly miscategorized as category="some_file.json" — the
        # filename itself — before this fix.
        path = os.path.join("results", "dqn_architecture_comparison.json")
        assert _infer_category(path) == "results_root"

    def test_sumo_training_logs_get_their_own_category(self):
        path = os.path.join("sumo", "training_logs", "multi_seed_variance_20260101.json")
        assert _infer_category(path) == "sumo_training_logs"


class TestSummarize:
    def test_extracts_environment_tag(self):
        assert "env=sumo" in _summarize({"environment": "sumo"})

    def test_extracts_seed_and_episodes(self):
        summary = _summarize({"seed": 42, "episodes": 1000})
        assert "seed=42" in summary
        assert "episodes=1000" in summary

    def test_falls_back_to_listing_keys_for_unrecognized_schema(self):
        summary = _summarize({"totally_custom_field": 1, "another_one": 2})
        assert summary is not None
        assert "totally_custom_field" in summary

    def test_returns_none_for_non_dict_payload(self):
        assert _summarize([1, 2, 3]) is None

    def test_handles_empty_dict_without_raising(self):
        # An empty dict has no fields to summarize — must not raise or
        # crash the registry build over one bad/empty file. The exact
        # fallback string ("keys=") isn't the point; not raising is.
        _summarize({})  # must not raise
