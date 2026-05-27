"""
Unit tests for the public check_answer() API.
Tests method resolution, output schema, and flag behavior.
NLI skipped in fast tests to avoid model loading overhead.
"""

import pytest
from veracitor import check_answer
from veracitor.checker import _resolve_methods
from veracitor.models import CheckResult, ConfidenceLevel

SOURCE = (
    "Apple Inc. reported total net sales of $391 billion for fiscal year 2025, "
    "representing a 4% increase compared to the prior year."
)


# --- Method alias resolution tests ---

class TestMethodAliases:

    def test_nli_alias(self):
        assert _resolve_methods(["nli"]) == ["nli_entailment"]

    def test_citation_alias(self):
        assert _resolve_methods(["citation"]) == ["citation_alignment"]

    def test_judge_alias(self):
        assert _resolve_methods(["judge"]) == ["llm_judge"]

    def test_span_alias(self):
        assert _resolve_methods(["span"]) == ["span_coverage"]

    def test_consistency_alias(self):
        assert _resolve_methods(["consistency"]) == ["self_consistency"]

    def test_deduplication(self):
        assert _resolve_methods(["nli", "nli_entailment"]) == ["nli_entailment"]

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            _resolve_methods(["invalid_method"])

    def test_multiple_methods(self):
        result = _resolve_methods(["citation", "span"])
        assert result == ["citation_alignment", "span_coverage"]


# --- Public API tests (citation + span only, no model loading) ---

class TestCheckAnswer:

    def test_clean_pair_passes(self):
        result = check_answer(
            question="What was Apple revenue?",
            answer="Apple reported total net sales of $391 billion in fiscal year 2025.",
            sources=[SOURCE],
            methods=["citation", "span"],
        )
        assert isinstance(result, CheckResult)
        assert result.flagged is False
        assert result.confidence == ConfidenceLevel.PASS
        assert result.overall_score > 0.5

    def test_faithfulness_violation_flagged(self):
        result = check_answer(
            question="What was Apple revenue?",
            answer="Apple reported total net sales of $450 billion in fiscal year 2025.",
            sources=[SOURCE],
            methods=["citation", "span"],
        )
        assert isinstance(result, CheckResult)
        assert result.flagged is True

    def test_output_schema(self):
        result = check_answer(
            question="What was Apple revenue?",
            answer="Apple reported total net sales of $391 billion.",
            sources=[SOURCE],
            methods=["citation"],
        )
        assert isinstance(result.flagged, bool)
        assert isinstance(result.overall_score, float)
        assert 0.0 <= result.overall_score <= 1.0
        assert isinstance(result.methods_used, list)
        assert isinstance(result.method_breakdown, dict)
        assert isinstance(result.flagged_claims, list)
        assert result.confidence in list(ConfidenceLevel)

    def test_multiple_sources_concatenated(self):
        result = check_answer(
            question="What was Apple revenue?",
            answer="Apple reported total net sales of $391 billion in fiscal year 2025.",
            sources=[
                "Apple Inc. reported total net sales of $391 billion",
                "for fiscal year 2025, a 4% increase year over year.",
            ],
            methods=["citation"],
        )
        assert result.flagged is False

    def test_methods_used_in_output(self):
        result = check_answer(
            question="What was Apple revenue?",
            answer="Apple reported $391 billion in revenue.",
            sources=[SOURCE],
            methods=["citation", "span"],
        )
        assert "citation_alignment" in result.methods_used
        assert "span_coverage" in result.methods_used
        assert "citation_alignment" in result.method_breakdown
        assert "span_coverage" in result.method_breakdown