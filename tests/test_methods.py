"""
Unit tests for individual detection methods.
Tests schema correctness and basic flag behavior.
NLI model is not loaded in these tests to keep them fast.
"""

import pytest
from veracitor.methods.citation_alignment import (
    CitationAlignmentInput, score_pair as citation_score
)
from veracitor.methods.span_coverage import (
    SpanCoverageInput, score_pair as span_score
)
from veracitor.models import ConfidenceLevel, score_to_confidence

SOURCE = (
    "Apple Inc. reported total net sales of $391 billion for fiscal year 2025, "
    "representing a 4% increase compared to the prior year."
)

CLEAN_ANSWER = "Apple reported total net sales of $391 billion in fiscal year 2025."
FAITH_ANSWER = "Apple reported total net sales of $450 billion in fiscal year 2025."
REL_ANSWER = "Apple was founded in 1976 by Steve Jobs in Cupertino, California."


# --- Citation alignment tests ---

class TestCitationAlignment:

    def test_clean_pair_not_flagged(self):
        inp = CitationAlignmentInput(
            question="What was Apple revenue?",
            answer=CLEAN_ANSWER,
            source=SOURCE,
        )
        result = citation_score(inp)
        assert result.flagged is False
        assert result.citation_alignment_score > 0.5

    def test_faithfulness_violation_lower_score(self):
        inp = CitationAlignmentInput(
            question="What was Apple revenue?",
            answer=FAITH_ANSWER,
            source=SOURCE,
        )
        result = citation_score(inp)
        # Citation alignment is weak on subtle faithfulness violations
        # Score should be lower than clean but may not flag
        clean_inp = CitationAlignmentInput(
            question="What was Apple revenue?",
            answer=CLEAN_ANSWER,
            source=SOURCE,
        )
        clean_result = citation_score(clean_inp)
        assert result.citation_alignment_score < clean_result.citation_alignment_score

    def test_relevance_violation_flagged(self):
        inp = CitationAlignmentInput(
            question="What was Apple revenue?",
            answer=REL_ANSWER,
            source=SOURCE,
        )
        result = citation_score(inp)
        assert result.flagged is True
        assert result.citation_alignment_score < 0.15

    def test_output_schema(self):
        inp = CitationAlignmentInput(
            question="What was Apple revenue?",
            answer=CLEAN_ANSWER,
            source=SOURCE,
        )
        result = citation_score(inp)
        assert 0.0 <= result.token_overlap <= 1.0
        assert 0.0 <= result.rouge1 <= 1.0
        assert 0.0 <= result.rouge2 <= 1.0
        assert 0.0 <= result.citation_alignment_score <= 1.0
        assert isinstance(result.flagged, bool)
        assert result.method == "citation_alignment"


# --- Span coverage tests ---

class TestSpanCoverage:

    def test_clean_pair_fully_covered(self):
        inp = SpanCoverageInput(
            question="What was Apple revenue?",
            answer=CLEAN_ANSWER,
            source=SOURCE,
        )
        result = span_score(inp)
        assert result.flagged is False
        assert result.coverage_score == 1.0

    def test_faithfulness_violation_flagged(self):
        inp = SpanCoverageInput(
            question="What was Apple revenue?",
            answer=FAITH_ANSWER,
            source=SOURCE,
        )
        result = span_score(inp)
        assert result.flagged is True
        assert "$450 billion" in result.uncovered_phrases or \
               any("450" in p for p in result.uncovered_phrases)

    def test_output_schema(self):
        inp = SpanCoverageInput(
            question="What was Apple revenue?",
            answer=CLEAN_ANSWER,
            source=SOURCE,
        )
        result = span_score(inp)
        assert 0.0 <= result.coverage_score <= 1.0
        assert isinstance(result.answer_phrases, list)
        assert isinstance(result.covered_phrases, list)
        assert isinstance(result.uncovered_phrases, list)
        assert result.method == "span_coverage"


# --- Confidence level tests ---

class TestConfidenceLevel:

    def test_pass_threshold(self):
        assert score_to_confidence(0.75) == ConfidenceLevel.PASS
        assert score_to_confidence(1.0) == ConfidenceLevel.PASS
        assert score_to_confidence(0.80) == ConfidenceLevel.PASS

    def test_review_threshold(self):
        assert score_to_confidence(0.40) == ConfidenceLevel.REVIEW
        assert score_to_confidence(0.60) == ConfidenceLevel.REVIEW
        assert score_to_confidence(0.74) == ConfidenceLevel.REVIEW

    def test_fail_threshold(self):
        assert score_to_confidence(0.0) == ConfidenceLevel.FAIL
        assert score_to_confidence(0.39) == ConfidenceLevel.FAIL
        assert score_to_confidence(0.20) == ConfidenceLevel.FAIL