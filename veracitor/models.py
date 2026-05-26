from enum import Enum
from typing import Optional
from pydantic import BaseModel


class ConfidenceLevel(str, Enum):
    """
    Human-readable confidence level for hallucination detection.
    More actionable than a binary flag for production use.
    """
    PASS = "PASS"       # score >= 0.75: answer appears well-supported
    REVIEW = "REVIEW"   # score 0.40-0.75: answer has suspicious elements
    FAIL = "FAIL"       # score < 0.40: answer likely contains hallucinations


def score_to_confidence(score: float) -> ConfidenceLevel:
    """Map a numeric score to a confidence level."""
    if score >= 0.75:
        return ConfidenceLevel.PASS
    elif score >= 0.40:
        return ConfidenceLevel.REVIEW
    else:
        return ConfidenceLevel.FAIL


class MethodResult(BaseModel):
    """Result from a single detection method."""
    method: str
    score: float                        # 0-1, higher = more supported
    flagged: bool                       # True if method thinks answer is hallucinated
    explanation: Optional[str] = None   # human-readable reason for flag
    details: Optional[dict] = None      # raw method-specific output


class CheckResult(BaseModel):
    """
    The public result object returned by check_answer().
    Designed to be actionable without requiring knowledge of internals.
    """
    question: str
    answer: str

    # Top-level verdict
    flagged: bool                           # True if ANY method flagged
    overall_score: float                    # weighted mean across methods, 0-1
    confidence: ConfidenceLevel             # PASS / REVIEW / FAIL

    # Method breakdown
    methods_used: list[str]
    method_breakdown: dict[str, MethodResult]

    # Specific flagged content
    flagged_claims: list[str]               # claims identified as unsupported

    # Metadata
    error: Optional[str] = None