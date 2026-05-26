from typing import Optional
from veracitor.models import CheckResult, MethodResult, ConfidenceLevel, score_to_confidence

# --- Method aliases ---
# Maps user-facing string aliases to internal method names.
# Reduces friction: "nli" is easier to remember than "nli_entailment".

METHOD_ALIASES = {
    "citation": "citation_alignment",
    "citation_alignment": "citation_alignment",
    "nli": "nli_entailment",
    "nli_entailment": "nli_entailment",
    "judge": "llm_judge",
    "llm_judge": "llm_judge",
    "span": "span_coverage",
    "span_coverage": "span_coverage",
    "consistency": "self_consistency",
    "self_consistency": "self_consistency",
}

# Default methods when user doesn't specify.
# Citation alignment + NLI: free, fast, covers most hallucination types.
# LLM judge is opt-in because it costs API credits.
DEFAULT_METHODS = ["citation_alignment", "nli_entailment"]

# Method weights for overall_score aggregation.
# NLI weighted highest: strongest single method in benchmark (F1 0.83).
# LLM judge would be highest but is optional, so excluded from default weights.
METHOD_WEIGHTS = {
    "citation_alignment": 0.25,
    "nli_entailment": 0.40,
    "span_coverage": 0.20,
    "self_consistency": 0.15,
    "llm_judge": 0.50,   # used only when judge is in the method list
}


def _resolve_methods(methods: list[str]) -> list[str]:
    """Resolve aliases and deduplicate method names."""
    resolved = []
    seen = set()
    for m in methods:
        canonical = METHOD_ALIASES.get(m.lower())
        if canonical is None:
            raise ValueError(
                f"Unknown method: '{m}'. "
                f"Valid options: {list(METHOD_ALIASES.keys())}"
            )
        if canonical not in seen:
            resolved.append(canonical)
            seen.add(canonical)
    return resolved


def _run_citation_alignment(question: str, answer: str, source: str) -> MethodResult:
    from veracitor.methods.citation_alignment import CitationAlignmentInput, score_pair
    inp = CitationAlignmentInput(question=question, answer=answer, source=source)
    out = score_pair(inp)
    explanation = None
    if out.flagged:
        explanation = (
            f"Low lexical overlap with source "
            f"(score {out.citation_alignment_score:.2f}, "
            f"ROUGE-1 {out.rouge1:.2f})"
        )
    return MethodResult(
        method="citation_alignment",
        score=out.citation_alignment_score,
        flagged=out.flagged,
        explanation=explanation,
        details={
            "token_overlap": out.token_overlap,
            "rouge1": out.rouge1,
            "rouge2": out.rouge2,
        },
    )


def _run_nli_entailment(question: str, answer: str, source: str) -> MethodResult:
    from veracitor.methods.nli_entailment import NLIEntailmentInput, score_pair
    inp = NLIEntailmentInput(question=question, answer=answer, source=source)
    out = score_pair(inp)

    # Convert nli_score to 0-1 range for consistent aggregation
    # nli_score = entailment - contradiction, range roughly -1 to 1
    # Normalize: (score + 1) / 2
    normalized = (out.nli_score + 1) / 2

    explanation = None
    flagged_claims = []
    if out.flagged:
        contradicted = [c for c in out.claims if c.verdict == "contradiction"]
        if contradicted:
            flagged_claims = [c.claim for c in contradicted]
            explanation = (
                f"{len(contradicted)} claim(s) contradict the source "
                f"(contradiction score {out.contradiction_score:.2f})"
            )
        else:
            explanation = (
                f"Low entailment score ({out.entailment_score:.2f}). "
                f"Claims are not well-supported by source."
            )

    return MethodResult(
        method="nli_entailment",
        score=round(normalized, 4),
        flagged=out.flagged,
        explanation=explanation,
        details={
            "entailment_score": out.entailment_score,
            "contradiction_score": out.contradiction_score,
            "nli_score": out.nli_score,
            "num_claims": len(out.claims),
        },
    ), flagged_claims


def _run_llm_judge(question: str, answer: str, source: str) -> MethodResult:
    from veracitor.methods.llm_judge import LLMJudgeInput, score_pair
    inp = LLMJudgeInput(question=question, answer=answer, source=source)
    out = score_pair(inp)
    explanation = None
    if out.flagged:
        parts = []
        if out.faithfulness.score < 0.5:
            parts.append(f"faithfulness: {out.faithfulness.reasoning}")
        if out.factuality.score < 0.5:
            parts.append(f"factuality: {out.factuality.reasoning}")
        if out.relevance.score < 0.5:
            parts.append(f"relevance: {out.relevance.reasoning}")
        explanation = " | ".join(parts) if parts else "Low judge score"
    return MethodResult(
        method="llm_judge",
        score=out.judge_score,
        flagged=out.flagged,
        explanation=explanation,
        details={
            "faithfulness": out.faithfulness.score,
            "factuality": out.factuality.score,
            "relevance": out.relevance.score,
            "faithfulness_reasoning": out.faithfulness.reasoning,
            "factuality_reasoning": out.factuality.reasoning,
            "relevance_reasoning": out.relevance.reasoning,
        },
    )


def _run_span_coverage(question: str, answer: str, source: str) -> MethodResult:
    from veracitor.methods.span_coverage import SpanCoverageInput, score_pair
    inp = SpanCoverageInput(question=question, answer=answer, source=source)
    out = score_pair(inp)
    explanation = None
    flagged_claims = []
    if out.flagged:
        flagged_claims = out.uncovered_phrases
        explanation = (
            f"{len(out.uncovered_phrases)} phrase(s) not found in source: "
            f"{', '.join(out.uncovered_phrases[:3])}"
            f"{'...' if len(out.uncovered_phrases) > 3 else ''}"
        )
    return MethodResult(
        method="span_coverage",
        score=out.coverage_score,
        flagged=out.flagged,
        explanation=explanation,
        details={
            "covered_phrases": out.covered_phrases,
            "uncovered_phrases": out.uncovered_phrases,
        },
    ), flagged_claims


def _run_self_consistency(question: str, answer: str, source: str) -> MethodResult:
    from veracitor.methods.self_consistency import SelfConsistencyInput, score_pair
    inp = SelfConsistencyInput(question=question, answer=answer, source=source)
    out = score_pair(inp)
    explanation = None
    if out.flagged:
        explanation = (
            f"Low consistency with regenerated answers "
            f"(score {out.consistency_score:.2f}). "
            f"Answer may contain unsupported claims."
        )
    return MethodResult(
        method="self_consistency",
        score=out.consistency_score,
        flagged=out.flagged,
        explanation=explanation,
        details={"rouge1_scores": out.rouge1_scores},
    )


def _compute_overall_score(
    method_results: dict[str, MethodResult],
    methods_used: list[str],
) -> float:
    """
    Weighted mean of method scores.
    If LLM judge is present, it dominates (weight 0.50).
    Otherwise NLI dominates (weight 0.40).
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for method in methods_used:
        result = method_results.get(method)
        if result is None:
            continue
        weight = METHOD_WEIGHTS.get(method, 0.20)
        weighted_sum += result.score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return round(weighted_sum / total_weight, 4)


def check_answer(
    question: str,
    answer: str,
    sources: list[str],
    methods: Optional[list[str]] = None,
) -> CheckResult:
    """
    Check whether a RAG-generated answer is supported by the provided sources.

    Args:
        question: The question that was asked.
        answer: The generated answer to check.
        sources: List of source text strings retrieved for this question.
                 Multiple sources are concatenated.
        methods: List of detection methods to run. Defaults to
                 ["citation_alignment", "nli_entailment"].
                 Valid aliases: "citation", "nli", "judge", "span", "consistency".

    Returns:
        CheckResult with overall verdict, per-method breakdown, and flagged claims.
    """
    # Resolve methods
    if methods is None:
        methods_to_run = DEFAULT_METHODS
    else:
        methods_to_run = _resolve_methods(methods)

    # Concatenate sources into a single string
    source = " ".join(sources)

    method_results: dict[str, MethodResult] = {}
    all_flagged_claims: list[str] = []

    for method in methods_to_run:
        try:
            if method == "citation_alignment":
                result = _run_citation_alignment(question, answer, source)
                method_results[method] = result

            elif method == "nli_entailment":
                result, flagged_claims = _run_nli_entailment(question, answer, source)
                method_results[method] = result
                all_flagged_claims.extend(flagged_claims)

            elif method == "llm_judge":
                result = _run_llm_judge(question, answer, source)
                method_results[method] = result

            elif method == "span_coverage":
                result, flagged_claims = _run_span_coverage(question, answer, source)
                method_results[method] = result
                all_flagged_claims.extend(flagged_claims)

            elif method == "self_consistency":
                result = _run_self_consistency(question, answer, source)
                method_results[method] = result

        except Exception as e:
            method_results[method] = MethodResult(
                method=method,
                score=0.0,
                flagged=True,
                explanation=f"Method failed: {str(e)}",
            )

    # Aggregate
    flagged = any(r.flagged for r in method_results.values())
    overall_score = _compute_overall_score(method_results, methods_to_run)
    confidence = score_to_confidence(overall_score)

    # Deduplicate flagged claims
    seen = set()
    unique_claims = []
    for claim in all_flagged_claims:
        if claim.lower() not in seen:
            seen.add(claim.lower())
            unique_claims.append(claim)

    return CheckResult(
        question=question,
        answer=answer,
        flagged=flagged,
        overall_score=overall_score,
        confidence=confidence,
        methods_used=methods_to_run,
        method_breakdown=method_results,
        flagged_claims=unique_claims,
    )