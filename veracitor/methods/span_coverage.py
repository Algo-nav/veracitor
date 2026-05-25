import re
from typing import Optional
from pydantic import BaseModel


# --- Pydantic schemas ---

class SpanCoverageInput(BaseModel):
    question: str
    answer: str
    source: str
    label: Optional[str] = None


class SpanCoverageOutput(BaseModel):
    answer_phrases: list[str]       # extracted noun phrases and entities from answer
    covered_phrases: list[str]      # phrases found in source
    uncovered_phrases: list[str]    # phrases not found in source
    coverage_score: float           # fraction of answer phrases covered by source
    flagged: bool
    threshold: float
    method: str = "span_coverage"
    error: Optional[str] = None


# --- Phrase extraction ---

def extract_key_phrases(text: str) -> list[str]:
    """
    Extract key phrases from text for coverage checking.
    Strategy: extract numbers, percentages, named capitalized terms,
    and quoted values. These are the claim-bearing tokens most likely
    to be hallucinated.
    """
    phrases = []

    # Numbers and percentages (e.g. "$391 billion", "4%", "2025")
    number_pattern = re.findall(
        r'\$[\d,\.]+\s*(?:billion|million|trillion|thousand)?|\d+\.?\d*\s*%|\b\d{4}\b',
        text,
        re.IGNORECASE
    )
    phrases.extend(number_pattern)

    # Capitalized proper nouns (2+ consecutive capitalized words)
    proper_noun_pattern = re.findall(
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',
        text
    )
    phrases.extend(proper_noun_pattern)

    # Single capitalized words that are likely entities (not sentence starters)
    # Look for capitalized words after lowercase words
    entity_pattern = re.findall(
        r'(?<=[a-z]\s)[A-Z][a-zA-Z]+',
        text
    )
    phrases.extend(entity_pattern)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in phrases:
        normalized = p.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(p.strip())

    return unique


def phrase_in_source(phrase: str, source: str) -> bool:
    """
    Check if a phrase appears in the source text.
    Case-insensitive, allows minor whitespace variation.
    """
    phrase_clean = re.sub(r'\s+', ' ', phrase.strip().lower())
    source_clean = re.sub(r'\s+', ' ', source.lower())
    return phrase_clean in source_clean


# --- Core scoring ---

def compute_span_coverage(
    answer: str,
    source: str,
    threshold: float = 0.60,
) -> SpanCoverageOutput:
    """
    Measure what fraction of key answer phrases are grounded in the source.

    Extracts numbers, percentages, named entities, and proper nouns from
    the answer. Checks each against the source text. Low coverage means
    the answer is making claims not present in the source.

    Threshold 0.60: below this, the answer is flagged as insufficiently
    grounded in the source.
    """
    try:
        phrases = extract_key_phrases(answer)

        if not phrases:
            # No extractable phrases: conservative, do not flag
            return SpanCoverageOutput(
                answer_phrases=[],
                covered_phrases=[],
                uncovered_phrases=[],
                coverage_score=1.0,
                flagged=False,
                threshold=threshold,
            )

        covered = [p for p in phrases if phrase_in_source(p, source)]
        uncovered = [p for p in phrases if not phrase_in_source(p, source)]
        coverage_score = len(covered) / len(phrases)

        return SpanCoverageOutput(
            answer_phrases=phrases,
            covered_phrases=covered,
            uncovered_phrases=uncovered,
            coverage_score=round(coverage_score, 4),
            flagged=coverage_score < threshold,
            threshold=threshold,
        )

    except Exception as e:
        return SpanCoverageOutput(
            answer_phrases=[],
            covered_phrases=[],
            uncovered_phrases=[],
            coverage_score=0.0,
            flagged=True,
            threshold=threshold,
            error=str(e),
        )


def score_pair(input_data: SpanCoverageInput) -> SpanCoverageOutput:
    """Public entry point. Scores a single Q&A pair."""
    return compute_span_coverage(
        answer=input_data.answer,
        source=input_data.source,
    )


# --- Test block ---

if __name__ == "__main__":
    test_cases = [
        {
            "label": "clean",
            "question": "What was Apple's revenue in fiscal 2025?",
            "answer": "Apple reported total net sales of $391 billion in fiscal year 2025.",
            "source": "Apple Inc. reported total net sales of $391 billion for fiscal year 2025, "
                      "representing a 4% increase compared to the prior year.",
        },
        {
            "label": "faithfulness",
            "question": "What was Apple's revenue in fiscal 2025?",
            "answer": "Apple reported total net sales of $450 billion in fiscal year 2025, "
                      "driven by strong iPhone sales in emerging markets.",
            "source": "Apple Inc. reported total net sales of $391 billion for fiscal year 2025, "
                      "representing a 4% increase compared to the prior year.",
        },
        {
            "label": "relevance",
            "question": "What was Apple's revenue in fiscal 2025?",
            "answer": "Apple was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne "
                      "in Cupertino, California.",
            "source": "Apple Inc. reported total net sales of $391 billion for fiscal year 2025, "
                      "representing a 4% increase compared to the prior year.",
        },
    ]

    for tc in test_cases:
        print(f"\nLabel: {tc['label']}")
        inp = SpanCoverageInput(**tc)
        result = score_pair(inp)

        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(f"  Phrases extracted: {result.answer_phrases}")
            print(f"  Covered:   {result.covered_phrases}")
            print(f"  Uncovered: {result.uncovered_phrases}")
            print(f"  Coverage score: {result.coverage_score}")
            print(f"  Flagged: {result.flagged}")