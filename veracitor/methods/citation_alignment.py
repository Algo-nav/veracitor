import re
import math
from typing import Optional
from pydantic import BaseModel


# --- Pydantic schemas ---

class CitationAlignmentInput(BaseModel):
    question: str
    answer: str
    source: str          # the retrieved source text (evidence span or full passage)
    label: Optional[str] = None   # ground truth label if available, for benchmark use


class CitationAlignmentOutput(BaseModel):
    token_overlap: float         # fraction of answer tokens found in source
    rouge1: float                # unigram F1 between answer and source
    rouge2: float                # bigram F1 between answer and source
    citation_alignment_score: float   # weighted combination of the three
    flagged: bool                # True if score is below the threshold
    threshold: float             # threshold used for flagging
    method: str = "citation_alignment"
    error: Optional[str] = None


# --- Tokenization ---

def tokenize(text: str) -> list[str]:
    """
    Lowercase, remove punctuation, split into tokens.
    Simple but consistent across all inputs.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def get_bigrams(tokens: list[str]) -> list[tuple]:
    """Return all consecutive token pairs."""
    return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]


# --- Scoring functions ---

def token_overlap(answer_tokens: list[str], source_tokens: list[str]) -> float:
    """
    Fraction of answer tokens that appear in the source.
    Precision-oriented: penalizes answer claims not grounded in source.
    """
    if not answer_tokens:
        return 0.0
    source_set = set(source_tokens)
    matched = sum(1 for t in answer_tokens if t in source_set)
    return matched / len(answer_tokens)


def rouge_n(answer_tokens: list[str], source_tokens: list[str], n: int) -> float:
    """
    ROUGE-N F1 score between answer and source.
    F1 balances precision (answer grounded in source) and recall (source covers answer).
    """
    if n == 1:
        answer_ngrams = answer_tokens
        source_ngrams = source_tokens
    elif n == 2:
        answer_ngrams = get_bigrams(answer_tokens)
        source_ngrams = get_bigrams(source_tokens)
    else:
        raise ValueError(f"n must be 1 or 2, got {n}")

    if not answer_ngrams or not source_ngrams:
        return 0.0

    answer_counts: dict = {}
    for ng in answer_ngrams:
        answer_counts[ng] = answer_counts.get(ng, 0) + 1

    source_counts: dict = {}
    for ng in source_ngrams:
        source_counts[ng] = source_counts.get(ng, 0) + 1

    # Count overlapping n-grams (clipped)
    overlap = sum(
        min(count, source_counts.get(ng, 0))
        for ng, count in answer_counts.items()
    )

    precision = overlap / len(answer_ngrams)
    recall = overlap / len(source_ngrams)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_citation_alignment(
    answer: str,
    source: str,
    threshold: float = 0.35,
) -> CitationAlignmentOutput:
    """
    Compute citation alignment score between an answer and its source.

    Weights:
    - token_overlap: 0.3 (directional signal, less robust alone)
    - rouge1: 0.4 (primary signal, unigram coverage)
    - rouge2: 0.3 (phrase-level signal, harder to game)

    Threshold: 0.35. Below this, the answer is flagged as potentially unsupported.
    Calibrated conservatively - we prefer false positives over missed hallucinations
    in a finance context.
    """
    try:
        answer_tokens = tokenize(answer)
        source_tokens = tokenize(source)

        to = token_overlap(answer_tokens, source_tokens)
        r1 = rouge_n(answer_tokens, source_tokens, n=1)
        r2 = rouge_n(answer_tokens, source_tokens, n=2)

        # Weighted combination
        score = 0.3 * to + 0.4 * r1 + 0.3 * r2

        return CitationAlignmentOutput(
            token_overlap=round(to, 4),
            rouge1=round(r1, 4),
            rouge2=round(r2, 4),
            citation_alignment_score=round(score, 4),
            flagged=score < threshold,
            threshold=threshold,
        )

    except Exception as e:
        return CitationAlignmentOutput(
            token_overlap=0.0,
            rouge1=0.0,
            rouge2=0.0,
            citation_alignment_score=0.0,
            flagged=True,
            threshold=threshold,
            error=str(e),
        )


def score_pair(input_data: CitationAlignmentInput) -> CitationAlignmentOutput:
    """Public entry point. Scores a single Q&A pair."""
    return compute_citation_alignment(
        answer=input_data.answer,
        source=input_data.source,
    )


# --- Test block ---

if __name__ == "__main__":
    # Test 1: clean pair - answer closely matches source
    clean_pair = CitationAlignmentInput(
        question="What was Apple's revenue in fiscal 2025?",
        answer="Apple reported total net sales of $391 billion in fiscal year 2025.",
        source="Apple Inc. reported total net sales of $391 billion for fiscal year 2025, "
               "representing a 4% increase compared to the prior year.",
        label="clean"
    )

    # Test 2: faithfulness violation - answer contains unsupported claim
    faith_pair = CitationAlignmentInput(
        question="What was Apple's revenue in fiscal 2025?",
        answer="Apple reported total net sales of $450 billion in fiscal year 2025, "
               "driven by strong iPhone sales in emerging markets.",
        source="Apple Inc. reported total net sales of $391 billion for fiscal year 2025, "
               "representing a 4% increase compared to the prior year.",
        label="faithfulness"
    )

    # Test 3: relevance violation - answer doesn't address the question
    rel_pair = CitationAlignmentInput(
        question="What was Apple's revenue in fiscal 2025?",
        answer="Apple was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne "
               "in Cupertino, California.",
        source="Apple Inc. reported total net sales of $391 billion for fiscal year 2025, "
               "representing a 4% increase compared to the prior year.",
        label="relevance"
    )

    for pair in [clean_pair, faith_pair, rel_pair]:
        result = score_pair(pair)
        print(f"\nLabel: {pair.label}")
        print(f"  Token overlap:  {result.token_overlap}")
        print(f"  ROUGE-1:        {result.rouge1}")
        print(f"  ROUGE-2:        {result.rouge2}")
        print(f"  Alignment score:{result.citation_alignment_score}")
        print(f"  Flagged:        {result.flagged}")