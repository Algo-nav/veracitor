import re
from typing import Optional
from pydantic import BaseModel

# Lazy imports - model is loaded once on first use, not at import time.
# This keeps the library fast to import when NLI isn't needed.
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("  [nli] Loading RoBERTa MNLI model (first run only)...")
        tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
        model.eval()

        class Pipeline:
            pass
        p = Pipeline()
        p.tokenizer = tokenizer
        p.model = model
        _pipeline = p
        print("  [nli] Model loaded.")
    return _pipeline


# --- Pydantic schemas ---

class NLIEntailmentInput(BaseModel):
    question: str
    answer: str
    source: str
    label: Optional[str] = None


class ClaimResult(BaseModel):
    claim: str
    entailment: float
    neutral: float
    contradiction: float
    verdict: str    # "entailed", "neutral", "contradiction"


class NLIEntailmentOutput(BaseModel):
    claims: list[ClaimResult]
    entailment_score: float       # mean entailment across claims
    contradiction_score: float    # max contradiction across claims (worst case)
    nli_score: float              # combined score: high = well supported
    flagged: bool
    threshold: float
    method: str = "nli_entailment"
    error: Optional[str] = None


# --- Claim extraction ---

def extract_claims(text: str) -> list[str]:
    """
    Split answer into individual claims at sentence boundaries.
    Each claim is checked independently against the source.
    Filters out very short fragments (under 5 words) that aren't checkable.
    """
    # Split on period, exclamation, question mark followed by space or end
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = [s.strip() for s in sentences if len(s.split()) >= 5]
    return claims if claims else [text.strip()]


# --- Core scoring ---

def score_claim(claim: str, source: str, pipe) -> ClaimResult:
    """
    Run NLI on a single claim against the source.
    Premise = source, Hypothesis = claim.
    DeBERTa NLI label order: contradiction=0, neutral=1, entailment=2
    """
    import torch

    inputs = pipe.tokenizer(
        source,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = pipe.model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()

    contradiction = probs[0]
    neutral = probs[1]
    entailment = probs[2]

    if entailment >= 0.5:
        verdict = "entailed"
    elif contradiction >= 0.4:
        verdict = "contradiction"
    else:
        verdict = "neutral"

    return ClaimResult(
        claim=claim,
        entailment=round(entailment, 4),
        neutral=round(neutral, 4),
        contradiction=round(contradiction, 4),
        verdict=verdict,
    )


def compute_nli_entailment(
    answer: str,
    source: str,
    threshold: float = 0.40,
) -> NLIEntailmentOutput:
    """
    Score an answer against a source using NLI entailment.

    Scoring logic:
    - Extract claims from answer
    - Score each claim against source
    - entailment_score = mean entailment probability across claims
    - contradiction_score = max contradiction probability (worst-case claim)
    - nli_score = entailment_score - contradiction_score (bounded signal)
    - flagged if nli_score < threshold

    Threshold 0.40 calibrated to balance precision and recall on finance text.
    """
    try:
        pipe = get_pipeline()
        claims = extract_claims(answer)
        claim_results = [score_claim(claim, source, pipe) for claim in claims]

        entailment_scores = [c.entailment for c in claim_results]
        contradiction_scores = [c.contradiction for c in claim_results]

        mean_entailment = sum(entailment_scores) / len(entailment_scores)
        max_contradiction = max(contradiction_scores)

        # Combined score: reward entailment, penalize contradiction
        nli_score = mean_entailment - max_contradiction

        return NLIEntailmentOutput(
            claims=claim_results,
            entailment_score=round(mean_entailment, 4),
            contradiction_score=round(max_contradiction, 4),
            nli_score=round(nli_score, 4),
            flagged=nli_score < threshold,
            threshold=threshold,
        )

    except Exception as e:
        return NLIEntailmentOutput(
            claims=[],
            entailment_score=0.0,
            contradiction_score=0.0,
            nli_score=0.0,
            flagged=True,
            threshold=threshold,
            error=str(e),
        )


def score_pair(input_data: NLIEntailmentInput) -> NLIEntailmentOutput:
    """Public entry point. Scores a single Q&A pair."""
    return compute_nli_entailment(
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
        inp = NLIEntailmentInput(**tc)
        result = score_pair(inp)

        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            for claim in result.claims:
                print(f"  Claim: {claim.claim[:80]}...")
                print(f"    Entailment: {claim.entailment}  "
                      f"Neutral: {claim.neutral}  "
                      f"Contradiction: {claim.contradiction}  "
                      f"Verdict: {claim.verdict}")
            print(f"  NLI score:          {result.nli_score}")
            print(f"  Flagged:            {result.flagged}")