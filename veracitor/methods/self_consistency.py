import os
import re
import time
import json
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# --- Pydantic schemas ---

class SelfConsistencyInput(BaseModel):
    question: str
    answer: str
    source: str
    label: Optional[str] = None


class SelfConsistencyOutput(BaseModel):
    samples: list[str]              # the N generated answers
    rouge1_scores: list[float]      # pairwise ROUGE-1 between samples
    consistency_score: float        # mean pairwise ROUGE-1
    flagged: bool
    threshold: float
    method: str = "self_consistency"
    error: Optional[str] = None


# --- Tokenization (reuse pattern from citation alignment) ---

def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def rouge1_f1(text_a: str, text_b: str) -> float:
    """ROUGE-1 F1 between two texts."""
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    counts_a: dict = {}
    for t in tokens_a:
        counts_a[t] = counts_a.get(t, 0) + 1

    counts_b: dict = {}
    for t in tokens_b:
        counts_b[t] = counts_b.get(t, 0) + 1

    overlap = sum(min(c, counts_b.get(t, 0)) for t, c in counts_a.items())
    precision = overlap / len(tokens_a)
    recall = overlap / len(tokens_b)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --- Sample generation ---

SAMPLE_PROMPT = """You are a finance research assistant. Answer the question based only 
on the provided source text. Be concise and specific."""

def generate_sample(question: str, source: str) -> str:
    """Generate one answer sample for the given question and source."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=SAMPLE_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Source: {source[:2000]}\n\nQuestion: {question}"
            }
        ]
    )
    return response.content[0].text.strip()


# --- Core scoring ---

def compute_self_consistency(
    question: str,
    source: str,
    n_samples: int = 3,
    threshold: float = 0.40,
) -> tuple[list[str], float]:
    """
    Generate n_samples independent answers and measure pairwise consistency.
    Returns (samples, consistency_score).
    """
    samples = []
    for i in range(n_samples):
        sample = generate_sample(question, source)
        samples.append(sample)
        if i < n_samples - 1:
            time.sleep(0.5)

    # Compute all pairwise ROUGE-1 scores
    pairwise_scores = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            score = rouge1_f1(samples[i], samples[j])
            pairwise_scores.append(score)

    consistency_score = sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else 0.0
    return samples, consistency_score


def answer_consistency_score(provided_answer: str, samples: list[str]) -> float:
    """
    Measure how consistent the provided answer is with model-generated samples.
    For each sample, compute ROUGE-1 against the provided answer.
    Returns mean score across all samples.
    Low score = provided answer diverges from what the model generates = likely hallucination.
    """
    scores = []
    for sample in samples:
        scores.append(rouge1_f1(provided_answer, sample))
    return sum(scores) / len(scores) if scores else 0.0


def score_pair(input_data: SelfConsistencyInput) -> SelfConsistencyOutput:
    """
    Score a Q&A pair using answer-grounded self-consistency (SelfCheckGPT pattern).

    Generate N independent answers from the source, then measure how well
    the PROVIDED answer agrees with the regenerations. If the provided answer
    diverges significantly from what the model generates from the source,
    it likely contains claims not grounded in the source.

    threshold = 0.48: below this, the provided answer is flagged as inconsistent
    with model regenerations.
    """
    threshold = 0.48
    try:
        samples, _ = compute_self_consistency(
            question=input_data.question,
            source=input_data.source,
            n_samples=3,
            threshold=threshold,
        )

        # Score the PROVIDED answer against regenerations
        consistency_score = answer_consistency_score(input_data.answer, samples)

        # Pairwise scores for transparency
        pairwise = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                pairwise.append(rouge1_f1(samples[i], samples[j]))

        return SelfConsistencyOutput(
            samples=samples,
            rouge1_scores=[round(s, 4) for s in pairwise],
            consistency_score=round(consistency_score, 4),
            flagged=consistency_score < threshold,
            threshold=threshold,
        )

    except Exception as e:
        return SelfConsistencyOutput(
            samples=[],
            rouge1_scores=[],
            consistency_score=0.0,
            flagged=True,
            threshold=threshold,
            error=str(e),
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
            "answer": "Apple reported total net sales of $450 billion in fiscal year 2025.",
            "source": "Apple Inc. reported total net sales of $391 billion for fiscal year 2025, "
                      "representing a 4% increase compared to the prior year.",
        },
    ]

    for tc in test_cases:
        print(f"\nLabel: {tc['label']}")
        inp = SelfConsistencyInput(**tc)
        result = score_pair(inp)

        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            for i, sample in enumerate(result.samples):
                print(f"  Sample {i+1}: {sample[:100]}...")
            print(f"  Pairwise ROUGE-1: {result.rouge1_scores}")
            print(f"  Consistency score: {result.consistency_score}")
            print(f"  Flagged: {result.flagged}")