"""
Finance RAG hallucination check example.

Simulates a realistic RAG pipeline scenario:
- A finance analyst asks a question
- A retrieval system returns relevant source chunks
- An LLM generates an answer
- veracitor checks whether the answer is supported
"""

from veracitor import check_answer, ConfidenceLevel

# Simulated retrieved source chunks from a 10-K filing
RETRIEVED_CHUNKS = [
    (
        "For fiscal year 2025, JPMorgan Chase reported net income of $58.5 billion, "
        "compared to $49.6 billion in fiscal year 2024. Return on equity was 17% "
        "for the full year 2025."
    ),
    (
        "Total assets as of December 31, 2025 were $4.0 trillion, up from $3.9 trillion "
        "at year-end 2024. The CET1 capital ratio was 15.7% under the standardized approach."
    ),
]

# Test cases: one supported, one hallucinated
test_cases = [
    {
        "label": "supported",
        "question": "What was JPMorgan's net income in fiscal 2025?",
        "answer": (
            "JPMorgan Chase reported net income of $58.5 billion in fiscal year 2025, "
            "up from $49.6 billion in the prior year."
        ),
    },
    {
        "label": "hallucinated",
        "question": "What was JPMorgan's net income in fiscal 2025?",
        "answer": (
            "JPMorgan Chase reported net income of $62 billion in fiscal year 2025, "
            "with return on equity improving to 19%."
        ),
    },
    {
        "label": "irrelevant",
        "question": "What was JPMorgan's net income in fiscal 2025?",
        "answer": (
            "JPMorgan Chase was founded in 1799 and is headquartered in New York City. "
            "It is the largest bank in the United States by assets."
        ),
    },
]

print("Finance RAG Hallucination Check")
print("=" * 60)

for tc in test_cases:
    result = check_answer(
        question=tc["question"],
        answer=tc["answer"],
        sources=RETRIEVED_CHUNKS,
    )

    verdict = result.confidence.value
    status_symbol = "+" if verdict == "PASS" else "x"

    print(f"\n[{status_symbol}] Label: {tc['label']}")
    print(f"    Confidence:    {verdict}")
    print(f"    Overall score: {result.overall_score}")
    print(f"    Flagged:       {result.flagged}")

    if result.flagged_claims:
        print(f"    Flagged claim: {result.flagged_claims[0][:80]}...")

    for method, r in result.method_breakdown.items():
        flag_str = f" <- FLAGGED: {r.explanation}" if r.flagged else ""
        print(f"    {method}: {r.score}{flag_str}")

    # Show what action to take based on confidence level
    if result.confidence == ConfidenceLevel.PASS:
        print("    Action: safe to return to user")
    elif result.confidence == ConfidenceLevel.REVIEW:
        print("    Action: flag for human review before returning")
    else:
        print("    Action: block response, retrieve new context or escalate")