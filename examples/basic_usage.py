"""
Basic usage examples for veracitor.

Shows how to check a RAG-generated answer for hallucinations
using the default two-method pipeline (citation alignment + NLI).
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from veracitor import check_answer

SOURCE = (
    "Apple Inc. reported total net sales of $391 billion for fiscal year 2025, "
    "representing a 4% increase compared to the prior year. iPhone revenue "
    "accounted for $201 billion, or approximately 51% of total net sales."
)

print("=" * 60)
print("Example 1: Clean answer (should PASS)")
print("=" * 60)

result = check_answer(
    question="What was Apple's total revenue in fiscal 2025?",
    answer="Apple reported total net sales of $391 billion in fiscal year 2025.",
    sources=[SOURCE],
)

print(f"Flagged:       {result.flagged}")
print(f"Confidence:    {result.confidence.value}")
print(f"Overall score: {result.overall_score}")
print(f"Methods used:  {result.methods_used}")
for method, r in result.method_breakdown.items():
    print(f"  {method}: score={r.score}, flagged={r.flagged}")

print()
print("=" * 60)
print("Example 2: Faithfulness violation (should FAIL)")
print("=" * 60)

result = check_answer(
    question="What was Apple's total revenue in fiscal 2025?",
    answer=(
        "Apple reported total net sales of $450 billion in fiscal year 2025, "
        "driven by strong iPhone sales in emerging markets."
    ),
    sources=[SOURCE],
)

print(f"Flagged:        {result.flagged}")
print(f"Confidence:     {result.confidence.value}")
print(f"Overall score:  {result.overall_score}")
if result.flagged_claims:
    print(f"Flagged claims: {result.flagged_claims[0][:80]}...")
for method, r in result.method_breakdown.items():
    status = f"flagged ({r.explanation})" if r.flagged else "ok"
    print(f"  {method}: {status}")

print()
print("=" * 60)
print("Example 3: Using the LLM judge for higher accuracy")
print("=" * 60)

result = check_answer(
    question="What was Apple's total revenue in fiscal 2025?",
    answer=(
        "Apple reported total net sales of $450 billion in fiscal year 2025, "
        "driven by strong iPhone sales in emerging markets."
    ),
    sources=[SOURCE],
    methods=["citation", "nli", "judge"],
)

print(f"Flagged:       {result.flagged}")
print(f"Confidence:    {result.confidence.value}")
print(f"Overall score: {result.overall_score}")
for method, r in result.method_breakdown.items():
    print(f"  {method}: score={r.score}, flagged={r.flagged}")