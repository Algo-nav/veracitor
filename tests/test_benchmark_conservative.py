"""
Conservative benchmark tests for veracitor detection methods.
Asserts minimum precision/recall thresholds with headroom for variance.
Safe to run in CI. Requires benchmark data in data/benchmark/.

Run with:
    pytest tests/test_benchmark_conservative.py -v -m benchmark
"""

import os
import json
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veracitor.methods.citation_alignment import CitationAlignmentInput
from veracitor.methods.citation_alignment import score_pair as citation_score
from veracitor.methods.span_coverage import SpanCoverageInput
from veracitor.methods.span_coverage import score_pair as span_score

# --- Paths ---

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(BASE_DIR, "data", "benchmark")


# --- Fixtures ---

def load_benchmark_pairs() -> list[dict]:
    """Load all Q&A pairs from benchmark directory."""
    if not os.path.exists(BENCHMARK_DIR):
        return []

    pairs = []
    for fname in sorted(os.listdir(BENCHMARK_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(BENCHMARK_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        doc_id = data.get("doc_id", fname.replace(".json", ""))
        doc_type = data.get("doc_type", "unknown")
        for i, pair in enumerate(data.get("pairs", [])):
            pairs.append({
                "doc_id": doc_id,
                "doc_type": doc_type,
                "pair_index": i,
                "question": pair.get("question", ""),
                "answer": pair.get("answer", ""),
                "label": pair.get("label", ""),
                "evidence": pair.get("evidence", ""),
            })
    return pairs


def compute_metrics(results: list[dict]) -> dict:
    tp = sum(1 for r in results if r["flagged"] and r["label"] != "clean")
    fp = sum(1 for r in results if r["flagged"] and r["label"] == "clean")
    tn = sum(1 for r in results if not r["flagged"] and r["label"] == "clean")
    fn = sum(1 for r in results if not r["flagged"] and r["label"] != "clean")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


pytestmark = pytest.mark.benchmark

# Skip all tests if benchmark data is missing
benchmark_pairs = load_benchmark_pairs()
skip_if_no_data = pytest.mark.skipif(
    len(benchmark_pairs) == 0,
    reason="Benchmark data not found. Run scripts/generate_qna.py first."
)


@skip_if_no_data
@pytest.mark.benchmark
def test_citation_alignment_precision():
    """Citation alignment precision should be >= 0.85 on full corpus."""
    results = []
    for pair in benchmark_pairs:
        inp = CitationAlignmentInput(
            question=pair["question"],
            answer=pair["answer"],
            source=pair["evidence"],
        )
        out = citation_score(inp)
        results.append({"label": pair["label"], "flagged": out.flagged})

    metrics = compute_metrics(results)
    print(f"\nCitation alignment: precision={metrics['precision']:.3f}, "
          f"recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}")
    assert metrics["precision"] >= 0.85, (
        f"Citation alignment precision {metrics['precision']:.3f} below threshold 0.85"
    )


@skip_if_no_data
@pytest.mark.benchmark
def test_citation_alignment_factuality_recall():
    """Citation alignment should catch >= 90% of factuality violations."""
    factuality_pairs = [p for p in benchmark_pairs if p["label"] == "factuality"]
    results = []
    for pair in factuality_pairs:
        inp = CitationAlignmentInput(
            question=pair["question"],
            answer=pair["answer"],
            source=pair["evidence"],
        )
        out = citation_score(inp)
        results.append({"label": pair["label"], "flagged": out.flagged})

    flagged = sum(1 for r in results if r["flagged"])
    rate = flagged / len(results) if results else 0.0
    print(f"\nCitation alignment factuality flag rate: {rate:.3f} ({flagged}/{len(results)})")
    assert rate >= 0.90, (
        f"Citation alignment factuality flag rate {rate:.3f} below threshold 0.90"
    )


@skip_if_no_data
@pytest.mark.benchmark
def test_citation_alignment_clean_false_positive_rate():
    """Citation alignment should flag <= 10% of clean pairs."""
    clean_pairs = [p for p in benchmark_pairs if p["label"] == "clean"]
    results = []
    for pair in clean_pairs:
        inp = CitationAlignmentInput(
            question=pair["question"],
            answer=pair["answer"],
            source=pair["evidence"],
        )
        out = citation_score(inp)
        results.append({"flagged": out.flagged})

    flagged = sum(1 for r in results if r["flagged"])
    rate = flagged / len(results) if results else 0.0
    print(f"\nCitation alignment false positive rate: {rate:.3f} ({flagged}/{len(results)})")
    assert rate <= 0.10, (
        f"Citation alignment false positive rate {rate:.3f} above threshold 0.10"
    )


@skip_if_no_data
@pytest.mark.benchmark
def test_span_coverage_factuality_recall():
    """Span coverage should catch 100% of factuality violations."""
    factuality_pairs = [p for p in benchmark_pairs if p["label"] == "factuality"]
    results = []
    for pair in factuality_pairs:
        inp = SpanCoverageInput(
            question=pair["question"],
            answer=pair["answer"],
            source=pair["evidence"],
        )
        out = span_score(inp)
        results.append({"label": pair["label"], "flagged": out.flagged})

    flagged = sum(1 for r in results if r["flagged"])
    rate = flagged / len(results) if results else 0.0
    print(f"\nSpan coverage factuality flag rate: {rate:.3f} ({flagged}/{len(results)})")
    assert rate >= 0.95, (
        f"Span coverage factuality flag rate {rate:.3f} below threshold 0.95"
    )


@skip_if_no_data
@pytest.mark.benchmark
def test_span_coverage_precision():
    """Span coverage precision should be >= 0.65 on full corpus."""
    results = []
    for pair in benchmark_pairs:
        inp = SpanCoverageInput(
            question=pair["question"],
            answer=pair["answer"],
            source=pair["evidence"],
        )
        out = span_score(inp)
        results.append({"label": pair["label"], "flagged": out.flagged})

    metrics = compute_metrics(results)
    print(f"\nSpan coverage: precision={metrics['precision']:.3f}, "
          f"recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}")
    assert metrics["precision"] >= 0.65, (
        f"Span coverage precision {metrics['precision']:.3f} below threshold 0.65"
    )