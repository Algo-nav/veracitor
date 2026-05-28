"""
Strict benchmark tests for veracitor detection methods.
Asserts results within 5% of published benchmark numbers.
Run manually before PyPI publish to verify no regression.

Run with:
    pytest tests/test_benchmark_strict.py -v -m benchmark_strict

Published benchmark numbers (from veracitor-finance-bench, 309 pairs):
    Citation alignment: precision=0.9479, recall=0.5056, f1=0.6594
    Span coverage:      precision=0.7179, recall=0.6222, f1=0.6667
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(BASE_DIR, "data", "benchmark")

TOLERANCE = 0.05  # 5% tolerance on published numbers

# Published benchmark results
CITATION_PUBLISHED = {"precision": 0.9479, "recall": 0.5056, "f1": 0.6594}
SPAN_PUBLISHED = {"precision": 0.7179, "recall": 0.6222, "f1": 0.6667}


def load_benchmark_pairs() -> list[dict]:
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
    return {"precision": precision, "recall": recall, "f1": f1}


benchmark_pairs = load_benchmark_pairs()
skip_if_no_data = pytest.mark.skipif(
    len(benchmark_pairs) == 0,
    reason="Benchmark data not found. Run scripts/generate_qna.py first."
)


@skip_if_no_data
@pytest.mark.benchmark_strict
def test_citation_alignment_matches_published():
    """Citation alignment metrics should be within 5% of published numbers."""
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
    print(f"\nCitation alignment actual:    "
          f"precision={metrics['precision']:.4f}, "
          f"recall={metrics['recall']:.4f}, "
          f"f1={metrics['f1']:.4f}")
    print(f"Citation alignment published: "
          f"precision={CITATION_PUBLISHED['precision']:.4f}, "
          f"recall={CITATION_PUBLISHED['recall']:.4f}, "
          f"f1={CITATION_PUBLISHED['f1']:.4f}")

    assert abs(metrics["precision"] - CITATION_PUBLISHED["precision"]) <= TOLERANCE, (
        f"Precision drift: {metrics['precision']:.4f} vs {CITATION_PUBLISHED['precision']:.4f}"
    )
    assert abs(metrics["recall"] - CITATION_PUBLISHED["recall"]) <= TOLERANCE, (
        f"Recall drift: {metrics['recall']:.4f} vs {CITATION_PUBLISHED['recall']:.4f}"
    )
    assert abs(metrics["f1"] - CITATION_PUBLISHED["f1"]) <= TOLERANCE, (
        f"F1 drift: {metrics['f1']:.4f} vs {CITATION_PUBLISHED['f1']:.4f}"
    )


@skip_if_no_data
@pytest.mark.benchmark_strict
def test_span_coverage_matches_published():
    """Span coverage metrics should be within 5% of published numbers."""
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
    print(f"\nSpan coverage actual:    "
          f"precision={metrics['precision']:.4f}, "
          f"recall={metrics['recall']:.4f}, "
          f"f1={metrics['f1']:.4f}")
    print(f"Span coverage published: "
          f"precision={SPAN_PUBLISHED['precision']:.4f}, "
          f"recall={SPAN_PUBLISHED['recall']:.4f}, "
          f"f1={SPAN_PUBLISHED['f1']:.4f}")

    assert abs(metrics["precision"] - SPAN_PUBLISHED["precision"]) <= TOLERANCE, (
        f"Precision drift: {metrics['precision']:.4f} vs {SPAN_PUBLISHED['precision']:.4f}"
    )
    assert abs(metrics["recall"] - SPAN_PUBLISHED["recall"]) <= TOLERANCE, (
        f"Recall drift: {metrics['recall']:.4f} vs {SPAN_PUBLISHED['recall']:.4f}"
    )
    assert abs(metrics["f1"] - SPAN_PUBLISHED["f1"]) <= TOLERANCE, (
        f"F1 drift: {metrics['f1']:.4f} vs {SPAN_PUBLISHED['f1']:.4f}"
    )


@skip_if_no_data
@pytest.mark.benchmark_strict
def test_citation_alignment_label_breakdown():
    """Per-label flag rates should match published numbers within 5%."""
    published = {
        "clean": 0.039,
        "factuality": 1.0,
        "faithfulness": 0.275,
        "relevance": 0.933,
    }

    from collections import defaultdict
    label_results = defaultdict(list)

    for pair in benchmark_pairs:
        inp = CitationAlignmentInput(
            question=pair["question"],
            answer=pair["answer"],
            source=pair["evidence"],
        )
        out = citation_score(inp)
        label_results[pair["label"]].append(out.flagged)

    print()
    for label, pub_rate in published.items():
        results = label_results.get(label, [])
        if not results:
            continue
        actual_rate = sum(results) / len(results)
        print(f"  {label:15s}: actual={actual_rate:.3f}, published={pub_rate:.3f}")
        assert abs(actual_rate - pub_rate) <= TOLERANCE, (
            f"{label} flag rate drift: {actual_rate:.3f} vs {pub_rate:.3f}"
        )