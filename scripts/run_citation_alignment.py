import os
import json
import sys
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veracitor.methods.citation_alignment import CitationAlignmentInput, score_pair

# --- Paths ---

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(BASE_DIR, "data", "benchmark")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "benchmark_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_all_pairs() -> list[dict]:
    """Load all generated Q&A pairs from benchmark directory."""
    all_pairs = []

    for fname in sorted(os.listdir(BENCHMARK_DIR)):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(BENCHMARK_DIR, fname)
        with open(path) as f:
            data = json.load(f)

        doc_id = data.get("doc_id", fname.replace(".json", ""))
        doc_type = data.get("doc_type", "unknown")

        for i, pair in enumerate(data.get("pairs", [])):
            all_pairs.append({
                "doc_id": doc_id,
                "doc_type": doc_type,
                "pair_index": i,
                "question": pair.get("question", ""),
                "answer": pair.get("answer", ""),
                "label": pair.get("label", ""),
                "evidence": pair.get("evidence", ""),
                "reasoning": pair.get("reasoning", ""),
            })

    return all_pairs


def run_benchmark(pairs: list[dict]) -> list[dict]:
    """Run citation alignment on all pairs. Returns results list."""
    results = []

    for pair in pairs:
        inp = CitationAlignmentInput(
            question=pair["question"],
            answer=pair["answer"],
            source=pair["evidence"],   # use evidence span as the source
            label=pair["label"],
        )
        output = score_pair(inp)

        results.append({
            "doc_id": pair["doc_id"],
            "doc_type": pair["doc_type"],
            "pair_index": pair["pair_index"],
            "label": pair["label"],
            "token_overlap": output.token_overlap,
            "rouge1": output.rouge1,
            "rouge2": output.rouge2,
            "citation_alignment_score": output.citation_alignment_score,
            "flagged": output.flagged,
            "error": output.error,
        })

    return results


def compute_metrics(results: list[dict]) -> dict:
    """
    Compute precision, recall, F1 for hallucination detection.

    Positive class = hallucinated (faithfulness, factuality, relevance).
    Negative class = clean.

    flagged=True means the method thinks it's hallucinated.
    """
    tp = sum(1 for r in results if r["flagged"] and r["label"] != "clean")
    fp = sum(1 for r in results if r["flagged"] and r["label"] == "clean")
    tn = sum(1 for r in results if not r["flagged"] and r["label"] == "clean")
    fn = sum(1 for r in results if not r["flagged"] and r["label"] != "clean")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-label breakdown
    label_breakdown = defaultdict(lambda: {"flagged": 0, "total": 0})
    for r in results:
        label_breakdown[r["label"]]["total"] += 1
        if r["flagged"]:
            label_breakdown[r["label"]]["flagged"] += 1

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "label_breakdown": {
            k: {
                "flagged": v["flagged"],
                "total": v["total"],
                "flag_rate": round(v["flagged"] / v["total"], 4) if v["total"] > 0 else 0.0
            }
            for k, v in label_breakdown.items()
        }
    }


def print_report(metrics: dict, total_pairs: int):
    """Print a clean benchmark report."""
    print(f"\n--- Citation Alignment Benchmark Results ---")
    print(f"Total pairs evaluated: {total_pairs}")
    print(f"\nOverall metrics (hallucination detection):")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1:        {metrics['f1']}")
    print(f"\nConfusion matrix:")
    print(f"  TP (hallucinated, flagged):     {metrics['tp']}")
    print(f"  FP (clean, flagged):            {metrics['fp']}")
    print(f"  TN (clean, not flagged):        {metrics['tn']}")
    print(f"  FN (hallucinated, not flagged): {metrics['fn']}")
    print(f"\nFlag rate by label:")
    for label, stats in sorted(metrics["label_breakdown"].items()):
        print(f"  {label:15s}: {stats['flagged']:3d}/{stats['total']:3d} flagged ({stats['flag_rate']*100:.1f}%)")


if __name__ == "__main__":
    print(f"Citation alignment benchmark started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    pairs = load_all_pairs()
    print(f"Loaded {len(pairs)} pairs")

    results = run_benchmark(pairs)

    # Save full results
    output_path = os.path.join(RESULTS_DIR, "citation_alignment_results.json")
    output = {
        "method": "citation_alignment",
        "run_at": datetime.now().isoformat(),
        "total_pairs": len(results),
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Compute and print metrics
    metrics = compute_metrics(results)

    # Save metrics separately
    metrics_path = os.path.join(RESULTS_DIR, "citation_alignment_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print_report(metrics, len(results))
    print(f"\nFull results saved to {output_path}")
    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")