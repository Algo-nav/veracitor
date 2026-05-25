import os
import json
import sys
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veracitor.methods.span_coverage import SpanCoverageInput, score_pair

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(BASE_DIR, "data", "benchmark")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "benchmark_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_all_pairs() -> list[dict]:
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
            })
    return all_pairs


def run_benchmark(pairs: list[dict]) -> list[dict]:
    results = []
    for i, pair in enumerate(pairs):
        if (i + 1) % 30 == 0:
            print(f"  Progress: {i+1}/{len(pairs)}")
        inp = SpanCoverageInput(
            question=pair["question"],
            answer=pair["answer"],
            source=pair["evidence"],
            label=pair["label"],
        )
        output = score_pair(inp)
        results.append({
            "doc_id": pair["doc_id"],
            "doc_type": pair["doc_type"],
            "pair_index": pair["pair_index"],
            "label": pair["label"],
            "coverage_score": output.coverage_score,
            "answer_phrases": output.answer_phrases,
            "uncovered_phrases": output.uncovered_phrases,
            "flagged": output.flagged,
            "error": output.error,
        })
    return results


def compute_metrics(results: list[dict]) -> dict:
    tp = sum(1 for r in results if r["flagged"] and r["label"] != "clean")
    fp = sum(1 for r in results if r["flagged"] and r["label"] == "clean")
    tn = sum(1 for r in results if not r["flagged"] and r["label"] == "clean")
    fn = sum(1 for r in results if not r["flagged"] and r["label"] != "clean")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
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
        "total_pairs": len(results),
        "label_breakdown": {
            k: {
                "flagged": v["flagged"],
                "total": v["total"],
                "flag_rate": round(v["flagged"] / v["total"], 4) if v["total"] > 0 else 0.0
            }
            for k, v in label_breakdown.items()
        }
    }


def print_report(metrics: dict):
    print(f"\n--- Span Coverage Benchmark Results ---")
    print(f"Total pairs evaluated: {metrics['total_pairs']}")
    print(f"\nOverall metrics:")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1:        {metrics['f1']}")
    print(f"\nConfusion matrix:")
    print(f"  TP: {metrics['tp']}  FP: {metrics['fp']}")
    print(f"  TN: {metrics['tn']}  FN: {metrics['fn']}")
    print(f"\nFlag rate by label:")
    for label, stats in sorted(metrics["label_breakdown"].items()):
        print(f"  {label:15s}: {stats['flagged']:3d}/{stats['total']:3d} ({stats['flag_rate']*100:.1f}%)")


if __name__ == "__main__":
    print(f"Span coverage benchmark started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pairs = load_all_pairs()
    print(f"Loaded {len(pairs)} pairs")
    results = run_benchmark(pairs)

    output_path = os.path.join(RESULTS_DIR, "span_coverage_results.json")
    output = {
        "method": "span_coverage",
        "run_at": datetime.now().isoformat(),
        "total_pairs": len(results),
        "note": "Retrieval-grounded span coverage proxy for token-level uncertainty. "
                "Replaces logprob-based uncertainty since Anthropic API does not expose logprobs.",
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    metrics = compute_metrics(results)
    metrics_path = os.path.join(RESULTS_DIR, "span_coverage_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print_report(metrics, )
    print(f"\nResults saved to {output_path}")
    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")