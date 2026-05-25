import os
import json
import sys
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "benchmark_results")


def load_results(filename: str) -> dict:
    """Load a benchmark results JSON file, keyed by (doc_id, pair_index)."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path) as f:
        data = json.load(f)
    return {
        (r["doc_id"], r["pair_index"]): r
        for r in data["results"]
    }


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


def print_metrics(name: str, metrics: dict):
    print(f"\n{name}:")
    print(f"  Precision: {metrics['precision']}  Recall: {metrics['recall']}  F1: {metrics['f1']}  (n={metrics['total_pairs']})")
    print(f"  Flag rate by label:")
    for label, stats in sorted(metrics["label_breakdown"].items()):
        print(f"    {label:15s}: {stats['flagged']:3d}/{stats['total']:3d} ({stats['flag_rate']*100:.1f}%)")


if __name__ == "__main__":
    print(f"Ensemble benchmark started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load all method results
    citation = load_results("citation_alignment_results.json")
    nli = load_results("nli_entailment_results.json")
    span = load_results("span_coverage_results.json")

    # Use only pairs present in all three full-corpus methods
    common_keys = set(citation.keys()) & set(nli.keys()) & set(span.keys())
    print(f"Pairs in all three full-corpus methods: {len(common_keys)}")

    # Ensemble 1: flag if ANY method flags (OR ensemble, max recall)
    or_results = []
    for key in sorted(common_keys):
        c = citation[key]
        n = nli[key]
        s = span[key]
        flagged = c["flagged"] or n["flagged"] or s["flagged"]
        or_results.append({
            "doc_id": c["doc_id"],
            "pair_index": c["pair_index"],
            "label": c["label"],
            "flagged": flagged,
            "methods_flagged": sum([c["flagged"], n["flagged"], s["flagged"]]),
        })

    # Ensemble 2: flag if ANY TWO methods flag (majority vote, balanced)
    majority_results = []
    for key in sorted(common_keys):
        c = citation[key]
        n = nli[key]
        s = span[key]
        votes = sum([c["flagged"], n["flagged"], s["flagged"]])
        flagged = votes >= 2
        majority_results.append({
            "doc_id": c["doc_id"],
            "pair_index": c["pair_index"],
            "label": c["label"],
            "flagged": flagged,
            "methods_flagged": votes,
        })

    # Ensemble 3: citation + NLI only (two strongest methods)
    citation_nli_results = []
    for key in sorted(common_keys):
        c = citation[key]
        n = nli[key]
        flagged = c["flagged"] or n["flagged"]
        citation_nli_results.append({
            "doc_id": c["doc_id"],
            "pair_index": c["pair_index"],
            "label": c["label"],
            "flagged": flagged,
        })

    # Print all comparisons
    print("\n=== Ensemble Benchmark Results ===")
    print_metrics("Citation alignment (baseline)", compute_metrics(list(citation.values())))
    print_metrics("NLI entailment (baseline)", compute_metrics(list(nli.values())))
    print_metrics("Span coverage (baseline)", compute_metrics(list(span.values())))
    print_metrics("Ensemble: OR (any method flags)", compute_metrics(or_results))
    print_metrics("Ensemble: Majority vote (2+ methods flag)", compute_metrics(majority_results))
    print_metrics("Ensemble: Citation + NLI only", compute_metrics(citation_nli_results))

    # Save ensemble results
    output = {
        "run_at": datetime.now().isoformat(),
        "ensembles": {
            "or": compute_metrics(or_results),
            "majority_vote": compute_metrics(majority_results),
            "citation_nli": compute_metrics(citation_nli_results),
        }
    }
    output_path = os.path.join(RESULTS_DIR, "ensemble_metrics.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")