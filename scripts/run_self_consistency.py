import os
import json
import sys
import time
import random
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veracitor.methods.self_consistency import SelfConsistencyInput, score_pair

# --- Paths ---

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


def stratified_sample(pairs: list[dict], n: int = 50) -> list[dict]:
    random.seed(42)
    by_label = defaultdict(list)
    for p in pairs:
        by_label[p["label"]].append(p)
    targets = {"clean": 20, "faithfulness": 20, "factuality": 5, "relevance": 5}
    sampled = []
    for label, target in targets.items():
        group = by_label.get(label, [])
        sampled.extend(random.sample(group, min(target, len(group))))
    random.shuffle(sampled)
    return sampled


def run_benchmark(pairs: list[dict]) -> list[dict]:
    results = []
    for i, pair in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {pair['doc_id']} | label: {pair['label']}")
        inp = SelfConsistencyInput(
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
            "consistency_score": output.consistency_score,
            "flagged": output.flagged,
            "error": output.error,
        })
        time.sleep(0.5)
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
        "sample_size": len(results),
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
    print(f"\n--- Self-Consistency Benchmark Results ---")
    print(f"Sample size: {metrics['sample_size']} pairs (stratified)")
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
    print(f"Self-consistency benchmark started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    all_pairs = load_all_pairs()
    print(f"Total pairs available: {len(all_pairs)}")
    sampled = stratified_sample(all_pairs, n=50)
    print(f"Sampled: {len(sampled)} pairs (3 API calls per pair, ~150 total calls)")
    results = run_benchmark(sampled)

    output_path = os.path.join(RESULTS_DIR, "self_consistency_results.json")
    output = {
        "method": "self_consistency",
        "model": "claude-sonnet-4-6",
        "n_samples": 3,
        "threshold": 0.48,
        "scoring": "answer_grounded_rouge1",
        "run_at": datetime.now().isoformat(),
        "sample_size": len(results),
        "note": "Answer-grounded SelfCheckGPT pattern. Provided answer compared against "
                "N model regenerations from source. ROUGE-1 scoring.",
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    metrics = compute_metrics(results)
    metrics_path = os.path.join(RESULTS_DIR, "self_consistency_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print_report(metrics)
    print(f"\nResults saved to {output_path}")
    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")