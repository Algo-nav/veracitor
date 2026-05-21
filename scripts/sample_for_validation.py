import os
import json
import csv
import random
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Paths ---

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(BASE_DIR, "data", "benchmark")
CORPUS_DIR = os.path.join(BASE_DIR, "data", "corpus")
OUTPUT_CSV = os.path.join(BENCHMARK_DIR, "validation_sample.csv")

# --Helpers--   

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
        pairs = data.get("pairs", [])

        for i, pair in enumerate(pairs):
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

def load_source_text(doc_id: str, doc_type: str) -> str:
    """Load the original source document text for a given doc_id."""
    type_to_dir = {
        "10-K": "10k",
        "earnings": "earnings",
        "prospectus": "prospectuses",
    }

    subdir = type_to_dir.get(doc_type, "")
    path = os.path.join(CORPUS_DIR, subdir, f"{doc_id}.json")

    if not os.path.exists(path):
        return ""

    with open(path) as f:
        data = json.load(f)

    # Return first 2000 chars as context for the validator
    return data.get("text", "")[:2000]

def stratified_sample(pairs: list[dict], sample_rate: float = 0.20) -> list[dict]:
    """
    Sample pairs stratified by doc_type and label.
    Ensures every category is represented proportionally.
    """
    # Group by (doc_type, label)
    groups = defaultdict(list)
    for pair in pairs:
        key = (pair["doc_type"], pair["label"])
        groups[key].append(pair)

    sampled = []
    for key, group in groups.items():
        n = max(1, round(len(group) * sample_rate))
        sampled.extend(random.sample(group, min(n, len(group))))

    # Shuffle so doc types are interleaved, not grouped
    random.shuffle(sampled)
    return sampled

def write_validation_csv(sampled: list[dict]):
    """Write the validation CSV with pre-filled columns and blank validator columns."""
    fieldnames = [
        "doc_id",
        "doc_type",
        "pair_index",
        "question",
        "answer",
        "label",
        "evidence",
        "reasoning",
        "source_text",
        "validator_decision",   # blank: accept / reject
        "validator_label",      # blank: correct label if rejected
        "validator_notes",      # blank: free text
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pair in sampled:
            source_text = load_source_text(pair["doc_id"], pair["doc_type"])
            writer.writerow({
                "doc_id": pair["doc_id"],
                "doc_type": pair["doc_type"],
                "pair_index": pair["pair_index"],
                "question": pair["question"],
                "answer": pair["answer"],
                "label": pair["label"],
                "evidence": pair["evidence"],
                "reasoning": pair["reasoning"],
                "source_text": source_text,
                "validator_decision": "",
                "validator_label": "",
                "validator_notes": "",
            })

def print_summary(pairs: list[dict], sampled: list[dict]):
    """Print a summary of what was sampled."""
    print(f"\nTotal pairs: {len(pairs)}")
    print(f"Sampled: {len(sampled)} ({len(sampled)/len(pairs)*100:.1f}%)")

    print("\nBreakdown by doc_type:")
    type_counts = defaultdict(int)
    for p in sampled:
        type_counts[p["doc_type"]] += 1
    for k, v in sorted(type_counts.items()):
        print(f"  {k}: {v}")

    print("\nBreakdown by label:")
    label_counts = defaultdict(int)
    for p in sampled:
        label_counts[p["label"]] += 1
    for k, v in sorted(label_counts.items()):
        print(f"  {k}: {v}")

    print(f"\nOutput: {OUTPUT_CSV}")

if __name__ == "__main__":
    random.seed(42)  # reproducible sample

    print(f"Sampling started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_pairs = load_all_pairs()
    sampled = stratified_sample(all_pairs, sample_rate=0.20)

    write_validation_csv(sampled)
    print_summary(all_pairs, sampled)

    print(f"\nDone. Open {OUTPUT_CSV} to begin validation.")
