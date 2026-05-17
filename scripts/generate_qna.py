import os
import json
import time
import argparse
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --- Paths ---

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(BASE_DIR, "data", "corpus")
BENCHMARK_DIR = os.path.join(BASE_DIR, "data", "benchmark")

TENK_DIR = os.path.join(CORPUS_DIR, "10k")
EARNINGS_DIR = os.path.join(CORPUS_DIR, "earnings")
PROSPECTUS_DIR = os.path.join(CORPUS_DIR, "prospectuses")


# --- Prompt ---

SYSTEM_PROMPT = """You are a benchmark dataset generator for hallucination detection in finance RAG systems.

Your task is to generate Q&A pairs from a given financial document excerpt. Each pair tests whether a hallucination detection system can correctly identify supported vs unsupported answers.

You must return ONLY a JSON array with no preamble, no explanation, no markdown code fences. Raw JSON only.

Each item in the array must have exactly these fields:
- "question": a specific, answerable question about the document content
- "answer": the generated answer (supported or hallucinated depending on label)
- "label": one of "clean", "faithfulness", "factuality", "relevance"
- "evidence": the exact span from the document that supports or contradicts the answer (empty string for relevance violations)
- "reasoning": one sentence explaining why this label applies

Label definitions:
- "clean": the answer is fully and accurately supported by the document
- "faithfulness": the answer makes a specific claim that contradicts or goes beyond what the document states (subtle distortions: wrong numbers, wrong direction, wrong timeframe, added claims not in source)
- "factuality": the answer contradicts a well-known real-world fact not dependent on the document (e.g. wrong CEO, wrong founding year, wrong headquarters country)
- "relevance": the answer does not address the question asked, regardless of whether it is factually accurate

Distribution rules (STRICTLY enforced - you will be penalized for violations):
- You MUST generate EXACTLY 4 pairs with label "clean". No more, no less.
- You MUST generate EXACTLY 4 pairs with label "faithfulness". No more, no less.
- You MUST generate EXACTLY 1 pair with label "factuality". No more, no less.
- You MUST generate EXACTLY 1 pair with label "relevance". No more, no less.
- Count your labels before returning. If the counts are wrong, fix them before returning.
- Generating fewer than 4 faithfulness pairs is a critical error.

Quality rules:
- Faithfulness violations must be subtle. Change a number by 10-30%, flip a direction (increase vs decrease), shift a timeframe, or add a plausible but unsupported claim. Do NOT make obvious errors.
- Questions must be specific and grounded in the document content.
- Evidence spans must be verbatim excerpts from the document (max 100 words).
- Do not repeat questions across pairs."""

USER_TEMPLATE = """Document type: {doc_type}
Document ID: {doc_id}

Document excerpt:
{text}

Generate exactly 10 Q&A pairs following the system instructions."""


# --- Core generation function ---

def sanitize_text(text: str) -> str:
    """
    Remove characters that break JSON generation when embedded in prompts.
    """
    # Replace characters that cause JSON encoding issues
    replacements = {
        "\u2019": "'", "\u2018": "'",   # smart apostrophes
        "\u201c": '"', "\u201d": '"',   # smart quotes
        "\u2013": "-", "\u2014": "-",   # en/em dashes
        "\u00a0": " ",                  # non-breaking space
        "\u2022": "-",                  # bullet point
        "\r\n": " ", "\r": " ",         # carriage returns
        "\n": " ",                      # newlines
        "\\": "/",                      # backslashes
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text

def generate_qna(doc_type: str, doc_id: str, text: str, retries: int = 2) -> list[dict]:
    """
    Generate 10 Q&A pairs for a document using Claude Sonnet.
    Retries on JSON parse failure up to `retries` times.
    """
    last_error = None

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": USER_TEMPLATE.format(
                            doc_type=doc_type,
                            doc_id=doc_id,
                            text=sanitize_text(text[:8000]),
                        )
                    }
                ]
            )

            raw = response.content[0].text.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            # Sanitize smart quotes in output
            raw = raw.replace("\u2019", "'").replace("\u2018", "'")
            raw = raw.replace("\u201c", '"').replace("\u201d", '"')

            # Fix unescaped newlines inside JSON string values
            import re
            raw = re.sub(r'(?<!\\)\n(?!["\[\]{},])', ' ', raw)

            pairs = json.loads(raw)
            return pairs

        except json.JSONDecodeError as e:
            last_error = e
            print(f"    [retry {attempt + 1}/{retries}] JSON parse failed: {e}")
            time.sleep(2)

    raise last_error


def validate_distribution(pairs: list[dict]) -> list[str]:
    """
    Check that the generated pairs match the required label distribution.
    Returns a list of validation errors (empty if valid).
    """
    errors = []
    from collections import Counter
    counts = Counter(p.get("label") for p in pairs)

    expected = {"clean": 4, "faithfulness": 4, "factuality": 1, "relevance": 1}
    for label, expected_count in expected.items():
        actual = counts.get(label, 0)
        if actual != expected_count:
            errors.append(f"Expected {expected_count} '{label}' pairs, got {actual}")

    return errors


def load_corpus_document(path: str) -> dict:
    """Load a saved corpus document JSON."""
    with open(path) as f:
        return json.load(f)


def get_all_documents() -> list[tuple[str, str, str]]:
    """
    Returns list of (doc_type, doc_id, text) tuples for all corpus documents.
    """
    docs = []

    for fname in sorted(os.listdir(TENK_DIR)):
        if fname.endswith(".json"):
            data = load_corpus_document(os.path.join(TENK_DIR, fname))
            doc_id = fname.replace(".json", "")
            docs.append(("10-K", doc_id, data.get("text", "")))

    for fname in sorted(os.listdir(EARNINGS_DIR)):
        if fname.endswith(".json"):
            data = load_corpus_document(os.path.join(EARNINGS_DIR, fname))
            doc_id = fname.replace(".json", "")
            docs.append(("earnings", doc_id, data.get("text", "")))

    for fname in sorted(os.listdir(PROSPECTUS_DIR)):
        if fname.endswith(".json"):
            data = load_corpus_document(os.path.join(PROSPECTUS_DIR, fname))
            doc_id = fname.replace(".json", "")
            docs.append(("prospectus", doc_id, data.get("text", "")))

    return docs


def run_generation(refresh: bool, doc_type_filter: str = "all"):
    """Main generation loop."""
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    docs = get_all_documents()

    if doc_type_filter != "all":
        docs = [d for d in docs if d[0] == doc_type_filter]

    print(f"Documents to process: {len(docs)}")
    print(f"Estimated Q&A pairs: {len(docs) * 10}")

    results = []
    for i, (doc_type, doc_id, text) in enumerate(docs):
        output_path = os.path.join(BENCHMARK_DIR, f"{doc_id}.json")

        if not refresh and os.path.exists(output_path):
            print(f"  [skip] {doc_id}")
            results.append({"doc_id": doc_id, "status": "skipped"})
            continue

        print(f"  [{i+1}/{len(docs)}] Generating Q&A for {doc_id} ({doc_type})...")

        try:
            pairs = generate_qna(doc_type, doc_id, text)
            errors = validate_distribution(pairs)

            if errors:
                print(f"  [warn] {doc_id} distribution issues: {errors}")

            output = {
                "doc_id": doc_id,
                "doc_type": doc_type,
                "generated_at": datetime.now().isoformat(),
                "num_pairs": len(pairs),
                "distribution_errors": errors,
                "pairs": pairs,
            }

            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)

            print(f"  [ok]   {doc_id}: {len(pairs)} pairs, errors: {errors or 'none'}")
            results.append({"doc_id": doc_id, "status": "ok", "pairs": len(pairs)})

        except Exception as e:
            print(f"  [fail] {doc_id}: {e}")
            results.append({"doc_id": doc_id, "status": "failed", "error": str(e)})

        # Rate limit buffer between API calls
        time.sleep(1)

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] == "failed"]
    total_pairs = sum(r.get("pairs", 0) for r in ok)

    print(f"\n--- Summary ---")
    print(f"  OK:      {len(ok)} documents, {total_pairs} pairs")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Failed:  {len(failed)}")
    if failed:
        for r in failed:
            print(f"  {r['doc_id']}: {r.get('error')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Q&A pairs for Veracitor benchmark")
    parser.add_argument("--refresh", action="store_true", help="Regenerate existing files")
    parser.add_argument(
        "--type",
        choices=["10-K", "earnings", "prospectus", "all"],
        default="all",
        help="Filter by document type"
    )
    args = parser.parse_args()

    print(f"Q&A generation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run_generation(refresh=args.refresh, doc_type_filter=args.type)
    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")