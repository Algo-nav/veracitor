import os
import json
import argparse
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veracitor.corpus.edgar_fetcher import fetch_10k, EdgarFetchInput
from veracitor.corpus.earnings_fetcher import fetch_earnings, EarningsFetchInput
from veracitor.corpus.prospectus_fetcher import fetch_prospectus, ProspectusFetchInput

# --- Document lists ---

TENK_TICKERS = [
    "AAPL", "MSFT", "JPM", "GS", "BRK-B",
    "BAC", "WFC", "AMZN", "GOOGL", "TSLA"
]

EARNINGS_CALLS = [
    ("AAPL", "Q1 2025"),
    ("MSFT", "Q2 2025"),
    ("JPM", "Q4 2024"),
    ("GS", "Q4 2024"),
    ("AMZN", "Q4 2024"),
    ("META", "Q4 2024"),
    ("NVDA", "Q4 2024"),
    ("BAC", "Q4 2024"),
    ("WFC", "Q4 2024"),
    ("GOOGL", "Q4 2024"),
]

PROSPECTUS_TICKERS = [
    "VFINX", "FCNTX", "AGTHX", "AIVSX", "PRGFX",
    "PRFDX", "DODGX", "SEQUX", "VWUSX", "FXAIX"
]

# --- Paths ---

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(BASE_DIR, "data", "corpus")
TENK_DIR = os.path.join(CORPUS_DIR, "10k")
EARNINGS_DIR = os.path.join(CORPUS_DIR, "earnings")
PROSPECTUS_DIR = os.path.join(CORPUS_DIR, "prospectuses")


def save_document(path: str, data: dict):
    """Save a document as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_existing(path: str) -> bool:
    """Check if a document already exists on disk."""
    return os.path.exists(path)


def run_10k(refresh: bool) -> list[dict]:
    """Fetch all 10-K filings."""
    results = []
    print("\n--- 10-K Filings ---")

    for ticker in TENK_TICKERS:
        path = os.path.join(TENK_DIR, f"{ticker}.json")

        if not refresh and load_existing(path):
            print(f"  [skip] {ticker} (already exists)")
            results.append({"type": "10-K", "id": ticker, "status": "skipped"})
            continue

        print(f"  Fetching {ticker}...")
        result = fetch_10k(EdgarFetchInput(ticker=ticker))

        if result.error:
            print(f"  [fail] {ticker}: {result.error}")
            results.append({"type": "10-K", "id": ticker, "status": "failed", "error": result.error})
        else:
            save_document(path, result.model_dump())
            print(f"  [ok]   {ticker}: {result.text_length} chars, filed {result.filing_date}")
            results.append({"type": "10-K", "id": ticker, "status": "ok"})

    return results


def run_earnings(refresh: bool) -> list[dict]:
    """Fetch all earnings call transcripts."""
    results = []
    print("\n--- Earnings Call Transcripts ---")

    for ticker, quarter in EARNINGS_CALLS:
        file_id = f"{ticker}_{quarter.replace(' ', '_')}"
        path = os.path.join(EARNINGS_DIR, f"{file_id}.json")

        if not refresh and load_existing(path):
            print(f"  [skip] {ticker} {quarter} (already exists)")
            results.append({"type": "earnings", "id": file_id, "status": "skipped"})
            continue

        print(f"  Fetching {ticker} {quarter}...")
        result = fetch_earnings(EarningsFetchInput(ticker=ticker, quarter=quarter))

        if result.error:
            print(f"  [fail] {ticker} {quarter}: {result.error}")
            results.append({"type": "earnings", "id": file_id, "status": "failed", "error": result.error})
        else:
            save_document(path, result.model_dump())
            print(f"  [ok]   {ticker} {quarter}: {result.text_length} chars")
            results.append({"type": "earnings", "id": file_id, "status": "ok"})

    return results


def run_prospectuses(refresh: bool) -> list[dict]:
    """Fetch all fund prospectuses."""
    results = []
    print("\n--- Fund Prospectuses ---")

    for ticker in PROSPECTUS_TICKERS:
        path = os.path.join(PROSPECTUS_DIR, f"{ticker}.json")

        if not refresh and load_existing(path):
            print(f"  [skip] {ticker} (already exists)")
            results.append({"type": "prospectus", "id": ticker, "status": "skipped"})
            continue

        print(f"  Fetching {ticker}...")
        result = fetch_prospectus(ProspectusFetchInput(ticker=ticker))

        if result.error:
            print(f"  [fail] {ticker}: {result.error}")
            results.append({"type": "prospectus", "id": ticker, "status": "failed", "error": result.error})
        else:
            save_document(path, result.model_dump())
            print(f"  [ok]   {ticker}: {result.text_length} chars, filed {result.filing_date}")
            results.append({"type": "prospectus", "id": ticker, "status": "ok"})

    return results


def print_summary(all_results: list[dict]):
    """Print a clean summary table."""
    ok = [r for r in all_results if r["status"] == "ok"]
    skipped = [r for r in all_results if r["status"] == "skipped"]
    failed = [r for r in all_results if r["status"] == "failed"]

    print("\n--- Summary ---")
    print(f"  Total:   {len(all_results)}")
    print(f"  OK:      {len(ok)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Failed:  {len(failed)}")

    if failed:
        print("\nFailures:")
        for r in failed:
            print(f"  {r['type']} {r['id']}: {r.get('error', 'unknown error')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Veracitor corpus")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch existing documents")
    parser.add_argument(
        "--type",
        choices=["10k", "earnings", "prospectuses", "all"],
        default="all",
        help="Which document type to fetch"
    )
    args = parser.parse_args()

    print(f"Corpus generation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Refresh: {args.refresh}")

    all_results = []

    if args.type in ("10k", "all"):
        all_results += run_10k(args.refresh)

    if args.type in ("earnings", "all"):
        all_results += run_earnings(args.refresh)

    if args.type in ("prospectuses", "all"):
        all_results += run_prospectuses(args.refresh)

    print_summary(all_results)
    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")