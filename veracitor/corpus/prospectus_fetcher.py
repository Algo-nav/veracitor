import os
import time
import requests
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Reuse core EDGAR functions from the 10-K fetcher
from veracitor.corpus.edgar_fetcher import (
    get_cik,
    get_latest_filing,
    get_filing_text,
    HEADERS,
    CRAWL_DELAY,
)

load_dotenv()


# --- Pydantic schemas ---

class ProspectusFetchInput(BaseModel):
    ticker: str
    form_type: str = "485BPOS"      # annual prospectus update, most current
    max_text_length: int = 50000


class ProspectusFetchOutput(BaseModel):
    ticker: str
    cik: str
    fund_name: str
    filing_date: str
    accession_number: str
    form_type: str
    text: str
    text_length: int
    truncated: bool
    source: str
    error: Optional[str] = None


def fetch_prospectus(input_data: ProspectusFetchInput) -> ProspectusFetchOutput:
    """
    Main entry point. Takes a fund ticker, returns structured prospectus text.
    Reuses EDGAR fetch logic from edgar_fetcher.
    Fails gracefully: returns error field instead of raising.
    """
    try:
        cik_padded, fund_name = get_cik(input_data.ticker)
        accession_number, filing_date = get_latest_filing(
            cik_padded, input_data.form_type
        )
        text, source_url = get_filing_text(cik_padded, accession_number)

        truncated = len(text) > input_data.max_text_length
        if truncated:
            text = text[:input_data.max_text_length]

        return ProspectusFetchOutput(
            ticker=input_data.ticker,
            cik=cik_padded.lstrip("0"),
            fund_name=fund_name,
            filing_date=filing_date,
            accession_number=accession_number,
            form_type=input_data.form_type,
            text=text,
            text_length=len(text),
            truncated=truncated,
            source=source_url,
        )

    except Exception as e:
        return ProspectusFetchOutput(
            ticker=input_data.ticker,
            cik="",
            fund_name="",
            filing_date="",
            accession_number="",
            form_type=input_data.form_type,
            text="",
            text_length=0,
            truncated=False,
            source="",
            error=str(e),
        )


# --- Test block ---

if __name__ == "__main__":
    # Mix of equity, bond, and money market funds
    test_tickers = ["VFINX", "FCNTX", "AGTHX"]

    for ticker in test_tickers:
        print(f"\nFetching prospectus for {ticker}...")
        result = fetch_prospectus(ProspectusFetchInput(ticker=ticker))

        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(f"  Fund: {result.fund_name}")
            print(f"  Filed: {result.filing_date}")
            print(f"  Form: {result.form_type}")
            print(f"  Text length: {result.text_length} chars")
            print(f"  Truncated: {result.truncated}")
            print(f"  Preview: {result.text[:200]}...")