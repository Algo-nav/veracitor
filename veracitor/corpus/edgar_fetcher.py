import os
import re
import time
import json
import requests
from html.parser import HTMLParser
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# EDGAR requires a User-Agent header identifying who you are.
# Requests without it get blocked. Format: "Name email@domain.com"
EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT", "Navneet Danturi navn07588@gmail.com")
HEADERS = {"User-Agent": EDGAR_USER_AGENT}

# Polite crawl delay. EDGAR rate limits aggressive scrapers.
CRAWL_DELAY = 0.5  # seconds between requests

# --- Pydantic schemas ---

class EdgarFetchInput(BaseModel):
    ticker: str
    filing_type : str = "10-K"
    max_text_length: int = 50000       # truncate long filings to keep token costs down

class EdgarFetchOutput(BaseModel):
    ticker: str
    cik: str
    company_name: str
    filing_date: str
    accession_number: str
    text: str                        # extracted plain text
    text_length: int
    truncated: bool
    source: str                      # full URL to the filing index
    error: Optional[str] = None

# --- HTML stripper ---

class HTMLStripper(HTMLParser):
    """Strips HTML tags and returns plain text. Standard library only."""

    def __init__(self):
        super().__init__()
        self.result = []
        self.skip_tags = {"script", "style", "ix:header"}
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            stripped = data.strip()
            if stripped:
                self.result.append(stripped)

    def get_text(self):
        return " ".join(self.result)
    

def strip_html(html: str) -> str:
    stripper = HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()


# --- Core fetcher functions ---

def get_cik(ticker: str) -> tuple[str, str]:
    """
    Resolve ticker to CIK using EDGAR company search.
    EDGAR resolves tickers directly when passed as the CIK parameter.
    Returns (cik_padded, company_name).
    """
    url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={ticker}&type=10-K"
        f"&dateb=&owner=include&count=10&output=atom"
    )
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    # CIK appears in atom feed as: CIK=0000320193 in href URLs
    cik_match = re.search(r"CIK=(\d{10})", response.text)
    if not cik_match:
        raise ValueError(f"Could not resolve CIK for ticker: {ticker}")

    cik_padded = cik_match.group(1)

    name_match = re.search(r"<company-name>(.*?)</company-name>", response.text)
    company_name = name_match.group(1) if name_match else ticker.upper()

    return cik_padded, company_name


def get_latest_filing(cik_padded: str, filing_type: str = "10-K") -> tuple[str, str]:
    """
    Get the most recent filing accession number and date for a given CIK.
    Returns (accession_number, filing_date).
    """
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    time.sleep(CRAWL_DELAY)
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    data = response.json()
    filings = data.get("filings", {}).get("recent", {})

    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])

    for form, accession, date in zip(forms, accessions, dates):
        if form == filing_type:
            return accession, date

    raise ValueError(f"No {filing_type} filing found for CIK {cik_padded}")


def get_filing_text(cik_padded: str, accession_number: str) -> tuple[str, str]:
    """
    Fetch the primary document from a filing using the JSON index.
    Works for both 10-K and 485BPOS form types.
    Returns (text, filing_index_url).
    """
    accession_clean = accession_number.replace("-", "")
    cik_int = int(cik_padded)

    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{accession_clean}/{accession_number}-index.htm"
    )

    json_index_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{accession_clean}/index.json"
    )

    time.sleep(CRAWL_DELAY)
    response = requests.get(json_index_url, headers=HEADERS)
    response.raise_for_status()

    files = response.json().get("directory", {}).get("item", [])

    # Filter to .htm files only
    htm_files = [f for f in files if f.get("name", "").endswith(".htm")]

    # Skip known exhibit types
    exhibit_keywords = ["ex-", "ex99", "consent", "opinion", "power", "certif"]

    # Strategy 1: find file whose type matches known primary types
    primary_types = {"10-K", "485BPOS", "N-1A", "S-1", "10-Q"}
    primary_doc = None
    for f in htm_files:
        if f.get("type", "") in primary_types:
            primary_doc = f.get("name")
            break

    # Strategy 2: largest .htm that isn't an exhibit
    if not primary_doc:
        candidates = []
        for f in htm_files:
            name = f.get("name", "").lower()
            if not any(kw in name for kw in exhibit_keywords):
                size = int(f.get("size", 0))
                candidates.append((size, f.get("name")))
        if candidates:
            candidates.sort(reverse=True)  # largest first
            primary_doc = candidates[0][1]

    if not primary_doc:
        raise ValueError(f"Could not find primary document in: {json_index_url}")

    doc_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{accession_clean}/{primary_doc}"
    )

    time.sleep(CRAWL_DELAY)
    doc_response = requests.get(doc_url, headers=HEADERS)
    doc_response.raise_for_status()

    text = strip_html(doc_response.text)
    return text, index_url

def fetch_10k(input_data: EdgarFetchInput) -> EdgarFetchOutput:
    """
    Main entry point. Takes a ticker, returns structured 10-K text.
    Fails gracefully: returns error field instead of raising.
    """
    try:
        cik_padded, company_name = get_cik(input_data.ticker)
        accession_number, filing_date = get_latest_filing(cik_padded, input_data.filing_type)
        text, source_url = get_filing_text(cik_padded, accession_number)

        truncated = len(text) > input_data.max_text_length
        if truncated:
            text = text[:input_data.max_text_length]

        return EdgarFetchOutput(
            ticker=input_data.ticker,
            cik=cik_padded.lstrip("0"),
            company_name=company_name,
            filing_date=filing_date,
            accession_number=accession_number,
            text=text,
            text_length=len(text),
            truncated=truncated,
            source=source_url,
        )

    except Exception as e:
        return EdgarFetchOutput(
            ticker=input_data.ticker,
            cik="",
            company_name="",
            filing_date="",
            accession_number="",
            text="",
            text_length=0,
            truncated=False,
            source="",
            error=str(e),
        )
    

# --- Test block ---

if __name__ == "__main__":
    test_tickers = ["AAPL", "MSFT", "JPM", "BRK-B", "GS"]

    for ticker in test_tickers:
        print(f"\nFetching 10-K for {ticker}...")
        result = fetch_10k(EdgarFetchInput(ticker=ticker))

        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(f"  Company: {result.company_name}")
            print(f"  Filed: {result.filing_date}")
            print(f"  Text length: {result.text_length} chars")
            print(f"  Truncated: {result.truncated}")
            print(f"  Preview: {result.text[:150]}...")