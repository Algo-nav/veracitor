import os
import time
import requests
from html.parser import HTMLParser
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
CRAWL_DELAY = 0.5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# --- Pydantic schemas ---

class EarningsFetchInput(BaseModel):
    ticker: str
    quarter: str          # e.g. "Q1 2025"
    max_text_length: int = 50000


class EarningsFetchOutput(BaseModel):
    ticker: str
    quarter: str
    source_url: str
    text: str
    text_length: int
    truncated: bool
    source: str
    error: Optional[str] = None


# --- HTML parser ---

class TranscriptStripper(HTMLParser):
    """
    Strips HTML and returns plain text.
    Skips script, style, and nav tags which contain boilerplate.
    """

    def __init__(self):
        super().__init__()
        self.result = []
        self.skip_tags = {"script", "style", "nav", "header", "footer", "aside"}
        self._skip = False
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self._skip = True
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self._skip_depth -= 1
            if self._skip_depth <= 0:
                self._skip = False
                self._skip_depth = 0

    def handle_data(self, data):
        if not self._skip:
            stripped = data.strip()
            if stripped:
                self.result.append(stripped)

    def get_text(self):
        return " ".join(self.result)


def strip_html(html: str) -> str:
    stripper = TranscriptStripper()
    stripper.feed(html)
    return stripper.get_text()


# --- Core fetcher functions ---

def search_transcript_url(ticker: str, quarter: str) -> str:
    """
    Use Tavily to find the most relevant earnings call transcript URL.
    Returns the top result URL.
    """
    client = TavilyClient(api_key=TAVILY_API_KEY)
    # Extract year from quarter string (e.g. "Q4 2024" -> "2024")
    year = quarter.split()[-1] if len(quarter.split()) > 1 else ""
    query = f"{ticker} {quarter} earnings call transcript {year} site:fool.com"

    response = client.search(
        query=query,
        max_results=5,
        include_domains=["fool.com", "seekingalpha.com", "motleyfool.com"],
    )

    results = response.get("results", [])
    if not results:
        # Retry without domain filter if no results
        response = client.search(query=query, max_results=5)
        results = response.get("results", [])

    if not results:
        raise ValueError(f"No transcript URL found for {ticker} {quarter}")

    # Return the first result URL
    return results[0]["url"]


def fetch_transcript_text(url: str) -> str:
    """
    Fetch a transcript page and return plain text body.
    """
    time.sleep(CRAWL_DELAY)
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return strip_html(response.text)


def fetch_earnings(input_data: EarningsFetchInput) -> EarningsFetchOutput:
    """
    Main entry point. Takes a ticker and quarter, returns structured transcript text.
    Fails gracefully: returns error field instead of raising.
    """
    try:
        url = search_transcript_url(input_data.ticker, input_data.quarter)
        text = fetch_transcript_text(url)

        # Basic quality check: transcript should be substantial
        if len(text) < 1000:
            raise ValueError(f"Fetched text too short ({len(text)} chars), likely not a transcript")

        truncated = len(text) > input_data.max_text_length
        if truncated:
            text = text[:input_data.max_text_length]

        return EarningsFetchOutput(
            ticker=input_data.ticker,
            quarter=input_data.quarter,
            source_url=url,
            text=text,
            text_length=len(text),
            truncated=truncated,
            source=url,
        )

    except Exception as e:
        return EarningsFetchOutput(
            ticker=input_data.ticker,
            quarter=input_data.quarter,
            source_url="",
            text="",
            text_length=0,
            truncated=False,
            source="",
            error=str(e),
        )


# --- Test block ---

if __name__ == "__main__":
    test_cases = [
        ("AAPL", "Q1 2025"),
        ("MSFT", "Q2 2025"),
        ("JPM", "Q4 2024"),
    ]

    for ticker, quarter in test_cases:
        print(f"\nFetching transcript for {ticker} {quarter}...")
        result = fetch_earnings(EarningsFetchInput(ticker=ticker, quarter=quarter))

        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(f"  URL: {result.source_url}")
            print(f"  Text length: {result.text_length} chars")
            print(f"  Truncated: {result.truncated}")
            print(f"  Preview: {result.text[:200]}...")