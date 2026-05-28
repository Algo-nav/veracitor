import re
from typing import Optional
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from veracitor.retrieval.chunker import Chunk, chunk_text, ChunkingConfig


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    rank: int


class BM25RetrieverConfig(BaseModel):
    top_k: int = 3
    min_score: float = -999.0


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase and tokenize text for BM25 indexing."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [t for t in text.split() if t]


class BM25Retriever:
    """
    BM25-based retriever over a document corpus.
    Chunks a document on initialization, builds BM25 index.
    """

    def __init__(self, document: str, config: Optional[BM25RetrieverConfig] = None):
        self.config = config or BM25RetrieverConfig()
        self.chunks = chunk_text(document)

        if not self.chunks:
            raise ValueError("Document produced no chunks. Check document length.")

        tokenized = [tokenize_for_bm25(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """
        Retrieve top-k chunks most relevant to the query.
        Returns ranked list of RetrievalResult objects.
        """
        query_tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for rank, (chunk_idx, score) in enumerate(ranked[:self.config.top_k]):
            if score >= self.config.min_score:
                results.append(RetrievalResult(
                    chunk=self.chunks[chunk_idx],
                    score=round(float(score), 4),
                    rank=rank + 1,
                ))

        return results

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)


# --- Test block ---

if __name__ == "__main__":
    document = """
    Apple Inc. reported total net sales of $391 billion for fiscal year 2025,
    representing a 4% increase compared to the prior year. iPhone revenue
    accounted for $201 billion, or approximately 51% of total net sales.
    Services revenue reached $96 billion, up 12% year over year.
    The company reported net income of $94 billion for the fiscal year.
    Operating cash flow was $118 billion, with free cash flow of $108 billion.
    Apple returned $110 billion to shareholders through dividends and buybacks.
    The board approved a new $90 billion share repurchase authorization.
    International sales accounted for 58% of total revenue.
    The company ended the fiscal year with $167 billion in cash and investments.
    Mac revenue was $16 billion, up 8% from the prior year.
    iPad revenue was $7 billion, down 6% year over year.
    Wearables, Home and Accessories revenue was $9 billion.
    JPMorgan Chase reported net income of $58.5 billion for fiscal year 2025.
    Return on equity was 17% for the full year. Total assets were $4.0 trillion.
    The CET1 capital ratio was 15.7% under the standardized approach.
    Net interest income was $92 billion for the full year 2025.
    Investment banking fees totaled $9.3 billion, up 18% year over year.
    Credit loss provisions were $10.7 billion, reflecting normalizing credit conditions.
    Consumer and Community Banking revenue was $58 billion for the year.
    Commercial Banking reported revenue of $16 billion, up 6% year over year.
    Asset and Wealth Management reported revenue of $20 billion for fiscal 2025.
    The firm employed approximately 317,000 people globally at year end.
    Vanguard 500 Index Fund seeks to track the S&P 500 index performance.
    The fund has an expense ratio of 0.04% and minimum investment of $3,000.
    Total net assets were $1.1 trillion as of the most recent quarter.
    The fund holds all 500 stocks in the S&P 500 index by market weight.
    """

    retriever = BM25Retriever(document)
    print(f"Document chunked into {retriever.num_chunks} chunks")

    queries = [
        "What was Apple's total revenue?",
        "What was JPMorgan net income?",
        "What is the expense ratio of the Vanguard fund?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query)
        if not results:
            print("  No results above threshold")
        for r in results:
            print(f"  Rank {r.rank} (score {r.score:.2f}): {r.chunk.text[:120]}...")