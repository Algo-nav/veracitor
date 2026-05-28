import os
import time
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic

from veracitor.retrieval.chunker import Chunk, chunk_text, ChunkingConfig
from veracitor.retrieval.bm25_retriever import (
    BM25Retriever,
    BM25RetrieverConfig,
    RetrievalResult,
    tokenize_for_bm25,
)
from rank_bm25 import BM25Okapi

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Context generation uses Haiku: cheapest model, sufficient for
# 50-100 token context snippets. Cost: ~$0.002 per 20-chunk document.
CONTEXT_MODEL = "claude-haiku-4-5-20251001"

CONTEXT_PROMPT = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Give a short succinct context (1-2 sentences) to situate this chunk within the overall document for improving search retrieval. Answer only with the context and nothing else."""


# --- Pydantic schemas ---

class ContextualChunk(BaseModel):
    chunk: Chunk
    context: str                    # Claude-generated context snippet
    enriched_text: str              # context + original chunk text


class ContextualRetrieverConfig(BaseModel):
    top_k: int = 3
    min_score: float = -999.0    # allow negative scores, BM25Okapi can produce them on small corpora
    context_model: str = CONTEXT_MODEL
    crawl_delay: float = 0.3

# --- Context generation ---

def generate_chunk_context(
    document: str,
    chunk_text: str,
    model: str = CONTEXT_MODEL,
) -> str:
    """
    Use Claude Haiku to generate a short context snippet for a chunk.
    The snippet situates the chunk within the full document for better retrieval.
    """
    response = client.messages.create(
        model=model,
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": CONTEXT_PROMPT.format(
                    document=document[:8000],   # cap document for cost control
                    chunk=chunk_text,
                )
            }
        ]
    )
    return response.content[0].text.strip()


# --- Contextual BM25 Retriever ---

class ContextualBM25Retriever:
    """
    BM25 retriever with Claude-generated context prepended to each chunk.

    Preprocessing (one-time, at index build time):
    1. Chunk the document
    2. For each chunk, call Claude Haiku to generate a context snippet
    3. Prepend context to chunk text
    4. Build BM25 index over enriched chunks

    At query time: same BM25 retrieval as plain BM25Retriever.

    Advantage over plain BM25: chunks are enriched with document-level
    context (entity name, time period, document type) so retrieval
    works even when the original chunk lacks those identifiers.
    """

    def __init__(
        self,
        document: str,
        config: Optional[ContextualRetrieverConfig] = None,
        verbose: bool = True,
    ):
        self.config = config or ContextualRetrieverConfig()
        self.document = document
        self.verbose = verbose

        # Step 1: chunk the document
        self.raw_chunks = chunk_text(document)
        if not self.raw_chunks:
            raise ValueError("Document produced no chunks.")

        # Step 2 + 3: generate context and build enriched chunks
        self.contextual_chunks = self._build_contextual_chunks()

        # Step 4: build BM25 index over enriched text
        tokenized = [
            tokenize_for_bm25(cc.enriched_text)
            for cc in self.contextual_chunks
        ]
        self.bm25 = BM25Okapi(tokenized)

        if self.verbose:
            print(f"  [contextual] Indexed {len(self.contextual_chunks)} enriched chunks")

    def _build_contextual_chunks(self) -> list[ContextualChunk]:
        """Generate context for each chunk using Claude Haiku."""
        contextual_chunks = []

        for i, chunk in enumerate(self.raw_chunks):
            if self.verbose:
                print(f"  [contextual] Generating context for chunk {i+1}/{len(self.raw_chunks)}...")

            try:
                context = generate_chunk_context(
                    document=self.document,
                    chunk_text=chunk.text,
                    model=self.config.context_model,
                )
            except Exception as e:
                # Fall back to empty context on failure
                context = ""
                if self.verbose:
                    print(f"  [contextual] Context generation failed for chunk {i+1}: {e}")

            enriched_text = f"{context} {chunk.text}" if context else chunk.text

            contextual_chunks.append(ContextualChunk(
                chunk=chunk,
                context=context,
                enriched_text=enriched_text,
            ))

            # Rate limit buffer between Haiku calls
            if i < len(self.raw_chunks) - 1:
                time.sleep(self.config.crawl_delay)

        return contextual_chunks

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Retrieve top-k chunks most relevant to the query."""
        query_tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for rank, (chunk_idx, score) in enumerate(ranked[:self.config.top_k]):
            if score >= self.config.min_score:
                results.append(RetrievalResult(
                    chunk=self.contextual_chunks[chunk_idx].chunk,
                    score=round(float(score), 4),
                    rank=rank + 1,
                ))

        return results

    def get_contextual_chunks(self) -> list[ContextualChunk]:
        """Return all enriched chunks for inspection."""
        return self.contextual_chunks

    @property
    def num_chunks(self) -> int:
        return len(self.contextual_chunks)


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

    print("Building contextual BM25 index...")
    retriever = ContextualBM25Retriever(document)

    print(f"\nContextual chunks built: {retriever.num_chunks}")
    print("\nGenerated contexts:")
    for cc in retriever.get_contextual_chunks():
        print(f"\n  Chunk {cc.chunk.chunk_id}:")
        print(f"    Context: {cc.context}")
        print(f"    Original: {cc.chunk.text[:80]}...")

    queries = [
        "What was Apple's total revenue?",
        "What was JPMorgan net income?",
        "What is the expense ratio of the Vanguard fund?",
    ]

    print("\n--- Retrieval results ---")
    for query in queries:
        print(f"\nQuery: {query}")
        query_tokens = tokenize_for_bm25(query)
        raw_scores = retriever.bm25.get_scores(query_tokens)
        print(f"  Raw BM25 scores: {[round(float(s), 3) for s in raw_scores]}")
        results = retriever.retrieve(query)
        if not results:
            print("  No results returned")
        for r in results:
            print(f"  Rank {r.rank} (score {r.score:.2f}): {r.chunk.text[:100]}...")


    