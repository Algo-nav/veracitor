import os
from typing import Optional
from pydantic import BaseModel

_rerank_pipeline = None


def get_rerank_pipeline():
    """
    Load the cross-encoder reranker once and cache it.
    Uses ms-marco-MiniLM-L-6-v2: fast, accurate, ~80MB.
    """
    global _rerank_pipeline
    if _rerank_pipeline is None:
        from sentence_transformers import CrossEncoder
        print("  [reranker] Loading cross-encoder model (first run only)...")
        _rerank_pipeline = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
        )
        print("  [reranker] Model loaded.")
    return _rerank_pipeline


class RerankResult(BaseModel):
    chunk_id: int
    text: str
    original_rank: int
    rerank_score: float
    final_rank: int


def rerank(
    query: str,
    retrieval_results: list,       # list of RetrievalResult from BM25
    top_k: int = 3,
) -> list[RerankResult]:
    """
    Rerank BM25 retrieval results using a cross-encoder.

    Cross-encoders score query-chunk pairs jointly, capturing
    semantic relevance that BM25 lexical matching misses.

    Flow:
    1. BM25 retrieves top-10 candidate chunks (coarse filter)
    2. Cross-encoder scores each candidate against the query (fine filter)
    3. Return top-k by rerank score

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Trained on MS MARCO passage ranking dataset
    - ~80MB, fast on CPU (~10ms per pair)
    - Strong performance on factual Q&A retrieval
    """
    if not retrieval_results:
        return []

    model = get_rerank_pipeline()

    # Build query-chunk pairs for scoring
    pairs = [(query, r.chunk.text) for r in retrieval_results]
    scores = model.predict(pairs)

    # Combine with original results
    scored = []
    for i, (result, score) in enumerate(zip(retrieval_results, scores)):
        scored.append({
            "chunk_id": result.chunk.chunk_id,
            "text": result.chunk.text,
            "original_rank": result.rank,
            "rerank_score": float(score),
        })

    # Sort by rerank score descending
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    return [
        RerankResult(
            chunk_id=item["chunk_id"],
            text=item["text"],
            original_rank=item["original_rank"],
            rerank_score=round(item["rerank_score"], 4),
            final_rank=rank + 1,
        )
        for rank, item in enumerate(scored[:top_k])
    ]


# --- Test block ---

if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))

    from veracitor.retrieval.bm25_retriever import BM25Retriever, BM25RetrieverConfig

    with open("data/corpus/earnings/AAPL_Q1_2025.json") as f:
        data = json.load(f)

    text = data["text"]
    print(f"Document length: {len(text)} chars")

    # Retrieve top-10 with BM25 (wider net for reranker)
    retriever = BM25Retriever(
        text,
        config=BM25RetrieverConfig(top_k=10, min_score=-999.0)
    )
    print(f"Chunks: {retriever.num_chunks}")

    queries = [
        "What was Apple total revenue?",
        "What was services revenue?",
        "What was iPhone revenue?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        bm25_results = retriever.retrieve(query)

        print(f"  BM25 top-3:")
        for r in bm25_results[:3]:
            print(f"    Rank {r.rank} (score {r.score:.2f}): {r.chunk.text[:100]}...")

        reranked = rerank(query, bm25_results, top_k=3)
        print(f"  Reranked top-3:")
        for r in reranked:
            print(f"    Rank {r.final_rank} (rerank {r.rerank_score:.2f}, "
                  f"was BM25 rank {r.original_rank}): {r.text[:100]}...")