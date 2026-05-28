import re
from typing import Optional
from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: int
    text: str
    start_char: int
    end_char: int
    word_count: int


class ChunkingConfig(BaseModel):
    max_words: int = 150        # max words per chunk
    overlap_words: int = 25     # overlap between consecutive chunks
    min_words: int = 20         # minimum words to keep a chunk


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences on period/question/exclamation boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    config: Optional[ChunkingConfig] = None,
) -> list[Chunk]:
    """
    Split text into overlapping chunks for retrieval.

    Strategy:
    1. Clean and normalize text
    2. Split into sentences
    3. Pack sentences into chunks up to max_words
    4. Add overlap from previous chunk to maintain context
    """
    if config is None:
        config = ChunkingConfig()

    text = clean_text(text)
    sentences = split_into_sentences(text)

    chunks = []
    current_sentences = []
    current_word_count = 0
    chunk_id = 0
    char_pos = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # If adding this sentence exceeds max, save current chunk and start new
        if current_word_count + sentence_words > config.max_words and current_sentences:
            chunk_text_str = " ".join(current_sentences)
            if len(chunk_text_str.split()) >= config.min_words:
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text_str,
                    start_char=char_pos,
                    end_char=char_pos + len(chunk_text_str),
                    word_count=len(chunk_text_str.split()),
                ))
                chunk_id += 1

            # Keep last N words as overlap for next chunk
            overlap_text = " ".join(current_sentences[-2:]) if len(current_sentences) >= 2 else ""
            overlap_words = overlap_text.split()[-config.overlap_words:]
            current_sentences = [" ".join(overlap_words)] if overlap_words else []
            current_word_count = len(current_sentences[0].split()) if current_sentences else 0
            char_pos += len(chunk_text_str)

        current_sentences.append(sentence)
        current_word_count += sentence_words

    # Save final chunk
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        if len(chunk_text_str.split()) >= config.min_words:
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text_str,
                start_char=char_pos,
                end_char=char_pos + len(chunk_text_str),
                word_count=len(chunk_text_str.split()),
            ))

    return chunks


# --- Test block ---

if __name__ == "__main__":
    sample_text = """
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
    """

    chunks = chunk_text(sample_text)
    print(f"Generated {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"\nChunk {chunk.chunk_id} ({chunk.word_count} words):")
        print(f"  {chunk.text[:100]}...")