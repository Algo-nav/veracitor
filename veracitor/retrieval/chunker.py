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
    max_words: int = 250
    overlap_words: int = 30
    min_words: int = 20
    respect_paragraphs: bool = True     # split on paragraphs first


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)      # collapse 3+ newlines to 2
    text = re.sub(r'[ \t]+', ' ', text)          # collapse spaces/tabs
    return text.strip()


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split text on paragraph boundaries (double newlines).
    Falls back to single newlines if no double newlines found.
    """
    paragraphs = re.split(r'\n\n+', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # If no paragraph breaks found, try single newlines
    if len(paragraphs) <= 1:
        paragraphs = re.split(r'\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences on period/question/exclamation boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def pack_sentences_into_chunks(
    sentences: list[str],
    config: ChunkingConfig,
    start_char: int = 0,
    chunk_id_start: int = 0,
) -> list[Chunk]:
    """
    Pack sentences into chunks up to max_words with overlap.
    Used when a paragraph exceeds max_words.
    """
    chunks = []
    current_sentences = []
    current_word_count = 0
    chunk_id = chunk_id_start
    char_pos = start_char

    for sentence in sentences:
        sentence_words = len(sentence.split())

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

            # Overlap: keep last 2 sentences
            overlap = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences[-1:]
            overlap_text = " ".join(overlap)
            overlap_words = overlap_text.split()[-config.overlap_words:]
            current_sentences = [" ".join(overlap_words)] if overlap_words else []
            current_word_count = len(current_sentences[0].split()) if current_sentences else 0
            char_pos += len(chunk_text_str)

        current_sentences.append(sentence)
        current_word_count += sentence_words

    # Final chunk
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


def chunk_text(
    text: str,
    config: Optional[ChunkingConfig] = None,
) -> list[Chunk]:
    """
    Split text into overlapping chunks for retrieval.

    Strategy:
    1. Clean and normalize text
    2. Split on paragraph boundaries first (respects document structure)
    3. Paragraphs within max_words: keep as single chunk
    4. Paragraphs exceeding max_words: sub-chunk by sentences with overlap

    Finance documents have natural paragraph structure (each paragraph
    discusses one topic). Respecting paragraphs produces more semantically
    coherent chunks than arbitrary sentence packing.
    """
    if config is None:
        config = ChunkingConfig()

    text = clean_text(text)
    chunks = []
    chunk_id = 0
    char_pos = 0

    if config.respect_paragraphs:
        paragraphs = split_into_paragraphs(text)
    else:
        paragraphs = [text]

    for para in paragraphs:
        para_words = len(para.split())

        if para_words < config.min_words:
            # Too short to be a useful chunk, skip
            char_pos += len(para)
            continue

        if para_words <= config.max_words:
            # Paragraph fits in one chunk
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=para,
                start_char=char_pos,
                end_char=char_pos + len(para),
                word_count=para_words,
            ))
            chunk_id += 1
            char_pos += len(para)

        else:
            # Paragraph too long: sub-chunk by sentences
            sentences = split_into_sentences(para)
            sub_chunks = pack_sentences_into_chunks(
                sentences,
                config,
                start_char=char_pos,
                chunk_id_start=chunk_id,
            )
            chunks.extend(sub_chunks)
            chunk_id += len(sub_chunks)
            char_pos += len(para)

    return chunks


# --- Test block ---

if __name__ == "__main__":
    sample_text = """Apple Inc. reported total net sales of $391 billion for fiscal year 2025, representing a 4% increase compared to the prior year. iPhone revenue accounted for $201 billion, or approximately 51% of total net sales.

    Services revenue reached $96 billion, up 12% year over year. The company reported net income of $94 billion for the fiscal year.

    Operating cash flow was $118 billion, with free cash flow of $108 billion. Apple returned $110 billion to shareholders through dividends and buybacks. The board approved a new $90 billion share repurchase authorization.

    International sales accounted for 58% of total revenue. The company ended the fiscal year with $167 billion in cash and investments. Mac revenue was $16 billion, up 8% from the prior year. iPad revenue was $7 billion, down 6% year over year. Wearables, Home and Accessories revenue was $9 billion."""

    chunks = chunk_text(sample_text)
    print(f"Generated {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"\nChunk {chunk.chunk_id} ({chunk.word_count} words):")
        print(f"  {chunk.text[:120]}...")