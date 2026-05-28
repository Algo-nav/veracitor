import os
import time
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# CAG context budget: how much of the document to include.
# Claude Sonnet context window is 200K tokens (~800K chars).
# We cap at 40,000 chars to keep costs bounded and latency acceptable.
CAG_CONTEXT_BUDGET = 40000


class CAGConfig(BaseModel):
    context_budget: int = CAG_CONTEXT_BUDGET
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 500


class CAGResult(BaseModel):
    answer: str                     # generated answer from full context
    context_used: str               # the document portion passed to LLM
    context_length: int             # chars of document used
    truncated: bool                 # whether document was truncated
    model: str
    error: Optional[str] = None


SYSTEM_PROMPT = """You are a precise finance research assistant.
Answer the question based ONLY on the provided document context.
Be specific and cite exact numbers and dates from the document.
If the answer is not in the document, say so explicitly."""


def generate_cag_answer(
    question: str,
    document: str,
    config: Optional[CAGConfig] = None,
) -> CAGResult:
    """
    Cache-Augmented Generation: pass the full document as context,
    generate an answer without a retrieval step.

    Advantages over RAG:
    - No retrieval miss: LLM sees everything
    - Better for cross-chunk reasoning
    - Simpler pipeline, fewer failure modes

    Disadvantages:
    - Higher token cost per query
    - Slower for very long documents
    - Not suitable when document > context window
    """
    if config is None:
        config = CAGConfig()

    truncated = len(document) > config.context_budget
    context = document[:config.context_budget]

    try:
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Document:\n{context}\n\n"
                        f"Question: {question}"
                    )
                }
            ]
        )

        answer = response.content[0].text.strip()

        return CAGResult(
            answer=answer,
            context_used=context,
            context_length=len(context),
            truncated=truncated,
            model=config.model,
        )

    except Exception as e:
        return CAGResult(
            answer="",
            context_used=context,
            context_length=len(context),
            truncated=truncated,
            model=config.model,
            error=str(e),
        )


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
    International sales accounted for 58% of total revenue.
    Mac revenue was $16 billion, up 8% from the prior year.
    iPad revenue was $7 billion, down 6% year over year.
    """

    questions = [
        "What was Apple's total revenue in fiscal 2025?",
        "How much did services contribute to revenue?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        result = generate_cag_answer(question, document)

        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(f"  Answer: {result.answer}")
            print(f"  Context length: {result.context_length} chars")
            print(f"  Truncated: {result.truncated}")