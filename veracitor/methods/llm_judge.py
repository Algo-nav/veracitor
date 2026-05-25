import os
import json
import time
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --- Pydantic schemas ---

class LLMJudgeInput(BaseModel):
    question: str
    answer: str
    source: str
    label: Optional[str] = None


class DimensionScore(BaseModel):
    score: float        # 0.0 to 1.0
    reasoning: str      # one sentence


class LLMJudgeOutput(BaseModel):
    faithfulness: DimensionScore
    factuality: DimensionScore
    relevance: DimensionScore
    judge_score: float      # weighted combination
    flagged: bool
    threshold: float
    method: str = "llm_judge"
    error: Optional[str] = None


# --- Prompt ---

SYSTEM_PROMPT = """You are a hallucination detection judge for finance RAG systems.

You evaluate whether a generated answer is supported by the provided source text.

You score three dimensions independently. Return ONLY a JSON object with no preamble,
no explanation, no markdown fences. Raw JSON only.

Scoring rubric:

FAITHFULNESS (is every claim in the answer directly supported by the source?):
- 1.0: every claim is explicitly supported by the source text
- 0.5: most claims are supported but one claim is vague or weakly implied
- 0.0: one or more claims contradict or go beyond what the source states

FACTUALITY (does the answer contradict well-known real-world facts?):
- 1.0: no real-world facts are contradicted
- 0.5: minor factual imprecision that does not materially mislead
- 0.0: a clearly incorrect real-world fact (wrong CEO, wrong company, wrong date)

RELEVANCE (does the answer address the question asked?):
- 1.0: answer directly and completely addresses the question
- 0.5: answer partially addresses the question but drifts or is incomplete
- 0.0: answer does not address the question at all

Critical rules:
- Be strict on faithfulness. Any specific number, percentage, or claim not in the source
  must be penalized.
- Do not give benefit of the doubt. If a claim cannot be verified from the source, score low.
- Scores must be exactly 0.0, 0.25, 0.5, 0.75, or 1.0.

Return this exact JSON structure:
{
  "faithfulness": {"score": 0.0, "reasoning": "one sentence"},
  "factuality": {"score": 0.0, "reasoning": "one sentence"},
  "relevance": {"score": 0.0, "reasoning": "one sentence"}
}"""

USER_TEMPLATE = """Question: {question}

Source text:
{source}

Generated answer:
{answer}

Grade the answer on faithfulness, factuality, and relevance."""


# --- Core scoring ---

def compute_llm_judge(
    question: str,
    answer: str,
    source: str,
    threshold: float = 0.60,
    retries: int = 2,
) -> LLMJudgeOutput:
    """
    Grade an answer using Claude Sonnet as judge.

    judge_score = 0.5 * faithfulness + 0.25 * factuality + 0.25 * relevance

    Faithfulness weighted highest because it is the primary detection target
    for RAG hallucination. Factuality and relevance are secondary signals.

    Flagged if judge_score < threshold (0.60).
    """
    last_error = None

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": USER_TEMPLATE.format(
                            question=question,
                            answer=answer,
                            source=source[:3000],   # cap source length for cost control
                        )
                    }
                ]
            )

            raw = response.content[0].text.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            data = json.loads(raw)

            faith = DimensionScore(**data["faithfulness"])
            fact = DimensionScore(**data["factuality"])
            rel = DimensionScore(**data["relevance"])

            judge_score = 0.5 * faith.score + 0.25 * fact.score + 0.25 * rel.score

            return LLMJudgeOutput(
                faithfulness=faith,
                factuality=fact,
                relevance=rel,
                judge_score=round(judge_score, 4),
                flagged=judge_score < threshold,
                threshold=threshold,
            )

        except Exception as e:
            last_error = e
            time.sleep(2)

    return LLMJudgeOutput(
        faithfulness=DimensionScore(score=0.0, reasoning="error"),
        factuality=DimensionScore(score=0.0, reasoning="error"),
        relevance=DimensionScore(score=0.0, reasoning="error"),
        judge_score=0.0,
        flagged=True,
        threshold=threshold,
        error=str(last_error),
    )


def score_pair(input_data: LLMJudgeInput) -> LLMJudgeOutput:
    """Public entry point. Scores a single Q&A pair."""
    return compute_llm_judge(
        question=input_data.question,
        answer=input_data.answer,
        source=input_data.source,
    )


# --- Test block ---

if __name__ == "__main__":
    test_cases = [
        {
            "label": "clean",
            "question": "What was Apple's revenue in fiscal 2025?",
            "answer": "Apple reported total net sales of $391 billion in fiscal year 2025.",
            "source": "Apple Inc. reported total net sales of $391 billion for fiscal year 2025, "
                      "representing a 4% increase compared to the prior year.",
        },
        {
            "label": "faithfulness",
            "question": "What was Apple's revenue in fiscal 2025?",
            "answer": "Apple reported total net sales of $450 billion in fiscal year 2025, "
                      "driven by strong iPhone sales in emerging markets.",
            "source": "Apple Inc. reported total net sales of $391 billion for fiscal year 2025, "
                      "representing a 4% increase compared to the prior year.",
        },
        {
            "label": "relevance",
            "question": "What was Apple's revenue in fiscal 2025?",
            "answer": "Apple was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne "
                      "in Cupertino, California.",
            "source": "Apple Inc. reported total net sales of $391 billion for fiscal year 2025, "
                      "representing a 4% increase compared to the prior year.",
        },
    ]

    for tc in test_cases:
        print(f"\nLabel: {tc['label']}")
        inp = LLMJudgeInput(**tc)
        result = score_pair(inp)

        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(f"  Faithfulness: {result.faithfulness.score} - {result.faithfulness.reasoning}")
            print(f"  Factuality:   {result.factuality.score} - {result.factuality.reasoning}")
            print(f"  Relevance:    {result.relevance.score} - {result.relevance.reasoning}")
            print(f"  Judge score:  {result.judge_score}")
            print(f"  Flagged:      {result.flagged}")