import os
import json
import time
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# --- CSS ---

CUSTOM_CSS = """
body { background-color: #0A0A0F !important; }
.gradio-container { background-color: #0A0A0F !important; }
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --bg-primary: #0A0A0F;
    --bg-secondary: #111118;
    --bg-card: #16161F;
    --bg-elevated: #1C1C28;
    --border: #2A2A3A;
    --border-subtle: #1E1E2A;
    --text-primary: #E8E8F0;
    --text-secondary: #8888AA;
    --text-muted: #555570;
    --accent-blue: #4A9EFF;
    --accent-green: #00CC88;
    --accent-amber: #FFB800;
    --accent-red: #FF4455;
    --pass-color: #00CC88;
    --review-color: #FFB800;
    --fail-color: #FF4455;
    --font-display: 'Playfair Display', serif;
    --font-body: 'IBM Plex Sans', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
}

body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}

/* Header */
.veracitor-header {
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}

.veracitor-title {
    font-family: var(--font-display);
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0;
}

.veracitor-subtitle {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-secondary);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

.veracitor-tagline {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-top: 0.8rem;
    max-width: 600px;
    line-height: 1.6;
}

/* Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Verdict banner */
.verdict-pass {
    background: rgba(0, 204, 136, 0.08);
    border: 1px solid rgba(0, 204, 136, 0.3);
    border-radius: 8px;
    padding: 1.5rem;
}

.verdict-review {
    background: rgba(255, 184, 0, 0.08);
    border: 1px solid rgba(255, 184, 0, 0.3);
    border-radius: 8px;
    padding: 1.5rem;
}

.verdict-fail {
    background: rgba(255, 68, 85, 0.08);
    border: 1px solid rgba(255, 68, 85, 0.3);
    border-radius: 8px;
    padding: 1.5rem;
}

.verdict-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.verdict-confidence {
    font-family: var(--font-display);
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
}

.verdict-explanation {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-top: 0.75rem;
    line-height: 1.6;
}

.verdict-score {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
}

/* Method score bars */
.method-bar-container {
    margin-bottom: 0.75rem;
}

.method-bar-label {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.method-bar-track {
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}

.method-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
}

/* Flagged claims */
.flagged-claim {
    background: rgba(255, 68, 85, 0.06);
    border-left: 3px solid var(--accent-red);
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    border-radius: 0 4px 4px 0;
    font-size: 0.875rem;
    line-height: 1.5;
    color: var(--text-primary);
}

/* Retrieved context */
.context-chunk {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    font-size: 0.85rem;
    line-height: 1.6;
    color: var(--text-secondary);
}

.context-rank {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}

/* Progress */
.progress-line {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--accent-blue);
    padding: 0.3rem 0;
    border-bottom: 1px solid var(--border-subtle);
}

/* Tabs */
.tab-nav button {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}

.tab-nav button.selected {
    color: var(--text-primary) !important;
    border-bottom-color: var(--accent-blue) !important;
}

/* Inputs - Gradio 6 dark mode force */
.gr-textbox textarea, .gr-textbox input,
input, textarea {
    background: #111118 !important;
    border: 1px solid #2A2A3A !important;
    color: #E8E8F0 !important;
    font-family: var(--font-body) !important;
    font-size: 0.875rem !important;
}

.gr-textbox textarea:focus, .gr-textbox input:focus,
input:focus, textarea:focus {
    border-color: var(--accent-blue) !important;
    outline: none !important;
}

[data-testid="textbox"] textarea,
[data-testid="textbox"] input {
    background: #111118 !important;
    color: #E8E8F0 !important;
}

label span {
    color: #8888AA !important;
}

/* Buttons */
.gr-button-primary {
    background: var(--accent-blue) !important;
    border: none !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* Dropdown */
.gr-dropdown select {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}

/* Section labels */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border-subtle);
}

/* Comparison */
.comparison-winner {
    border-color: rgba(0, 204, 136, 0.4) !important;
}

.comparison-loser {
    border-color: rgba(255, 68, 85, 0.3) !important;
}

/* Gallery */
.gallery-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    cursor: pointer;
    transition: border-color 0.2s ease;
}

.gallery-card:hover {
    border-color: var(--accent-blue);
}

.gallery-type-badge {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.2rem 0.5rem;
    border-radius: 3px;
    background: var(--bg-elevated);
    color: var(--text-muted);
    display: inline-block;
    margin-bottom: 0.5rem;
}

.gallery-hallucination-badge {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.5rem;
    border-radius: 3px;
    display: inline-block;
    margin-left: 0.4rem;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* Fix dropdown border */
[data-testid="dropdown"] {
    background: #111118 !important;
    border: 1px solid #2A2A3A !important;
}

/* Fix button text color */
button.primary {
    color: #ffffff !important;
    font-family: var(--font-mono) !important;
    letter-spacing: 0.1em !important;
}

/* Fix all button text */
button {
    font-family: var(--font-mono) !important;
}

/* Remove white border on focused inputs */
input:focus, textarea:focus, select:focus {
    outline: none !important;
    box-shadow: none !important;
    border-color: #4A9EFF !important;
}
/* Remove white outline on focused dropdown */
[data-testid="dropdown"] input:focus,
[data-testid="dropdown"] input,
.gr-dropdown input,
select {
    outline: none !important;
    box-shadow: none !important;
    border: 1px solid #2A2A3A !important;
}
/* Radio buttons */
input[type="radio"] {
    accent-color: #4A9EFF !important;
    width: 16px !important;
    height: 16px !important;
    cursor: pointer !important;
}

fieldset label, .gr-radio label {
    background: #16161F !important;
    border: 1px solid #2A2A3A !important;
    border-radius: 6px !important;
    padding: 0.5rem 0.75rem !important;
    color: #8888AA !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    cursor: pointer !important;
    transition: border-color 0.15s ease !important;
}

fieldset label:has(input[type="radio"]:checked),
.gr-radio label:has(input[type="radio"]:checked) {
    border-color: #4A9EFF !important;
    color: #E8E8F0 !important;
    background: #1C1C28 !important;
}

fieldset label:hover {
    border-color: #4A9EFF88 !important;
}

/* Dropdown - kill focus ring */
.svelte-1ouin19,
.wrap.svelte-1ouin19 {
    background: #111118 !important;
    border: 1px solid #2A2A3A !important;
}

.svelte-1ouin19:focus-within {
    border-color: #4A9EFF !important;
    box-shadow: none !important;
    outline: none !important;
}

.options.svelte-1ouin19 {
    background: #16161F !important;
    border: 1px solid #2A2A3A !important;
}

.item.svelte-1ouin19:hover {
    background: #1C1C28 !important;
}
"""

# --- HTML helpers ---

def render_verdict(confidence: str, score: float, explanation: str) -> str:
    css_class = f"verdict-{confidence.lower()}"
    color_map = {"PASS": "#00CC88", "REVIEW": "#FFB800", "FAIL": "#FF4455"}
    color = color_map.get(confidence, "#8888AA")
    return f"""
    <div class="{css_class}">
        <div class="verdict-label" style="color:{color}">Verdict</div>
        <div class="verdict-confidence" style="color:{color}">{confidence}</div>
        <div class="verdict-explanation">{explanation}</div>
        <div class="verdict-score">Overall score: {score:.3f}</div>
    </div>
    """


def render_method_bars(method_breakdown: dict) -> str:
    if not method_breakdown:
        return ""

    color_map = {
        "citation_alignment": "#4A9EFF",
        "nli_entailment": "#00CC88",
        "llm_judge": "#FFB800",
        "span_coverage": "#AA88FF",
        "self_consistency": "#FF8844",
    }

    label_map = {
        "citation_alignment": "Citation Alignment",
        "nli_entailment": "NLI Entailment",
        "llm_judge": "LLM Judge",
        "span_coverage": "Span Coverage",
        "self_consistency": "Self-Consistency",
    }

    bars = ""
    for method, result in method_breakdown.items():
        score = result.score
        color = color_map.get(method, "#8888AA")
        label = label_map.get(method, method)
        flag_str = " — FLAGGED" if result.flagged else ""
        pct = int(score * 100)
        bars += f"""
        <div class="method-bar-container">
            <div class="method-bar-label">
                <span>{label}{flag_str}</span>
                <span>{score:.3f}</span>
            </div>
            <div class="method-bar-track">
                <div class="method-bar-fill"
                     style="width:{pct}%; background:{color}">
                </div>
            </div>
        </div>
        """
    return bars


def render_flagged_claims(claims: list[str]) -> str:
    if not claims:
        return '<p style="color:var(--text-muted); font-size:0.85rem;">No specific claims flagged.</p>'
    html = ""
    for claim in claims:
        html += f'<div class="flagged-claim">{claim}</div>'
    return html


def render_context_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return '<p style="color:var(--text-muted); font-size:0.85rem;">No context retrieved.</p>'
    html = ""
    for chunk in chunks:
        html += f"""
        <div class="context-chunk">
            <div class="context-rank">Rank {chunk['rank']} &nbsp;|&nbsp; Score {chunk['score']:.3f}</div>
            {chunk['text']}
        </div>
        """
    return html


def build_explanation(result) -> str:
    """Build plain-English explanation from result."""
    if result.confidence.value == "PASS":
        return "All claims in the answer are supported by the retrieved source text."
    elif result.confidence.value == "REVIEW":
        flagged_methods = [m for m, r in result.method_breakdown.items() if r.flagged]
        if flagged_methods:
            return f"Some claims may not be fully supported. Flagged by: {', '.join(flagged_methods)}."
        return "Answer may contain claims that are weakly supported by the source."
    else:
        flagged = result.flagged_claims
        if flagged:
            return f"{len(flagged)} claim(s) appear unsupported or contradicted by the source."
        return "Answer contains claims that contradict or go beyond the source text."


# --- Pre-loaded gallery examples ---

GALLERY_EXAMPLES = [
    {
        "id": "ex1",
        "title": "Apple Revenue - Clean",
        "doc_type": "10-K",
        "hallucination_type": "clean",
        "document": "Apple Inc. reported total net sales of $391 billion for fiscal year 2025, representing a 4% increase compared to the prior year. iPhone revenue accounted for $201 billion, or approximately 51% of total net sales. Services revenue reached $96 billion, up 12% year over year.",
        "question": "What was Apple's total revenue in fiscal 2025?",
        "answer_a": "Apple reported total net sales of $391 billion in fiscal year 2025, up 4% year over year.",
        "answer_b": "",
        "expected_verdict": "PASS",
        "note": "Answer accurately reflects source. All methods should agree.",
    },
    {
        "id": "ex2",
        "title": "Apple Revenue - Faithfulness Violation",
        "doc_type": "10-K",
        "hallucination_type": "faithfulness",
        "document": "Apple Inc. reported total net sales of $391 billion for fiscal year 2025, representing a 4% increase compared to the prior year. iPhone revenue accounted for $201 billion, or approximately 51% of total net sales.",
        "question": "What was Apple's total revenue in fiscal 2025?",
        "answer_a": "Apple reported total net sales of $391 billion in fiscal year 2025, up 4% year over year.",
        "answer_b": "Apple reported total net sales of $450 billion in fiscal year 2025, driven by strong iPhone sales in emerging markets.",
        "expected_verdict": "FAIL",
        "note": "Answer B contains $450B vs $391B. NLI catches this; citation alignment may miss it.",
    },
    {
        "id": "ex3",
        "title": "JPMorgan Net Income - Subtle Drift",
        "doc_type": "Earnings",
        "hallucination_type": "faithfulness",
        "document": "JPMorgan Chase reported net income of $58.5 billion for fiscal year 2025, compared to $49.6 billion in fiscal year 2024. Return on equity was 17% for the full year 2025. Total assets as of December 31, 2025 were $4.0 trillion.",
        "question": "What was JPMorgan's net income in fiscal 2025?",
        "answer_a": "JPMorgan Chase reported net income of $58.5 billion in fiscal year 2025, up from $49.6 billion the prior year.",
        "answer_b": "JPMorgan Chase reported net income of $62 billion in fiscal year 2025, with return on equity improving to 19%.",
        "expected_verdict": "FAIL",
        "note": "Answer B changes net income and ROE. Good test for subtle number drift.",
    },
    {
        "id": "ex4",
        "title": "Vanguard Fund - Relevance Violation",
        "doc_type": "Prospectus",
        "hallucination_type": "relevance",
        "document": "The Vanguard 500 Index Fund seeks to track the performance of the S&P 500 Index. The fund has an expense ratio of 0.04% and requires a minimum investment of $3,000. Total net assets were $1.1 trillion as of the most recent quarter.",
        "question": "What is the expense ratio of the Vanguard 500 Index Fund?",
        "answer_a": "The Vanguard 500 Index Fund has an expense ratio of 0.04%.",
        "answer_b": "Vanguard was founded in 1975 by John Bogle and is headquartered in Malvern, Pennsylvania.",
        "expected_verdict": "FAIL",
        "note": "Answer B is factually accurate but does not address the question. Tests relevance detection.",
    },
    {
        "id": "ex5",
        "title": "Apple Services - Factuality Violation",
        "doc_type": "Earnings",
        "hallucination_type": "factuality",
        "document": "Our services revenue reached an all-time high of $26.3 billion, up 14% year over year. We now have over 1 billion paid subscriptions across our services portfolio.",
        "question": "What was Apple's services revenue?",
        "answer_a": "Apple's services revenue reached an all-time high of $26.3 billion, up 14% year over year.",
        "answer_b": "Apple's services revenue was $26.3 billion. Apple was founded in 1976 by Bill Gates and Steve Jobs.",
        "expected_verdict": "FAIL",
        "note": "Answer B introduces a factuality violation: Bill Gates did not found Apple.",
    },
    {
        "id": "ex6",
        "title": "Method Divergence - NLI vs Citation",
        "doc_type": "10-K",
        "hallucination_type": "faithfulness",
        "document": "Goldman Sachs reported net revenues of $53.5 billion for fiscal year 2025, up 11% from the prior year. Investment banking revenues were $8.7 billion, up 24% year over year. Global Markets revenues were $22.0 billion for the year.",
        "question": "What were Goldman Sachs net revenues in fiscal 2025?",
        "answer_a": "Goldman Sachs reported net revenues of $53.5 billion in fiscal 2025, an 11% increase year over year.",
        "answer_b": "Goldman Sachs reported net revenues of $53.5 billion in fiscal 2025, driven by a 24% increase in equity underwriting.",
        "expected_verdict": "FAIL",
        "note": "Answer B has correct revenue but attributes it to equity underwriting (not in source). Citation alignment may miss this; NLI should catch it. Classic method divergence example.",
    },
]


# --- Core verification pipeline ---

def run_verification(
    document: str,
    question: str,
    answer: str,
    retrieval_mode: str,
    detection_methods: list[str],
    answer_b: str = "",
    top_k: int = 5,
    progress=gr.Progress(),
) -> tuple:
    """
    Full verification pipeline:
    1. Chunk and index document
    2. Retrieve relevant context
    3. Run detection methods
    4. Return structured results
    """
    from veracitor import check_answer
    from veracitor.retrieval.bm25_retriever import BM25Retriever, BM25RetrieverConfig
    from veracitor.retrieval.contextual_retriever import ContextualBM25Retriever, ContextualRetrieverConfig
    from veracitor.retrieval.cag_retriever import generate_cag_answer, CAGConfig
    from veracitor.retrieval.reranker import rerank

    if not document.strip():
        return "Please paste a document.", "", "", "", "", ""
    if not question.strip():
        return "Please enter a question.", "", "", "", "", ""
    if not answer.strip():
        return "Please enter an answer to verify.", "", "", "", "", ""

    retrieved_chunks = []
    context_html = ""

    try:
        if retrieval_mode == "Full Document (CAG)":
            progress(0.2, desc="Loading full document context...")
            cag_result = generate_cag_answer(question, document)
            sources = [cag_result.context_used]
            context_html = f"""
            <div class="section-label">Full Document Context (CAG)</div>
            <div class="context-chunk">
                <div class="context-rank">
                    {cag_result.context_length:,} chars &nbsp;|&nbsp;
                    Truncated: {cag_result.truncated}
                </div>
                {document[:500]}{'...' if len(document) > 500 else ''}
            </div>
            """

        elif retrieval_mode == "Smart Keyword Search (Contextual BM25)":
            progress(0.1, desc="Chunking document...")
            retriever = ContextualBM25Retriever(
                document,
                config=ContextualRetrieverConfig(top_k=top_k * 2),
                verbose=False,
            )
            progress(0.6, desc="Retrieving and reranking...")
            bm25_results = retriever.retrieve(question)
            reranked = rerank(question, bm25_results, top_k=top_k)
            sources = [r.text for r in reranked]
            retrieved_chunks = [
                {"rank": r.final_rank, "score": r.rerank_score, "text": r.text}
                for r in reranked
            ]
            context_html = f'<div class="section-label">Retrieved Context (Contextual BM25 + Reranking)</div>'
            context_html += render_context_chunks(retrieved_chunks)

        else:  # Keyword Search (BM25)
            progress(0.2, desc="Building BM25 index...")
            retriever = BM25Retriever(
                document,
                config=BM25RetrieverConfig(top_k=top_k * 2, min_score=-999.0)
            )
            progress(0.5, desc="Retrieving and reranking...")
            bm25_results = retriever.retrieve(question)
            reranked = rerank(question, bm25_results, top_k=top_k)
            sources = [r.text for r in reranked]
            retrieved_chunks = [
                {"rank": r.final_rank, "score": r.rerank_score, "text": r.text}
                for r in reranked
            ]
            context_html = f'<div class="section-label">Retrieved Context (BM25 + Reranking)</div>'
            context_html += render_context_chunks(retrieved_chunks)

        progress(0.75, desc="Running detection methods...")

        # Map display names to method aliases
        method_alias_map = {
            "Citation Alignment": "citation",
            "NLI Entailment": "nli",
            "LLM Judge": "judge",
            "Span Coverage": "span",
            "Self-Consistency": "consistency",
        }
        methods = [method_alias_map[m] for m in detection_methods if m in method_alias_map]
        if not methods:
            methods = ["citation", "nli"]

        result = check_answer(
            question=question,
            answer=answer,
            sources=sources,
            methods=methods,
        )

        progress(1.0, desc="Done.")

        verdict_html = render_verdict(
            result.confidence.value,
            result.overall_score,
            build_explanation(result),
        )

        bars_html = f'<div class="section-label">Method Scores</div>'
        bars_html += render_method_bars(result.method_breakdown)

        claims_html = f'<div class="section-label">Flagged Claims</div>'
        claims_html += render_flagged_claims(result.flagged_claims)

        result_json = json.dumps({
            "flagged": result.flagged,
            "confidence": result.confidence.value,
            "overall_score": result.overall_score,
            "methods_used": result.methods_used,
            "flagged_claims": result.flagged_claims,
            "method_scores": {
                m: {"score": r.score, "flagged": r.flagged}
                for m, r in result.method_breakdown.items()
            },
        }, indent=2)

        comparison_html = ""
        if answer_b and answer_b.strip():
            progress(0.9, desc="Running comparison...")
            comparison_html, _ = run_comparison(
                document, question, answer, answer_b,
                retrieval_mode, detection_methods,
            )

        return verdict_html, context_html, bars_html, claims_html, result_json, "", comparison_html

    except Exception as e:
        return f"Error: {str(e)}", "", "", "", "", str(e), ""

        return verdict_html, context_html, bars_html, claims_html, result_json, ""

    except Exception as e:
        return f"Error: {str(e)}", "", "", "", "", str(e)


def run_comparison(
    document: str,
    question: str,
    answer_a: str,
    answer_b: str,
    retrieval_mode: str,
    detection_methods: list[str],
    progress=gr.Progress(),
) -> tuple:
    """Run verification on both answers and return comparison HTML."""
    from veracitor import check_answer
    from veracitor.retrieval.bm25_retriever import BM25Retriever, BM25RetrieverConfig
    from veracitor.retrieval.reranker import rerank

    if not answer_b.strip():
        return "<p style='color:var(--text-muted)'>Enter Answer B to enable comparison mode.</p>", ""

    progress(0.2, desc="Indexing document...")
    retriever = BM25Retriever(
        document,
        config=BM25RetrieverConfig(top_k=10, min_score=-999.0)
    )
    bm25_results = retriever.retrieve(question)
    reranked = rerank(question, bm25_results, top_k=3)
    sources = [r.text for r in reranked]

    method_alias_map = {
        "Citation Alignment": "citation",
        "NLI Entailment": "nli",
        "LLM Judge": "judge",
        "Span Coverage": "span",
        "Self-Consistency": "consistency",
    }
    methods = [method_alias_map[m] for m in detection_methods if m in method_alias_map]
    if not methods:
        methods = ["citation", "nli"]

    progress(0.5, desc="Verifying Answer A...")
    result_a = check_answer(question=question, answer=answer_a, sources=sources, methods=methods)

    progress(0.8, desc="Verifying Answer B...")
    result_b = check_answer(question=question, answer=answer_b, sources=sources, methods=methods)

    progress(1.0, desc="Done.")

    # Determine winner
    winner = "A" if result_a.overall_score >= result_b.overall_score else "B"

    color_a = {"PASS": "#00CC88", "REVIEW": "#FFB800", "FAIL": "#FF4455"}[result_a.confidence.value]
    color_b = {"PASS": "#00CC88", "REVIEW": "#FFB800", "FAIL": "#FF4455"}[result_b.confidence.value]

    border_a = "comparison-winner" if winner == "A" else "comparison-loser"
    border_b = "comparison-winner" if winner == "B" else "comparison-loser"

    comparison_html = f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;">
        <div class="card {border_a}">
            <div class="section-label" style="color:{color_a}">
                Answer A {'— BETTER SUPPORTED' if winner == 'A' else ''}
            </div>
            <div style="font-family:var(--font-display); font-size:1.4rem; color:{color_a}; margin-bottom:0.5rem;">
                {result_a.confidence.value}
            </div>
            <div style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted); margin-bottom:1rem;">
                Score: {result_a.overall_score:.3f}
            </div>
            <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:1rem;">
                {answer_a[:200]}{'...' if len(answer_a) > 200 else ''}
            </div>
            {render_method_bars(result_a.method_breakdown)}
            {render_flagged_claims(result_a.flagged_claims) if result_a.flagged_claims else ''}
        </div>
        <div class="card {border_b}">
            <div class="section-label" style="color:{color_b}">
                Answer B {'— BETTER SUPPORTED' if winner == 'B' else ''}
            </div>
            <div style="font-family:var(--font-display); font-size:1.4rem; color:{color_b}; margin-bottom:0.5rem;">
                {result_b.confidence.value}
            </div>
            <div style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted); margin-bottom:1rem;">
                Score: {result_b.overall_score:.3f}
            </div>
            <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:1rem;">
                {answer_b[:200]}{'...' if len(answer_b) > 200 else ''}
            </div>
            {render_method_bars(result_b.method_breakdown)}
            {render_flagged_claims(result_b.flagged_claims) if result_b.flagged_claims else ''}
        </div>
    </div>
    """

    comparison_json = json.dumps({
        "answer_a": {"confidence": result_a.confidence.value, "score": result_a.overall_score},
        "answer_b": {"confidence": result_b.confidence.value, "score": result_b.overall_score},
        "winner": f"Answer {winner}",
    }, indent=2)

    return comparison_html, comparison_json


def load_gallery_example(example_id: str) -> tuple:
    """Load a gallery example into the input fields."""
    for ex in GALLERY_EXAMPLES:
        if ex["id"] == example_id:
            return (
                ex["document"],
                ex["question"],
                ex["answer_a"],
                ex["answer_b"],
                ex.get("note", ""),
            )
    return "", "", "", "", ""


def render_gallery() -> str:
    """Render the gallery as HTML cards."""
    color_map = {
        "clean": "#00CC88",
        "faithfulness": "#FF4455",
        "factuality": "#FF8844",
        "relevance": "#FFB800",
    }

    html = '<div style="display:grid; grid-template-columns:repeat(2, 1fr); gap:1rem;">'
    for ex in GALLERY_EXAMPLES:
        h_type = ex["hallucination_type"]
        color = color_map.get(h_type, "#8888AA")
        has_comparison = "Yes" if ex["answer_b"] else "No"

        html += f"""
        <div class="gallery-card" onclick="document.getElementById('gallery-select').value='{ex['id']}'">
            <span class="gallery-type-badge">{ex['doc_type']}</span>
            <span class="gallery-hallucination-badge" style="background:rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.15); color:{color};">
                {h_type}
            </span>
            <div style="font-weight:500; font-size:0.9rem; margin:0.6rem 0 0.3rem; color:var(--text-primary);">
                {ex['title']}
            </div>
            <div style="font-size:0.78rem; color:var(--text-secondary); line-height:1.5;">
                {ex['note']}
            </div>
            <div style="font-family:var(--font-mono); font-size:0.65rem; color:var(--text-muted); margin-top:0.75rem;">
                Comparison mode: {has_comparison} &nbsp;|&nbsp; Expected: {ex['expected_verdict']}
            </div>
        </div>
        """

    html += '</div>'
    return html


# --- Gradio app ---

HEADER_HTML = """
<div class="veracitor-header">
    <div class="veracitor-title">Veracitor</div>
    <div class="veracitor-subtitle">Hallucination Detection for Document Intelligence</div>
    <div class="veracitor-tagline">
        Somewhere between the SEC filing and the model's answer, a number changed. Veracitor finds it.
    </div>
</div>
"""

DETECTION_CHOICES = [
    "Citation Alignment",
    "NLI Entailment",
    "Span Coverage",
    "LLM Judge",
    "Self-Consistency",
]

RETRIEVAL_CHOICES = [
    "Keyword Search (BM25)",
    "Smart Keyword Search (Contextual BM25)",
    "Full Document (CAG)",
]

with gr.Blocks() as demo:
    gr.HTML(HEADER_HTML)

    with gr.Tabs(elem_classes="tab-nav"):

        # ------------------------------------------------------------------ #
        # Tab 1: Verify Answer
        # ------------------------------------------------------------------ #
        with gr.Tab("Verify Answer"):
            with gr.Row():
                # Left panel: inputs
                with gr.Column(scale=5):
                    gr.HTML('<div class="section-label">Document</div>')
                    gr.HTML('<div style="font-family:var(--font-mono); font-size:0.7rem; color:#555570; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem;">Supports: 10-K filings, earnings transcripts, fund prospectuses, or any text document</div>')
                    document_input = gr.Textbox(
                        placeholder="Paste the source document or relevant excerpt here...",
                        lines=8,
                        label="",
                        container=False,
                    )

                    gr.HTML('<div class="section-label" style="margin-top:1rem;">Question & Answers</div>')
                    question_input = gr.Textbox(
                        placeholder="What question was asked?",
                        lines=2,
                        label="Question",
                    )
                    answer_a_input = gr.Textbox(
                        placeholder="Paste the generated answer to verify...",
                        lines=3,
                        label="Answer",
                    )

                    with gr.Accordion("Comparison Mode (Answer A vs B)", open=False):
                        answer_b_input = gr.Textbox(
                            placeholder="Paste a second answer to compare side by side...",
                            lines=3,
                            label="Answer B (optional)",
                        )

                    with gr.Accordion("Advanced Settings", open=False):
                        retrieval_mode = gr.Dropdown(
                            choices=RETRIEVAL_CHOICES,
                            value="Keyword Search (BM25)",
                            label="Retrieval Strategy",
                        )
                        detection_methods = gr.CheckboxGroup(
                            choices=DETECTION_CHOICES,
                            value=["Citation Alignment", "NLI Entailment"],
                            label="Detection Methods",
                        )

                    check_btn = gr.Button(
                        "Check Answer",
                        variant="primary",
                        size="lg",
                    )

                # Right panel: outputs
                with gr.Column(scale=6):
                    verdict_output = gr.HTML(
                        '<div style="color:#555570; font-size:0.85rem; padding:2rem 0;">Run verification to see results.</div>',
                        min_height=120,
                    )
                    comparison_output = gr.HTML("", min_height=0)
                    context_output = gr.HTML("", min_height=0)
                    bars_output = gr.HTML("", min_height=0)
                    claims_output = gr.HTML("", min_height=0)

                    with gr.Accordion("Export Results (JSON)", open=False):
                        json_output = gr.Code(language="json", label="")

                    with gr.Row():
                        feedback_up = gr.Button("Verdict is correct", size="sm")
                        feedback_down = gr.Button("Verdict is wrong", size="sm")
                    feedback_msg = gr.HTML("")

            # Wire check button
            check_btn.click(
                fn=run_verification,
                inputs=[
                    document_input,
                    question_input,
                    answer_a_input,
                    retrieval_mode,
                    detection_methods,
                ],
                outputs=[
                    verdict_output,
                    context_output,
                    bars_output,
                    claims_output,
                    json_output,
                    feedback_msg,
                    comparison_output
                ],
            )
           

            # Feedback buttons
            feedback_up.click(
                fn=lambda: '<p style="color:var(--accent-green); font-family:var(--font-mono); font-size:0.75rem;">Feedback recorded. Thank you.</p>',
                outputs=feedback_msg,
            )
            feedback_down.click(
                fn=lambda: '<p style="color:var(--accent-amber); font-family:var(--font-mono); font-size:0.75rem;">Feedback recorded. We will review.</p>',
                outputs=feedback_msg,
            )

        # ------------------------------------------------------------------ #
        # Tab 2: Method Comparison
        # ------------------------------------------------------------------ #
        with gr.Tab("Method Comparison"):
            gr.HTML('<div style="color:var(--text-secondary); font-size:0.875rem; margin-bottom:1.5rem;">Run all detection methods simultaneously and see where they agree or diverge.</div>')

            with gr.Row():
                with gr.Column(scale=5):
                    mc_document = gr.Textbox(
                        placeholder="Paste source document...",
                        lines=8,
                        label="Document",
                    )
                    mc_question = gr.Textbox(
                        placeholder="Question",
                        lines=2,
                        label="Question",
                    )
                    mc_answer = gr.Textbox(
                        placeholder="Generated answer to verify...",
                        lines=3,
                        label="Answer",
                    )
                    mc_retrieval = gr.Dropdown(
                        choices=RETRIEVAL_CHOICES,
                        value="Keyword Search (BM25)",
                        label="Retrieval Strategy",
                    )
                    mc_btn = gr.Button("Run All Methods", variant="primary")

                with gr.Column(scale=6):
                    mc_output = gr.HTML(
                        '<div style="color:var(--text-muted); font-size:0.85rem; padding:2rem 0;">Run comparison to see method-by-method breakdown.</div>'
                    )

            def run_all_methods(document, question, answer, retrieval_mode, progress=gr.Progress()):
                from veracitor import check_answer
                from veracitor.retrieval.bm25_retriever import BM25Retriever, BM25RetrieverConfig
                from veracitor.retrieval.reranker import rerank

                if not all([document.strip(), question.strip(), answer.strip()]):
                    return "<p style='color:var(--text-muted)'>Please fill in all fields.</p>"

                progress(0.2, desc="Indexing...")
                retriever = BM25Retriever(document, config=BM25RetrieverConfig(top_k=10, min_score=-999.0))
                bm25_results = retriever.retrieve(question)
                reranked = rerank(question, bm25_results, top_k=3)
                sources = [r.text for r in reranked]

                progress(0.5, desc="Running all methods...")
                all_methods = ["citation", "nli", "span"]
                result = check_answer(question=question, answer=answer, sources=sources, methods=all_methods)

                progress(1.0)

                color_map = {"PASS": "#00CC88", "REVIEW": "#FFB800", "FAIL": "#FF4455"}
                color = color_map[result.confidence.value]

                method_label_map = {
                    "citation_alignment": "Citation Alignment",
                    "nli_entailment": "NLI Entailment",
                    "span_coverage": "Span Coverage",
                }

                method_rows = ""
                for method, r in result.method_breakdown.items():
                    flag_color = "#FF4455" if r.flagged else "#00CC88"
                    flag_text = "FLAGGED" if r.flagged else "OK"
                    explanation = r.explanation or ""
                    label = method_label_map.get(method, method)
                    method_rows += f"""
                    <tr>
                        <td style="font-family:var(--font-mono); font-size:0.78rem; padding:0.75rem; color:var(--text-secondary); border-bottom:1px solid var(--border-subtle);">{label}</td>
                        <td style="font-family:var(--font-mono); font-size:0.78rem; padding:0.75rem; border-bottom:1px solid var(--border-subtle);">{r.score:.3f}</td>
                        <td style="font-family:var(--font-mono); font-size:0.72rem; padding:0.75rem; color:{flag_color}; border-bottom:1px solid var(--border-subtle);">{flag_text}</td>
                        <td style="font-size:0.78rem; padding:0.75rem; color:var(--text-secondary); border-bottom:1px solid var(--border-subtle);">{explanation[:80] if explanation else '-'}</td>
                    </tr>
                    """

                all_agree = all(r.flagged == result.flagged for r in result.method_breakdown.values())
                agreement_text = "All methods agree" if all_agree else "Methods diverge on this answer"
                agreement_color = "#00CC88" if all_agree else "#FFB800"

                html = f"""
                <div class="card" style="margin-bottom:1rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div class="section-label">Overall Verdict</div>
                            <div style="font-family:var(--font-display); font-size:1.8rem; color:{color};">{result.confidence.value}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-family:var(--font-mono); font-size:0.72rem; color:{agreement_color};">{agreement_text}</div>
                            <div style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted); margin-top:0.25rem;">Score: {result.overall_score:.3f}</div>
                        </div>
                    </div>
                </div>
                <div class="section-label">Method Breakdown</div>
                <table style="width:100%; border-collapse:collapse; background:var(--bg-card); border:1px solid var(--border); border-radius:8px; overflow:hidden;">
                    <thead>
                        <tr style="background:var(--bg-elevated);">
                            <th style="font-family:var(--font-mono); font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; padding:0.75rem; text-align:left; color:var(--text-muted); border-bottom:1px solid var(--border);">Method</th>
                            <th style="font-family:var(--font-mono); font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; padding:0.75rem; text-align:left; color:var(--text-muted); border-bottom:1px solid var(--border);">Score</th>
                            <th style="font-family:var(--font-mono); font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; padding:0.75rem; text-align:left; color:var(--text-muted); border-bottom:1px solid var(--border);">Status</th>
                            <th style="font-family:var(--font-mono); font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; padding:0.75rem; text-align:left; color:var(--text-muted); border-bottom:1px solid var(--border);">Explanation</th>
                        </tr>
                    </thead>
                    <tbody>{method_rows}</tbody>
                </table>
                """
                return html

            mc_btn.click(
                fn=run_all_methods,
                inputs=[mc_document, mc_question, mc_answer, mc_retrieval],
                outputs=mc_output,
            )

        # ------------------------------------------------------------------ #
        # Tab 3: Gallery
        # ------------------------------------------------------------------ #
        with gr.Tab("Gallery"):
            gr.HTML('<div style="color:var(--text-secondary); font-size:0.875rem; margin-bottom:1.5rem;">Pre-computed examples demonstrating each hallucination type and detection method behavior. Click a button below to load an example into the Verify Answer tab.</div>')

            gallery_html = gr.HTML(render_gallery())

            gallery_note = gr.HTML("")

            with gr.Row():
                for ex in GALLERY_EXAMPLES:
                    load_ex_btn = gr.Button(
                        ex["title"],
                        size="sm",
                        variant="secondary",
                    )
                    load_ex_btn.click(
                        fn=lambda eid=ex["id"]: load_and_clear(eid),
                        outputs=[
                            document_input,
                            question_input,
                            answer_a_input,
                            answer_b_input,
                            gallery_note,
                            verdict_output,
                            comparison_output,
                            context_output,
                            bars_output,
                            claims_output,
                            json_output,
                        ],
                    )

            gallery_note = gr.HTML("")

            def load_and_clear(example_id):
                doc, q, a, b, note = load_gallery_example(example_id)
                return doc, q, a, b, note, "", "", "", "", "", ""


if __name__ == "__main__":
    demo.launch(
    show_error=True,
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
    ).set(
        body_background_fill="#0A0A0F",
        body_background_fill_dark="#0A0A0F",
        block_background_fill="#16161F",
        block_background_fill_dark="#16161F",
        border_color_primary="#2A2A3A",
        border_color_primary_dark="#2A2A3A",
        color_accent_soft="#4A9EFF",
    ),
    share=True
)