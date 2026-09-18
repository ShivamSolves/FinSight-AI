"""
qa_pipeline.py — Retrieval-Augmented Generation pipeline for finance Q&A.

Architecture
────────────
                 ┌─────────────┐    embed query    ┌──────────────┐
  User question  │  qa_pipeline │ ────────────────► │  ChromaDB    │
                 │  .ask()      │ ◄──── top-K chunks │  vector store│
                 └──────┬──────┘                    └──────────────┘
                        │  format inventory prompt
                        ▼
                 ┌─────────────┐
                 │  GPT-4o-mini│  ← system prompt locks it to INR + grounded math
                 └──────┬──────┘
                        │
                        ▼
                 Structured answer: total, evidence rows, confidence

Key design rules:
  1. The LLM never retrieves — it only sees what we explicitly pass.
     Retrieval is ChromaDB's job. The LLM is a calculator + formatter.
  2. The context is structured as a numbered inventory with explicit ₹
     amounts, so the model has everything it needs to add up figures.
  3. The system prompt enforces three hard constraints:
       a) Always answer in ₹ (INR) — never assume USD or another currency
       b) Calculate totals from the transaction list provided; never estimate
       c) If the data is insufficient, say "I cannot determine" explicitly
  4. The model is asked to show its working (which transactions it used),
     making every answer independently verifiable.

Public API
──────────
  ask(question, n_results=None) -> AnswerResult (dataclass)
  build_and_ask(df, question)   -> AnswerResult  (builds store then asks)
"""

import logging
from dataclasses import dataclass

from finsight.config import (
    CURRENCY_NAME,
    CURRENCY_SYMBOL,
    LLM_MODEL,
    LLM_PROVIDER,
    TOP_K_RESULTS,
)
from finsight.embeddings.vector_store import (
    build_vector_store,
    get_collection_count,
    query_vector_store,
)
from finsight.rag.query_parser import build_where, parse_query

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AnswerResult:
    """
    Everything the pipeline produces for one question.
    Keeping this as a dataclass (not just a dict) makes it easy to
    iterate over in the evaluation module later.
    """
    question:        str
    answer:          str                  # the LLM's final answer text
    evidence:        list[dict]           # retrieved chunks that were passed in
    n_chunks_used:   int                  # how many chunks the retriever returned
    model:           str                  # which LLM was used
    raw_response:    str = ""             # full LLM output, kept for debugging
    error:           str = ""            # non-empty if something went wrong
    scope:           str = ""            # human-readable description of the retrieval filter


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """
    The system prompt is the contract between us and the LLM.
    It is intentionally strict: we want grounded, auditable answers,
    not fluent-but-wrong summaries.
    """
    return f"""You are FinSight, a personal finance assistant. Your job is to
answer questions about a user's bank transactions accurately and honestly.

STRICT RULES — follow these exactly:

1. CURRENCY: Always express all amounts in {CURRENCY_SYMBOL} ({CURRENCY_NAME}).
   Never convert to USD or any other currency. Never omit the {CURRENCY_SYMBOL} symbol.

2. GROUNDING: Base every number in your answer exclusively on the transaction
   list provided in the user's message. Do not estimate, infer, or recall
   information that is not explicitly present in the list.

3. ARITHMETIC: When a question asks for a total or aggregate, add up the
   relevant amounts from the list yourself. Show which transactions you
   included (by description and date) so the user can verify your working.

4. UNCERTAINTY: If the transaction list does not contain enough information
   to answer the question confidently, say so clearly:
   "I cannot determine this from the available transactions."
   Do not guess or approximate.

5. ANSWER FORMAT: Structure every response exactly as follows:

   **Answer:** <your direct answer with {CURRENCY_SYMBOL} amounts>

   **Transactions used:**
   - <Date> | <Description> | <{CURRENCY_SYMBOL} Amount>
   (list every transaction you counted; write "None" if not applicable)

   **Confidence:** <High | Medium | Low | Cannot determine>
   <one sentence explaining why — e.g., "All relevant transactions for the
   period are present in the data" or "Only partial data available for March">

Do not add extra sections, caveats, or marketing language beyond this format.
"""


# ── Context builder ───────────────────────────────────────────────────────────

def _build_context(hits: list[dict], scope: str = "", complete: bool = False) -> str:
    """
    Format retrieved chunks into a numbered transaction inventory.

    Giving each transaction a number makes it easy for the LLM to reference
    them by index when showing its working, and makes the prompt token-efficient
    (we pack all the relevant data into a structured list rather than prose).

    `complete` changes the framing: when True the list is EVERY transaction
    matching the filter (so totals are exact), when False it's a relevance-ranked
    sample (so the model must not assume it can compute an exhaustive total).
    """
    if not hits:
        return "No relevant transactions found."

    if complete:
        header = (
            f"RETRIEVED TRANSACTIONS — the COMPLETE set matching your filter "
            f"({len(hits)} transactions)."
        )
    else:
        header = (
            f"RETRIEVED TRANSACTIONS ({len(hits)} most relevant to your question):"
        )

    lines = [header]
    if scope:
        lines.append(f"Filter applied: {scope}")
    lines.append("")

    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        # Use abs_amount from metadata (always positive) for clean display
        amount = abs(m["amount"])
        lines.append(
            f"  [{i:02d}] {m['date']}  |  {m['description']:<45}  |  "
            f"{CURRENCY_SYMBOL}{amount:>10,.2f}  |  {m['category']}  |  {m['transaction_type']}"
        )

    if complete:
        lines += [
            "",
            "Note: This is the full set of transactions matching the filter above.",
            "You may compute exact totals from it.",
        ]
    else:
        lines += [
            "",
            "Note: These are the transactions the retrieval system judged most",
            "relevant to your question. This may NOT be the complete set, so do",
            "not assume a total over these is exhaustive unless the question is",
            "about these specific rows.",
        ]
    return "\n".join(lines)


# ── LLM caller ────────────────────────────────────────────────────────────────

def _call_llm(system_prompt: str, user_message: str) -> str:
    """
    Send the prompt to the configured LLM and return the response text.

    Currently supports OpenAI (gpt-4o-mini) and Anthropic (claude-haiku).
    Provider is set in config.py — LLM_PROVIDER and LLM_MODEL.

    The API key must be set as an environment variable:
      OpenAI:    $env:OPENAI_API_KEY    = "sk-..."
      Anthropic: $env:ANTHROPIC_API_KEY = "sk-ant-..."
    Or place in a .env file at the project root (never commit this file).
    """
    # Load .env if present (python-dotenv)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv is optional; env vars may already be set

    provider = LLM_PROVIDER.lower().strip()

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from environment
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_message},
            ],
            temperature=0,     # deterministic — we want math, not creativity
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' in config.py. "
            "Supported: 'openai', 'anthropic'."
        )


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(question: str, n_results: int | None = None) -> tuple[list[dict], "object", bool]:
    """
    Run the parse → filter → retrieve path and return the matching transactions.

    This is the retrieval half of ask(), factored out so the evaluation module
    and tests exercise the exact same code path the LLM sees.

    Returns:
        (hits, parsed, complete) where `parsed` is the ParsedQuery plan and
        `complete` is True when the retriever fetched the full matching set
        (rather than a relevance-ranked top-K sample).
    """
    parsed = parse_query(question)
    where  = build_where(parsed)

    # When the question needs a COMPLETE set (aggregate or hard filter), ignore
    # the top-K cutoff and fetch every matching row so totals are exact.
    complete = parsed.wants_all
    if n_results is None:
        n_results = get_collection_count() if complete else TOP_K_RESULTS
    elif complete:
        n_results = max(n_results, get_collection_count())

    logger.info(
        f"  Parsed → intent={parsed.intent} merchant={parsed.merchant} "
        f"categories={parsed.categories} months={parsed.months or parsed.month_nums} "
        f"wants_all={parsed.wants_all}"
    )

    hits = query_vector_store(
        question,
        n_results=n_results,
        where=where,
        merchant=parsed.merchant,
    )
    return hits, parsed, complete


def ask(question: str, n_results: int | None = None) -> AnswerResult:
    """
    Answer a natural language question about transactions using RAG.

    Assumes the ChromaDB vector store has already been built via
    build_vector_store(). Call build_and_ask() if you want to build + ask
    in a single step.

    Args:
        question:  Plain English question, e.g. "How much did I spend on
                   Swiggy in January?"
        n_results: Number of chunks to retrieve (default: TOP_K_RESULTS
                   from config.py, currently 10). Aggregate/filtered questions
                   override this to fetch the complete matching set.

    Returns:
        AnswerResult with .answer, .evidence, and .raw_response fields.
    """
    logger.info(f"Question: {question!r}")

    # ── 1. Retrieve relevant transactions ─────────────────────────────────────
    try:
        hits, parsed, complete = retrieve(question, n_results=n_results)
    except Exception as e:
        msg = f"Vector store query failed: {e}. Has build_vector_store() been run?"
        logger.error(msg)
        scope = parse_query(question).describe_scope()
        return AnswerResult(
            question=question, answer="", evidence=[], n_chunks_used=0,
            model=LLM_MODEL, error=msg, scope=scope,
        )

    logger.info(f"  Retrieved {len(hits)} chunks (filter: {parsed.describe_scope()})")

    # ── 2. Build prompt ────────────────────────────────────────────────────────
    system_prompt = _build_system_prompt()
    context       = _build_context(hits, scope=parsed.describe_scope(), complete=complete)
    user_message  = f"{context}\n\nQuestion: {question}"

    # ── 3. Call LLM ───────────────────────────────────────────────────────────
    try:
        raw_response = _call_llm(system_prompt, user_message)
    except Exception as e:
        msg = f"LLM call failed: {e}"
        logger.error(msg)
        return AnswerResult(
            question=question, answer="", evidence=hits,
            n_chunks_used=len(hits), model=LLM_MODEL, error=msg,
            scope=parsed.describe_scope(),
        )

    logger.info("  LLM response received ✓")

    return AnswerResult(
        question=question,
        answer=raw_response,
        evidence=hits,
        n_chunks_used=len(hits),
        model=LLM_MODEL,
        raw_response=raw_response,
        scope=parsed.describe_scope(),
    )


def build_and_ask(df, question: str, n_results: int | None = None) -> AnswerResult:
    """
    Convenience wrapper: build the vector store from a DataFrame, then ask.

    Use this when you want a single-call pipeline, e.g. in tests or scripts
    where you don't want to manage the store lifecycle separately.

    Args:
        df:        Categorized transactions DataFrame (from categorize_dataframe).
        question:  Plain English question.
        n_results: Number of chunks to retrieve.

    Returns:
        AnswerResult.
    """
    build_vector_store(df)
    return ask(question, n_results=n_results)
