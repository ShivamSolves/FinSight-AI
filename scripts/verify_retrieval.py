"""
verify_retrieval.py — Offline correctness check for the retrieval layer.

This does NOT call the LLM (no API key needed). It rebuilds the vector store,
then for each ground-truth question it runs the SAME parse → filter → retrieve
path that qa_pipeline.ask() uses, sums the amounts of the rows that came back,
and compares against a total computed directly from the CSV.

If retrieval is complete and correctly filtered, the retrieved total equals the
CSV total. A mismatch means the retriever dropped or wrongly included rows —
which is exactly the bug that makes aggregate answers silently wrong.

Run: python scripts/verify_retrieval.py
"""

import sys
import logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")

import pandas as pd

from finsight.config import RAW_DIR
from finsight.ingestion.csv_extractor import load_csv
from finsight.categorization.categorizer import categorize_dataframe
from finsight.embeddings.vector_store import (
    build_vector_store, query_vector_store, get_collection_count,
)
from finsight.rag.query_parser import parse_query, build_where


def csv_truth(df: pd.DataFrame, *, merchant=None, category=None, month=None) -> float:
    """Ground-truth total (absolute ₹) computed straight from the DataFrame."""
    mask = pd.Series(True, index=df.index)
    if merchant:
        mask &= df["description"].str.lower().str.contains(merchant.lower())
    if category:
        mask &= df["category"] == category
    if month:
        mask &= df["date"].dt.strftime("%Y-%m") == month
    return round(df.loc[mask, "amount"].abs().sum(), 2)


def retrieve(question: str) -> list[dict]:
    """Mirror qa_pipeline.ask()'s retrieval logic, minus the LLM call."""
    parsed = parse_query(question)
    where = build_where(parsed)
    n_results = get_collection_count() if parsed.wants_all else None
    return query_vector_store(
        question, n_results=n_results, where=where, merchant=parsed.merchant
    )


# question, ground-truth kwargs, human label
CASES = [
    ("How much did I spend at Swiggy in total?",
     dict(merchant="swiggy"), "Merchant lookup — Swiggy"),
    ("What was my total grocery spend in February 2026?",
     dict(category="Groceries", month="2026-02"), "Category+month — Feb groceries"),
    ("How much did I spend on subscriptions across all 3 months?",
     dict(category="Subscriptions"), "Category — all subscriptions"),
    ("What was my total income across January, February, and March?",
     dict(category="Income"), "Category — total income"),
    ("Show my transport spending in January",
     dict(category="Transport", month="2026-01"), "Category+month — Jan transport"),
]


def main() -> int:
    print("\nBuilding pipeline (ingest → categorize → embed)...")
    df = categorize_dataframe(load_csv(RAW_DIR / "bank_statement.csv"))
    build_vector_store(df)
    print(f"  {get_collection_count()} transactions indexed\n")

    sep = "─" * 74
    failures = 0

    for question, truth_kw, label in CASES:
        parsed = parse_query(question)
        hits = retrieve(question)
        retrieved_total = round(sum(abs(h["metadata"]["amount"]) for h in hits), 2)
        expected = csv_truth(df, **truth_kw)
        ok = abs(retrieved_total - expected) < 0.01

        failures += 0 if ok else 1
        status = "PASS" if ok else "FAIL"

        print(sep)
        print(f"  [{status}] {label}")
        print(f"    Q        : {question}")
        print(f"    Scope    : {parsed.describe_scope()}  (wants_all={parsed.wants_all})")
        print(f"    Retrieved: {len(hits):>2} txns  →  ₹{retrieved_total:,.2f}")
        print(f"    Expected :          ₹{expected:,.2f}")
        if not ok:
            print(f"    Δ        : ₹{retrieved_total - expected:,.2f}")

    print(sep)
    if failures:
        print(f"  ✗ {failures}/{len(CASES)} case(s) FAILED — retrieval is incomplete or mis-filtered.")
    else:
        print(f"  ✓ All {len(CASES)} cases PASSED — retrieval returns complete, correct sets.")
    print(sep + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
