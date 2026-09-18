"""
evaluation/dataset.py — The golden Q&A set and its ground truth.

A "golden" case pairs a natural-language question with the answer we can prove
is correct straight from the CSV. Ground truth is computed from the source
DataFrame (independent of the retrieval code under test), and the expected
totals are also hardcoded from a manual read of the statement so a regression
in *either* the data or the retriever is caught.

Used by:
  • finsight/evaluation/runner.py  — the evaluation report
  • tests/test_retrieval.py        — the integration test
  • scripts/run_evaluation.py      — the CLI
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EvalCase:
    """One golden question plus the filter that defines its correct answer set."""
    id:             str
    question:       str
    label:          str
    expected_total: float                 # hardcoded ground truth (₹, absolute)
    merchant:       str | None = None     # description substring filter
    category:       str | None = None     # exact category filter
    month:          str | None = None     # "YYYY-MM" filter


# Totals below were verified directly against data/raw/bank_statement.csv.
GOLDEN_CASES: list[EvalCase] = [
    EvalCase("swiggy", "How much did I spend at Swiggy in total?",
             "Merchant lookup — Swiggy", 910.0, merchant="swiggy"),
    EvalCase("feb_groceries", "What was my total grocery spend in February 2026?",
             "Category + month — Feb groceries", 6070.0, category="Groceries", month="2026-02"),
    EvalCase("all_subscriptions", "How much did I spend on subscriptions across all 3 months?",
             "Category — all subscriptions", 4519.0, category="Subscriptions"),
    EvalCase("total_income", "What was my total income across January, February, and March?",
             "Category — total income", 204500.0, category="Income"),
    EvalCase("jan_transport", "Show my transport spending in January",
             "Category + month — Jan transport", 3200.0, category="Transport", month="2026-01"),
]


def expected_rows(df: pd.DataFrame, case: EvalCase) -> pd.DataFrame:
    """The subset of the CSV that *should* be retrieved for this case."""
    mask = pd.Series(True, index=df.index)
    if case.merchant:
        mask &= df["description"].str.lower().str.contains(case.merchant.lower(), regex=False)
    if case.category:
        mask &= df["category"] == case.category
    if case.month:
        mask &= df["date"].dt.strftime("%Y-%m") == case.month
    return df[mask]


def truth_total(df: pd.DataFrame, case: EvalCase) -> float:
    """Absolute ₹ total of the expected rows, computed from the CSV."""
    return round(expected_rows(df, case)["amount"].abs().sum(), 2)


def row_key(date_str: str, description: str, amount: float) -> tuple:
    """Stable identity for a transaction, used to compare retrieved vs expected."""
    return (str(date_str), str(description).strip().lower(), round(abs(float(amount)), 2))


def expected_keys(df: pd.DataFrame, case: EvalCase) -> set[tuple]:
    rows = expected_rows(df, case)
    return {
        row_key(r["date"].date(), r["description"], r["amount"])
        for _, r in rows.iterrows()
    }
