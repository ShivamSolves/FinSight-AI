"""
tests/test_retrieval.py — Integration test for retrieval completeness.

Marked `integration` because it builds the ChromaDB store and loads the
embedding model (slow, and needs network on the very first run to fetch the
~80 MB model). Run with:

    pytest -m integration        # just these
    pytest                       # everything (CI runs unit + integration)

It exercises the SAME retrieve() path the LLM sees, and asserts that every
golden case pulls the complete, correct set — recall 1.0 and an exact total.
"""

import pytest

from finsight.categorization.categorizer import categorize_dataframe
from finsight.config import RAW_DIR
from finsight.embeddings.vector_store import build_vector_store
from finsight.evaluation.dataset import GOLDEN_CASES, truth_total
from finsight.evaluation.metrics import retrieval_scores
from finsight.ingestion.csv_extractor import load_csv
from finsight.rag.qa_pipeline import retrieve

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def prepared_df():
    """Load, categorize, and index the sample statement once for the module."""
    df = categorize_dataframe(load_csv(RAW_DIR / "bank_statement.csv"))
    build_vector_store(df)
    return df


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c.id for c in GOLDEN_CASES])
def test_retrieval_is_complete_and_correct(prepared_df, case):
    hits, _parsed, _complete = retrieve(case.question)
    scores = retrieval_scores(prepared_df, hits, case)

    # Every expected transaction was retrieved (no silent top-K truncation)...
    assert scores["recall"] == 1.0, f"{case.id}: recall {scores['recall']}"
    # ...nothing spurious was pulled in...
    assert scores["precision"] == 1.0, f"{case.id}: precision {scores['precision']}"
    # ...and the retrieved total equals both the CSV truth and the hardcoded value.
    assert scores["total_ok"], f"{case.id}: {scores['retrieved_total']} != {scores['expected_total']}"
    assert scores["matches_hardcoded"], f"{case.id}: CSV truth drifted from golden value"


def test_aggregate_exceeds_topk(prepared_df):
    """
    Regression guard for the core bug: the subscriptions answer spans more rows
    than the default TOP_K_RESULTS (10), so a naive top-K retriever would drop
    some and under-report the total. Completeness must beat the cutoff.
    """
    from finsight.config import TOP_K_RESULTS

    subs_case = next(c for c in GOLDEN_CASES if c.id == "all_subscriptions")
    hits, _parsed, complete = retrieve(subs_case.question)

    expected_n = len(prepared_df[prepared_df["category"] == "Subscriptions"])
    assert expected_n > TOP_K_RESULTS, "test premise: subscriptions exceed top-K"
    assert complete is True
    assert len(hits) == expected_n
    assert round(sum(abs(h["metadata"]["amount"]) for h in hits), 2) == truth_total(prepared_df, subs_case)


def test_open_ended_query_uses_topk(prepared_df):
    """A filter-free question should NOT fetch the whole collection."""
    from finsight.config import TOP_K_RESULTS

    hits, _parsed, complete = retrieve("what did I buy")
    assert complete is False
    assert len(hits) <= TOP_K_RESULTS
