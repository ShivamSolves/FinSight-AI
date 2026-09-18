"""
Tests for the synthetic statement generator (finsight/synthetic_statement.py).

The generator is the project's stand-in for real bank data (which is never
public), so these tests guard the properties the rest of the pipeline relies on:
determinism, schema, sign mix, date span — and that every generated merchant
still resolves through the real cleaning + categorization chain.
"""

from datetime import date

import pandas as pd
import pytest

from finsight.categorization.categorizer import categorize_dataframe
from finsight.ingestion.csv_extractor import clean_transactions
from finsight.synthetic_statement import generate_statement

START = date(2026, 1, 1)


@pytest.fixture(scope="module")
def statement() -> pd.DataFrame:
    return generate_statement(months=6, seed=42, start=START)


def test_reproducible_given_same_seed(statement):
    again = generate_statement(months=6, seed=42, start=START)
    pd.testing.assert_frame_equal(statement, again)


def test_different_seed_gives_different_data(statement):
    other = generate_statement(months=6, seed=7, start=START)
    assert not statement.equals(other)


def test_schema_and_sign_mix(statement):
    assert list(statement.columns) == ["Date", "Description", "Amount"]
    assert (statement["Amount"] != 0).all()
    assert (statement["Amount"] > 0).any() and (statement["Amount"] < 0).any()
    parsed = pd.to_datetime(statement["Date"])
    assert parsed.min() >= pd.Timestamp(START)
    assert parsed.max() < pd.Timestamp(START) + pd.DateOffset(months=6)


def test_sorted_by_date(statement):
    parsed = pd.to_datetime(statement["Date"])
    assert parsed.is_monotonic_increasing


def test_scales_with_months():
    short = generate_statement(months=2, seed=42, start=START)
    long = generate_statement(months=12, seed=42, start=START)
    assert len(long) > 3 * len(short)


def test_rejects_zero_months():
    with pytest.raises(ValueError):
        generate_statement(months=0, seed=1, start=START)


def test_survives_full_ingest_and_categorization_chain(statement):
    """Generated rows must flow through the real pipeline with zero drops."""
    clean = clean_transactions(statement.rename(
        columns={"Date": "date", "Description": "description", "Amount": "amount"}))
    assert len(clean) == len(statement)          # no row lost to parsing
    categorized = categorize_dataframe(clean)
    assert (categorized["category"] != "Uncategorized").all()
