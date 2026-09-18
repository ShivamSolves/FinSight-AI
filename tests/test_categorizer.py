"""
tests/test_categorizer.py — Unit tests for the rule-based categorizer.
"""

import pandas as pd
import pytest

from finsight.categorization.categorizer import (
    categorize_dataframe,
    categorize_transaction,
    get_category_summary,
)


@pytest.mark.parametrize("description,expected", [
    ("SWIGGY ORDER 1234", "Food & Dining"),
    ("ZOMATO DELIVERY", "Food & Dining"),
    ("BIGBASKET GROCERY", "Groceries"),
    ("DMART SUPERMARKET", "Groceries"),
    ("SALARY CREDIT - INFOSYS", "Income"),
    ("FREELANCE PAYMENT", "Income"),
    ("NETFLIX SUBSCRIPTION", "Subscriptions"),
    ("SPOTIFY PREMIUM", "Subscriptions"),
    ("RENT PAYMENT - PRESTIGE", "Rent/Housing"),
    ("UBER INDIA TRIP", "Transport"),
    ("BESCOM ELECTRICITY BILL", "Utilities"),
    ("APOLLO PHARMACY", "Healthcare"),
    ("PVR CINEMAS", "Entertainment"),
    ("AMAZON.IN ORDER", "Shopping"),
    ("ATM WITHDRAWAL", "Other"),
    ("RANDOM UNKNOWN MERCHANT XYZ", "Uncategorized"),
])
def test_categorize_transaction(description, expected):
    assert categorize_transaction(description) == expected


def test_groceries_checked_before_food_dining():
    # "kirana" (Groceries) and "restaurant" (Food & Dining) both present —
    # Groceries is checked first, so it must win.
    assert categorize_transaction("Kirana Store Restaurant") == "Groceries"


def test_plural_groceries_matches():
    assert categorize_transaction("Wholesome Market - Groceries") == "Groceries"


def test_amazon_prime_is_subscription_not_shopping():
    assert categorize_transaction("AMAZON PRIME MEMBERSHIP") == "Subscriptions"


def test_uncategorized_is_not_other():
    # No match → "Uncategorized" (needs review), NOT the "Other" catch-all.
    result = categorize_transaction("ZZZQ UNKNOWN 999")
    assert result == "Uncategorized"
    assert result != "Other"


def _sample_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-03", "2026-01-04", "2026-01-05", "2026-02-01"]),
        "description": ["SALARY CREDIT", "RENT PAYMENT", "BIGBASKET GROCERY", "UNKNOWN XYZ"],
        "amount": [62000.0, -18000.0, -1840.0, -500.0],
        "transaction_type": ["credit", "debit", "debit", "debit"],
    })


def test_categorize_dataframe_adds_column_without_mutating_input():
    df = _sample_df()
    original_cols = list(df.columns)
    result = categorize_dataframe(df)
    assert "category" in result.columns
    assert "category" not in df.columns          # input untouched
    assert list(df.columns) == original_cols
    # categorize_dataframe does NOT re-case descriptions (that's the extractor's job)
    assert result.loc[result["description"] == "SALARY CREDIT", "category"].iloc[0] == "Income"
    assert result.loc[result["description"] == "UNKNOWN XYZ", "category"].iloc[0] == "Uncategorized"


def test_category_summary_excludes_non_spending_from_pct():
    result = categorize_dataframe(_sample_df())
    summary = get_category_summary(result)
    assert set(summary.columns) == {"category", "total_amount", "txn_count", "pct_of_spending"}

    income_row = summary[summary["category"] == "Income"].iloc[0]
    assert income_row["pct_of_spending"] == "—"   # Income is not part of the spending base

    rent_row = summary[summary["category"] == "Rent/Housing"].iloc[0]
    # spending base = 18000 + 1840 + 500 = 20340 → rent ≈ 88.5%
    assert rent_row["pct_of_spending"] == f"{18000 / 20340 * 100:.1f}%"
