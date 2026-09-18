"""Tests for the PDF table → DataFrame conversion (no real PDF needed)."""

import pandas as pd
import pytest

from finsight.ingestion.csv_extractor import clean_transactions
from finsight.ingestion.pdf_extractor import tables_to_dataframe

HEADERED = [
    [
        ["Date", "Description", "Amount"],
        ["2026-01-05", "BIGBASKET - GROCERY DELIVERY", "-1840.00"],
        ["2026-01-03", "SALARY CREDIT - INFOSYS LTD", "62000.00"],
        [None, None, None],
    ]
]

REPEATED_HEADER = [
    [
        ["Date", "Description", "Amount"],
        ["2026-01-05", "UBER INDIA", "-300.00"],
        ["Date", "Description", "Amount"],
        ["2026-01-06", "CHAI POINT CAFE", "-120.00"],
    ]
]

SPLIT = [
    [
        ["Transaction Date", "Narrative", "Debit", "Credit"],
        ["2026-02-01", "RENT PAYMENT", "18000.00", ""],
        ["2026-02-03", "SALARY", "", "62000.00"],
    ]
]

HEADERLESS = [
    [
        ["2026-03-01", "SWIGGY ORDER", "450.00", "REF-1"],
        ["2026-03-02", "DMART - GROCERIES", "1980.00", "REF-2"],
        ["2026-03-03", "UBER INDIA", "300.00", "REF-3"],
    ]
]


def test_headered_table_keeps_names_and_drops_empty_rows():
    df = tables_to_dataframe(HEADERED)
    assert list(df.columns) == ["Date", "Description", "Amount"]
    assert len(df) == 2


def test_repeated_per_page_header_is_dropped():
    df = tables_to_dataframe(REPEATED_HEADER)
    assert len(df) == 2
    assert (df["Description"] == ["UBER INDIA", "CHAI POINT CAFE"]).all()


def test_headered_flows_through_shared_cleaning():
    df = clean_transactions(tables_to_dataframe(HEADERED))
    assert list(df.columns) == ["date", "description", "amount", "transaction_type"]
    assert df.loc[df["description"].str.contains("Infosys"), "amount"].iloc[0] == 62000.0
    assert df.loc[df["description"].str.contains("Bigbasket"), "transaction_type"].iloc[0] == "debit"


def test_split_debit_credit_columns_merge():
    df = clean_transactions(tables_to_dataframe(SPLIT))
    rent = df.loc[df["description"].str.contains("Rent"), "amount"].iloc[0]
    salary = df.loc[df["description"].str.contains("Salary"), "amount"].iloc[0]
    assert rent == -18000.0
    assert salary == 62000.0


def test_headerless_columns_inferred_positionally():
    df = tables_to_dataframe(HEADERLESS)
    assert set(df.columns) == {"date", "description", "amount"}
    assert len(df) == 3
    clean = clean_transactions(df)
    assert clean["amount"].abs().sum() == pytest.approx(2730.0)


def test_no_usable_tables_raises():
    with pytest.raises(ValueError, match="No usable transaction tables"):
        tables_to_dataframe([[["foo", "bar"], ["baz", "qux"]]])


def test_empty_table_list_raises():
    with pytest.raises(ValueError):
        tables_to_dataframe([])


def test_multiple_tables_concatenate():
    df = tables_to_dataframe(HEADERED + [REPEATED_HEADER[0]])
    assert len(df) == 4
    assert isinstance(df, pd.DataFrame)
