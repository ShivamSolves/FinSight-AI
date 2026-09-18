"""
tests/test_csv_extractor.py — Unit tests for tolerant CSV ingestion.

Each test writes a small CSV to a temp dir and asserts the clean output schema:
    date (datetime64), description (title-cased str), amount (float,
    +credit/-debit), transaction_type ("credit"/"debit").
"""

import textwrap

import pytest

from finsight.ingestion.csv_extractor import load_csv


def _write(tmp_path, content: str, name: str = "stmt.csv"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(content).strip(), encoding="utf-8")
    return path


def test_basic_schema_and_signs(tmp_path):
    path = _write(tmp_path, """
        date,description,amount
        2026-01-05,BIGBASKET GROCERY,-1840.00
        2026-01-03,SALARY CREDIT,62000.00
    """)
    df = load_csv(path)
    assert list(df.columns) == ["date", "description", "amount", "transaction_type"]
    # sorted by date ascending
    assert df["date"].tolist() == sorted(df["date"].tolist())
    salary = df[df["description"] == "Salary Credit"].iloc[0]
    assert salary["amount"] == 62000.0
    assert salary["transaction_type"] == "credit"


def test_parentheses_mean_negative(tmp_path):
    path = _write(tmp_path, """
        date,description,amount
        2026-02-01,SOME FEE,(50.00)
    """)
    df = load_csv(path)
    assert df.iloc[0]["amount"] == -50.0
    assert df.iloc[0]["transaction_type"] == "debit"


def test_currency_symbols_and_thousands_separators(tmp_path):
    path = _write(tmp_path, """
        date,description,amount
        2026-02-01,AMAZON ORDER,"$1,234.56"
        2026-02-02,REFUND,-$20.00
    """)
    df = load_csv(path)
    amounts = sorted(df["amount"].tolist())
    assert amounts == [-20.0, 1234.56]


def test_split_debit_credit_columns_are_merged(tmp_path):
    path = _write(tmp_path, """
        date,description,debit,credit
        2026-01-10,RENT PAYMENT,18000,
        2026-01-03,SALARY,,62000
    """)
    df = load_csv(path)
    rent = df[df["description"] == "Rent Payment"].iloc[0]
    assert rent["amount"] == -18000.0
    assert rent["transaction_type"] == "debit"
    salary = df[df["description"] == "Salary"].iloc[0]
    assert salary["amount"] == 62000.0
    assert "debit" not in df.columns and "credit" not in df.columns


def test_column_aliases_are_resolved(tmp_path):
    path = _write(tmp_path, """
        Transaction Date,Merchant,Transaction Amount
        2026-03-01,SWIGGY,-420.00
    """)
    df = load_csv(path)
    assert df.iloc[0]["description"] == "Swiggy"
    assert df.iloc[0]["amount"] == -420.0


def test_zero_amount_rows_dropped(tmp_path):
    path = _write(tmp_path, """
        date,description,amount
        2026-01-01,Opening Balance,0.00
        2026-01-02,COFFEE,-150.00
    """)
    df = load_csv(path)
    assert len(df) == 1
    assert "Opening Balance" not in df["description"].tolist()


def test_unparseable_date_row_dropped(tmp_path):
    path = _write(tmp_path, """
        date,description,amount
        not-a-date,GHOST TXN,-10.00
        2026-01-02,REAL TXN,-20.00
    """)
    df = load_csv(path)
    assert len(df) == 1
    assert df.iloc[0]["description"] == "Real Txn"


def test_descriptions_are_cleaned_and_title_cased(tmp_path):
    path = _write(tmp_path, """
        date,description,amount
        2026-01-02,"   cafe    COFFEE   DAY  ",-20.00
    """)
    df = load_csv(path)
    assert df.iloc[0]["description"] == "Cafe Coffee Day"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_csv("does/not/exist.csv")


def test_missing_required_columns_raises(tmp_path):
    path = _write(tmp_path, """
        when,what,howmuch
        2026-01-02,COFFEE,-20.00
    """)
    with pytest.raises(ValueError):
        load_csv(path)


def test_dates_are_normalized_to_midnight(tmp_path):
    path = _write(tmp_path, """
        date,description,amount
        2026-01-02 13:45:00,LUNCH,-200.00
    """)
    df = load_csv(path)
    assert df.iloc[0]["date"].hour == 0
    assert df.iloc[0]["date"].minute == 0
