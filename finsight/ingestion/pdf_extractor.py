"""
pdf_extractor.py — Ingests a bank statement PDF and returns a clean DataFrame.

Bank PDFs are usually a sequence of tables (one per page). pdfplumber extracts
those tables as lists of rows; this module turns them into a DataFrame and then
hands off to csv_extractor.clean_transactions() so CSV and PDF inputs converge
on the identical schema (date, description, amount, transaction_type).

Two table shapes are supported:
  1. Headered tables — the first row contains recognizable column names
     ("Date", "Description", "Amount", or split "Debit"/"Credit").
  2. Headerless tables — columns are inferred positionally: the column whose
     values look like dates, the column whose values look like money, and the
     widest remaining text column as the description.

Public API
──────────
  load_pdf(filepath) -> pd.DataFrame
  tables_to_dataframe(tables) -> pd.DataFrame   (pure, unit-testable)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from finsight.config import CSV_COLUMN_ALIASES
from finsight.ingestion.csv_extractor import clean_transactions

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(
    r"^\d{1,2}[\/\-.]([A-Za-z]{3,9}|\d{1,2})[\/\-.]\d{2,4}$|^\d{4}-\d{2}-\d{2}$"
)
_MONEY_RE = re.compile(r"^\(?\s*[₹$]?\s?-?\s?\d[\d,]*(\.\d{1,2})?\s?\)?$")


def _alias_lookup() -> dict[str, str]:
    lookup = {}
    for standard, aliases in CSV_COLUMN_ALIASES.items():
        for alias in aliases:
            lookup[alias.lower()] = standard
    return lookup


def _cell(row: list, idx: int) -> str:
    if idx >= len(row) or row[idx] is None:
        return ""
    return str(row[idx]).strip()


def _looks_like_date(value: str) -> bool:
    return bool(_DATE_RE.match(value))


def _looks_like_money(value: str) -> bool:
    return bool(_MONEY_RE.match(value))


def _column_values(table: list[list], idx: int, skip: int = 0) -> list[str]:
    return [_cell(row, idx) for row in table[skip:] if _cell(row, idx)]


def _majority(values: list[str], predicate) -> bool:
    if not values:
        return False
    return sum(1 for v in values if predicate(v)) / len(values) >= 0.6


def tables_to_dataframe(tables: list[list[list]]) -> pd.DataFrame:
    """
    Merge pdfplumber tables into one raw DataFrame.

    Headered tables keep their original header names so the shared alias
    resolution in csv_extractor (including split debit/credit merging) applies
    identically to PDF and CSV input. Headerless tables get their columns
    inferred positionally. Empty rows and repeated per-page header rows are
    dropped so multi-page statements concatenate cleanly.
    """
    lookup = _alias_lookup()
    frames = []

    for table in tables:
        if not table:
            continue
        header = [_cell(table[0], i).lower() for i in range(len(table[0]))]
        has_header = any(lookup.get(h) is not None for h in header)

        if has_header:
            names = [_cell(table[0], i) for i in range(len(table[0]))]
            body = [row for row in table[1:] if any(_cell(row, i) for i in range(len(row)))]
            # Some banks repeat the header on every page — drop those rows.
            body = [r for r in body if _cell(r, 0).lower() not in header]
            frames.append(pd.DataFrame(body, columns=names))
        else:
            frame = _infer_positional(table)
            if frame is not None:
                frames.append(frame)

    if not frames:
        raise ValueError(
            "No usable transaction tables found in the PDF. "
            "Expected a table with date / description / amount columns."
        )

    combined = pd.concat(frames, ignore_index=True)
    # Different pages can yield different column sets; align on the union.
    return combined.dropna(how="all")


def _infer_positional(table: list[list]) -> pd.DataFrame | None:
    """Guess date / amount / description columns for a headerless table."""
    n_cols = max(len(row) for row in table)
    date_col = amount_col = desc_col = None

    for idx in range(n_cols):
        values = _column_values(table, idx)
        if date_col is None and _majority(values, _looks_like_date):
            date_col = idx
        elif amount_col is None and _majority(values, _looks_like_money):
            amount_col = idx

    if date_col is None or amount_col is None:
        return None

    # Description = the remaining column with the longest average text.
    best_len = 0
    for idx in range(n_cols):
        if idx in (date_col, amount_col):
            continue
        values = _column_values(table, idx)
        if not values:
            continue
        avg = sum(len(v) for v in values) / len(values)
        if avg > best_len:
            best_len, desc_col = avg, idx

    if desc_col is None:
        return None

    rows = [
        {"date": _cell(r, date_col), "description": _cell(r, desc_col),
         "amount": _cell(r, amount_col)}
        for r in table
        if _looks_like_date(_cell(r, date_col)) and _looks_like_money(_cell(r, amount_col))
    ]
    return pd.DataFrame(rows) if rows else None


def load_pdf(filepath: str | Path) -> pd.DataFrame:
    """
    Main entry point. Accepts a path to a bank statement PDF, returns a clean
    DataFrame with columns: date, description, amount, transaction_type.
    """
    import pdfplumber

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    logger.info(f"Loading PDF: {filepath.name}")

    tables = []
    n_pages = 0
    with pdfplumber.open(filepath) as pdf:
        n_pages = len(pdf.pages)
        for page in pdf.pages:
            tables.extend(page.extract_tables() or [])

    logger.info(f"  Extracted {len(tables)} table(s) from {n_pages} page(s)")

    raw_df = tables_to_dataframe(tables)
    logger.info(f"  Raw shape: {raw_df.shape} | Columns: {list(raw_df.columns)}")

    return clean_transactions(raw_df)
