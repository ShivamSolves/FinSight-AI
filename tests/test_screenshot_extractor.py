"""
Tests for the screenshot extractor (finsight/ingestion/screenshot_extractor.py).

The vision call is the only networked part, so these tests exercise everything
around it offline: JSON parsing tolerance, UPI-specific description derivation,
sign/UTR/date handling, and the full bytes→DataFrame path with the vision call
monkeypatched. A separate integration-marked test hits real Gemini.
"""

import pytest

from finsight.ingestion import screenshot_extractor as se
from finsight.ingestion.screenshot_extractor import (
    ScreenshotExtractionError,
    _build_description,
    parse_transaction_json,
    transactions_from_records,
)

# ── parse_transaction_json ────────────────────────────────────────────────────

def test_parses_bare_array():
    text = '[{"amount": 100, "upi_id": "a@b", "type": "debit"}]'
    assert parse_transaction_json(text) == [{"amount": 100, "upi_id": "a@b", "type": "debit"}]


def test_parses_fenced_array():
    text = '```json\n[{"amount": 5}]\n```'
    assert parse_transaction_json(text) == [{"amount": 5}]


def test_wraps_single_object_in_list():
    assert parse_transaction_json('{"amount": 5}') == [{"amount": 5}]


def test_recovers_json_from_surrounding_text():
    text = 'Sure! Here you go: [{"amount": 7}] hope that helps'
    assert parse_transaction_json(text) == [{"amount": 7}]


def test_empty_and_junk_return_empty_list():
    assert parse_transaction_json("") == []
    assert parse_transaction_json("   ") == []
    assert parse_transaction_json("no json here") == []
    assert parse_transaction_json("[]") == []


def test_drops_non_dict_elements():
    assert parse_transaction_json('[{"amount": 1}, "junk", 5]') == [{"amount": 1}]


# ── _build_description ────────────────────────────────────────────────────────

def test_description_prefers_payee_name():
    assert _build_description({"payee_name": "Swiggy Ltd", "upi_id": "swiggy@paytm"}) == "Swiggy Ltd"


def test_description_falls_back_to_upi_local_part():
    # The local part is what the categorizer substring-matches on.
    assert _build_description({"payee_name": "", "upi_id": "swiggy@paytm"}) == "swiggy"


def test_description_falls_back_to_full_upi_when_no_local_part():
    assert _build_description({"upi_id": "@weird"}) == "@weird"


def test_description_last_resort_plain_field():
    assert _build_description({"description": "Coffee Day"}) == "Coffee Day"


# ── transactions_from_records ─────────────────────────────────────────────────

def test_debit_is_negative_credit_is_positive():
    df = transactions_from_records([
        {"amount": 100, "upi_id": "x@y", "type": "debit"},
        {"amount": 50, "upi_id": "z@y", "type": "credit"},
    ])
    assert list(df["amount"]) == [-100.0, 50.0]


def test_amount_defaults_to_debit_when_type_missing():
    df = transactions_from_records([{"amount": 20, "upi_id": "x@y"}])
    assert df["amount"].iloc[0] == -20.0


def test_strips_commas_and_takes_absolute_value():
    df = transactions_from_records([{"amount": "-1,249.50", "upi_id": "x@y", "type": "debit"}])
    assert df["amount"].iloc[0] == -1249.50


def test_missing_date_defaults_to_today():
    from datetime import date
    df = transactions_from_records([{"amount": 10, "upi_id": "x@y", "date": ""}])
    assert df["date"].iloc[0] == date.today().isoformat()


def test_dedupes_on_utr():
    df = transactions_from_records([
        {"amount": 10, "upi_id": "x@y", "utr": "SAME"},
        {"amount": 10, "upi_id": "x@y", "utr": "SAME"},
        {"amount": 20, "upi_id": "x@y", "utr": "OTHER"},
    ])
    assert len(df) == 2


def test_skips_records_without_amount_or_payee():
    df = transactions_from_records([
        {"upi_id": "x@y"},                 # no amount
        {"amount": 10},                     # no payee signal
        {"amount": 10, "upi_id": "x@y"},    # valid
    ])
    assert len(df) == 1


def test_skips_unparseable_amount():
    df = transactions_from_records([{"amount": "abc", "upi_id": "x@y"}])
    assert df.empty


# ── extract_transactions_from_image (vision mocked) ───────────────────────────

def _fake_vision(text):
    def _inner(image_bytes, mime_type):
        return text
    return _inner


def test_end_to_end_with_mocked_vision(monkeypatch):
    monkeypatch.setattr(se, "_call_vision", _fake_vision(
        '[{"amount": 1249.0, "upi_id": "swiggy@paytm", "payee_name": "Swiggy Limited",'
        ' "utr": "SBIN439218574321", "date": "2026-09-18", "type": "debit"}]'
    ))
    df = se.extract_transactions_from_image(b"\x89PNG...", "image/png")
    assert list(df.columns) == ["date", "description", "amount", "transaction_type"]
    assert df["description"].iloc[0] == "Swiggy Limited"
    assert df["amount"].iloc[0] == -1249.0
    assert df["transaction_type"].iloc[0] == "debit"
    assert str(df["date"].iloc[0].date()) == "2026-09-18"


def test_empty_image_raises(monkeypatch):
    monkeypatch.setattr(se, "_call_vision", _fake_vision("[]"))
    with pytest.raises(ScreenshotExtractionError):
        se.extract_transactions_from_image(b"")


def test_no_usable_transaction_raises(monkeypatch):
    monkeypatch.setattr(se, "_call_vision", _fake_vision("[]"))
    with pytest.raises(ScreenshotExtractionError):
        se.extract_transactions_from_image(b"\x89PNG...")


def test_extracted_row_categorizes(monkeypatch):
    """The UPI-derived description must flow through the real categorizer."""
    from finsight.categorization.categorizer import categorize_dataframe
    monkeypatch.setattr(se, "_call_vision", _fake_vision(
        '[{"amount": 399, "upi_id": "zeptonow@axis", "type": "debit", "date": "2026-09-18"}]'
    ))
    df = se.extract_transactions_from_image(b"\x89PNG...", "image/png")
    out = categorize_dataframe(df)
    assert out["category"].iloc[0] == "Groceries"


# ── real network (skipped by default) ─────────────────────────────────────────

@pytest.mark.integration
def test_real_vision_call_on_sample(tmp_path):
    """Hits real Gemini against a generated UPI screenshot. Needs GEMINI_API_KEY."""
    import os
    if not os.getenv("GEMINI_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    from PIL import Image, ImageDraw
    img = Image.new("RGB", (600, 400), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 40), "Payment Successful", fill="black")
    d.text((30, 110), "Rs 1,249.00", fill="black")
    d.text((30, 180), "UPI ID: swiggy@paytm", fill="black")
    d.text((30, 250), "UTR: SBIN439218574321", fill="black")
    path = tmp_path / "upi.png"
    img.save(path)

    df = se.extract_transactions_from_file(path)
    assert len(df) == 1
    assert abs(df["amount"].iloc[0]) == 1249.0
