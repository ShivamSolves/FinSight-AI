"""
tests/test_metrics.py — Unit tests for the answer-parsing metrics (pure, offline).
"""

from finsight.evaluation.metrics import (
    answer_matches,
    answer_total,
    extract_rupee_amounts,
)


def test_extract_rupee_amounts_handles_commas_and_decimals():
    text = "**Answer:** You spent ₹4,519.00 in total."
    assert extract_rupee_amounts(text) == [4519.0]


def test_extract_multiple_amounts():
    text = "₹910.00 across ₹420.00 and ₹490.00"
    assert extract_rupee_amounts(text) == [910.0, 420.0, 490.0]


def test_answer_total_prefers_answer_line():
    answer = (
        "**Answer:** ₹6,070.00\n\n"
        "**Transactions used:**\n"
        "- 2026-02-02 | Dmart | ₹1,980.00\n"
    )
    assert answer_total(answer) == 6070.0


def test_answer_total_falls_back_to_max():
    answer = "You spent ₹420.00 then ₹490.00, total ₹910.00"
    assert answer_total(answer) == 910.0


def test_answer_total_none_when_no_amount():
    assert answer_total("I cannot determine this from the available transactions.") is None


def test_answer_matches():
    assert answer_matches("**Answer:** ₹910.00", 910.0)
    assert not answer_matches("**Answer:** ₹900.00", 910.0)
