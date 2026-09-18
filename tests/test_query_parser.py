"""
tests/test_query_parser.py — Unit tests for the natural-language query planner.

These are pure/offline: no vector store, no LLM, no network. They lock in the
parsing rules that make retrieval complete and correctly filtered.
"""

import pytest

from finsight.rag.query_parser import build_where, parse_query

# ── Month parsing ─────────────────────────────────────────────────────────────

def test_month_with_year_becomes_qualified_string():
    p = parse_query("What was my total grocery spend in February 2026?")
    assert p.months == ["2026-02"]
    assert p.month_nums == []


def test_month_without_year_becomes_month_num():
    p = parse_query("Show my rent in January")
    assert p.months == []
    assert p.month_nums == [1]


def test_multiple_months_no_year():
    p = parse_query("Income across January, February, and March")
    assert p.month_nums == [1, 2, 3]


def test_abbreviation_matches():
    p = parse_query("spend in Feb")
    assert p.month_nums == [2]


def test_all_months_phrase_suppresses_month_filter():
    p = parse_query("How much did I spend on subscriptions across all months?")
    assert p.months == []
    assert p.month_nums == []


@pytest.mark.parametrize("question", [
    "money spent at the market",     # 'mar' must NOT match inside 'market'
    "maybe I bought groceries",      # 'may' must NOT match inside 'maybe'
    "tickets for the march concert", # 'march' as a verb/noun edge — still a month token,
])
def test_word_boundary_month_matching(question):
    # The first two must not infer a month; documented here to catch regressions
    # where substring matching leaks.
    p = parse_query(question)
    if question == "tickets for the march concert":
        assert 3 in p.month_nums
    else:
        assert p.month_nums == []


# ── Category parsing ──────────────────────────────────────────────────────────

def test_category_groceries():
    assert parse_query("how much on groceries").categories == ["Groceries"]


def test_category_subscriptions_synonym():
    assert parse_query("my streaming subscriptions").categories == ["Subscriptions"]


def test_category_income():
    assert parse_query("what was my total income").categories == ["Income"]


def test_category_transport_synonym():
    assert parse_query("how much did I spend on fuel and cabs").categories == ["Transport"]


# ── Merchant parsing (beats category) ─────────────────────────────────────────

def test_merchant_detected():
    p = parse_query("How much did I spend at Swiggy in total?")
    assert p.merchant == "swiggy"


def test_merchant_takes_precedence_over_category():
    # "Swiggy" is also a Food & Dining trigger, but the specific merchant wins.
    p = parse_query("total spend on Swiggy")
    assert p.merchant == "swiggy"
    assert p.categories == []


def test_longer_brand_wins():
    p = parse_query("renew amazon prime membership")
    assert p.merchant == "amazon prime"


# ── Intent / completeness ─────────────────────────────────────────────────────

def test_aggregate_intent_sets_wants_all():
    p = parse_query("What is my total spending?")
    assert p.intent == "aggregate"
    assert p.wants_all is True


def test_filter_alone_implies_wants_all():
    p = parse_query("list my grocery transactions")
    assert p.has_filter is True
    assert p.wants_all is True


def test_open_ended_query_is_semantic_topk():
    p = parse_query("what did I buy")
    assert p.has_filter is False
    assert p.wants_all is False
    assert build_where(p) is None


# ── where-clause construction ─────────────────────────────────────────────────

def test_build_where_single_category():
    p = parse_query("groceries")
    assert build_where(p) == {"category": "Groceries"}


def test_build_where_category_and_month_num():
    p = parse_query("groceries in February")
    assert build_where(p) == {
        "$and": [{"category": "Groceries"}, {"month_num": 2}]
    }


def test_build_where_category_and_qualified_month():
    p = parse_query("groceries in February 2026")
    assert build_where(p) == {
        "$and": [{"category": "Groceries"}, {"month": "2026-02"}]
    }


def test_build_where_multiple_months_uses_in():
    p = parse_query("spending in January and February")
    assert build_where(p) == {"month_num": {"$in": [1, 2]}}


def test_build_where_merchant_only_is_none():
    # Merchant is a post-filter, not a metadata clause.
    p = parse_query("spend at Swiggy")
    assert build_where(p) is None


def test_scope_description_is_human_readable():
    p = parse_query("grocery spend in February 2026")
    scope = p.describe_scope()
    assert "Groceries" in scope
    assert "2026-02" in scope
