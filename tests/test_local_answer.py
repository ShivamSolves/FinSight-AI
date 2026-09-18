"""
tests/test_local_answer.py — Unit tests for the offline grounded-answer engine.
"""

from finsight.rag.local_answer import summarize_hits


def _hit(date, desc, amount, category, ttype):
    return {"metadata": {"date": date, "description": desc, "amount": amount,
                         "category": category, "transaction_type": ttype}}


def test_spending_sum_and_headline():
    hits = [
        _hit("2026-01-05", "Swiggy", -420.0, "Food & Dining", "debit"),
        _hit("2026-03-05", "Swiggy", -490.0, "Food & Dining", "debit"),
    ]
    a = summarize_hits(hits, complete=True)
    assert a.spend_total == 910.0
    assert a.headline_total == 910.0
    assert a.income_total == 0.0
    assert a.count == 2
    assert a.confidence == "High"


def test_income_only_headline_is_income():
    hits = [
        _hit("2026-01-03", "Salary", 62000.0, "Income", "credit"),
        _hit("2026-02-03", "Salary", 62000.0, "Income", "credit"),
    ]
    a = summarize_hits(hits, complete=True)
    assert a.income_total == 124000.0
    assert a.headline_total == 124000.0
    assert a.spend_total == 0.0


def test_net_and_mixed():
    hits = [
        _hit("2026-01-03", "Salary", 1000.0, "Income", "credit"),
        _hit("2026-01-05", "Rent", -400.0, "Rent/Housing", "debit"),
    ]
    a = summarize_hits(hits)
    assert a.net == 600.0
    assert a.spend_total == 400.0
    assert a.income_total == 1000.0
    assert a.headline_total == 400.0   # spend wins when present


def test_incomplete_set_is_medium_confidence():
    a = summarize_hits([_hit("2026-01-05", "Swiggy", -420.0, "Food & Dining", "debit")],
                       complete=False)
    assert a.confidence == "Medium"


def test_markdown_has_required_sections():
    hits = [_hit("2026-01-05", "Swiggy", -420.0, "Food & Dining", "debit")]
    md = summarize_hits(hits, complete=True).to_markdown("How much at Swiggy?")
    assert "**Answer:**" in md
    assert "**Transactions used:**" in md
    assert "**Confidence:**" in md
    assert "₹420.00" in md


def test_evidence_sorted_biggest_spend_first():
    hits = [
        _hit("2026-01-05", "Small", -100.0, "Shopping", "debit"),
        _hit("2026-01-06", "Big", -900.0, "Shopping", "debit"),
    ]
    a = summarize_hits(hits)
    assert a.evidence[0]["metadata"]["description"] == "Big"


def test_empty_hits():
    a = summarize_hits([])
    assert a.count == 0
    assert a.headline_total == 0.0
    assert a.transactions_md() == "_None_"
