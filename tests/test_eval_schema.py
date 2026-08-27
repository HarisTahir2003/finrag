"""The evaluation sets are data now, not a notebook cell."""

from __future__ import annotations

from finrag.eval.schema import load_agent_cases, load_retrieval_cases


def test_retrieval_cases_load():
    cases = load_retrieval_cases()
    assert len(cases) >= 9
    assert all(c.expect_any for c in cases), "a case with nothing to match cannot fail"
    assert all(c.id and c.query and c.ticker for c in cases)


def test_retrieval_case_ids_are_unique():
    ids = [c.id for c in load_retrieval_cases()]
    assert len(ids) == len(set(ids))


def test_agent_suite_is_the_full_twenty():
    """All 20 questions were carried over from the original notebook cell."""
    cases = load_agent_cases()
    assert len(cases) == 20
    assert {c.ticker for c in cases} == {
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
        "NVDA",
        "NFLX",
        "JPM",
        "V",
    }


def test_agent_categories_are_balanced():
    from collections import Counter

    counts = Counter(c.category for c in load_agent_cases())
    assert counts == {"calc": 5, "narrative": 5, "mixed": 10}


def test_calculation_cases_expect_the_calculator():
    for case in load_agent_cases():
        if case.category == "calc":
            assert "calculator" in case.expected_tools
            assert "search_10k_reports" in case.expected_tools


def test_fiscal_years_are_plausible():
    assert all(2018 <= c.fiscal_year <= 2026 for c in load_agent_cases())


def test_every_agent_case_has_a_reference_answer():
    assert all(c.reference_answer.strip() for c in load_agent_cases())
