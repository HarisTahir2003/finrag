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


def test_answer_text_flattens_gemini_content_blocks():
    """Gemini returns content blocks, and a thinking model adds a signature one.

    Printed raw, that put a Python repr of a list of dicts on the terminal in
    place of a sentence. Worse for scoring: a substring check for a figure
    would have been searching a few hundred characters of base64 too.
    """
    from finrag.agent import answer_text

    blocks = [
        {
            "type": "text",
            "text": "Apple's total net sales in fiscal 2024 was $391,035 million.",
            "thought_signature": "Ci8Bjz1rX6OkVnGjWG3EssK4XaXByvEPAQB8pobjsI0N",
        }
    ]
    out = answer_text(blocks)
    assert out == "Apple's total net sales in fiscal 2024 was $391,035 million."
    assert "thought_signature" not in out
    assert "Ci8Bjz1rX" not in out, "the signature must not reach a scored answer"


def test_answer_text_passes_plain_strings_through():
    """Groq and the OpenAI-compatible backends already return a string."""
    from finrag.agent import answer_text

    assert answer_text("  Net sales were $391,035 million.  ") == (
        "Net sales were $391,035 million."
    )


def test_answer_text_joins_multiple_blocks():
    from finrag.agent import answer_text

    assert answer_text([{"text": "Revenue rose. "}, {"text": "Margins held."}]) == (
        "Revenue rose. Margins held."
    )
