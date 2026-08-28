"""The dataset validator, tested against the defects that motivated it.

Each of these reproduces a real case that shipped, scored, and quietly moved a
headline metric without ever raising an error.
"""

from __future__ import annotations

import pytest

from finrag.eval.schema import AgentCase, RetrievalCase, reference_figures
from finrag.eval.validate import ValidationReport, _leading_percentage, validate_against_corpus


class FakeStore:
    """Just enough Chroma to answer `get(where=..., include=["documents"])`."""

    def __init__(self, documents_by_filing: dict[tuple[str, int], list[str]]):
        self._docs = documents_by_filing

    def get(self, where, include):  # noqa: A002 - chroma's signature
        conditions = {list(c.keys())[0]: list(c.values())[0] for c in where["$and"]}
        key = (conditions["ticker"], conditions["year"])
        return {"documents": self._docs.get(key, [])}


def _run(store, retrieval=(), agent=(), monkeypatch=None):
    monkeypatch.setattr("finrag.eval.validate.load_retrieval_cases", lambda: list(retrieval))
    monkeypatch.setattr("finrag.eval.validate.load_agent_cases", lambda: list(agent))
    return validate_against_corpus(store=store)


def test_flags_a_probe_whose_target_is_not_in_the_filing(monkeypatch):
    """The regionalization bug: expects a word no Amazon 10-K ever contained."""
    store = FakeStore({("AMZN", 2022): ["We operate fulfillment networks worldwide."]})
    case = RetrievalCase(
        id="amzn-2022-fulfillment",
        query="why did management change the fulfillment network",
        ticker="AMZN",
        fiscal_year=2022,
        expect_any=["regionalization"],
    )
    report = _run(store, retrieval=[case], monkeypatch=monkeypatch)

    assert not report.ok
    assert "none of expect_any" in report.failures[0].message


def test_accepts_a_probe_where_only_one_alternative_matches(monkeypatch):
    """expect_any is a disjunction, not a conjunction.

    amzn-2022-net-loss offers "(2,722)" and "2,722" precisely because the sign
    convention varies between the statement and the discussion. Requiring both
    was this checker's own version of the bug it exists to catch.
    """
    store = FakeStore({("AMZN", 2022): ["Net loss of 2,722 for the year."]})
    case = RetrievalCase(
        id="amzn-2022-net-loss",
        query="net loss",
        ticker="AMZN",
        fiscal_year=2022,
        expect_any=["(2,722)", "2,722"],
    )
    assert _run(store, retrieval=[case], monkeypatch=monkeypatch).ok


def test_flags_a_reference_quoting_a_figure_the_filing_never_reports(monkeypatch):
    """The MSFT total-debt error: 47,193 where the filing supports 47,237."""
    store = FakeStore(
        {("MSFT", 2023): ["Current portion of long-term debt 5,247 Long-term debt 41,990"]}
    )
    case = AgentCase(
        id="calc-msft-2023-04",
        question="debt to equity",
        ticker="MSFT",
        fiscal_year=2023,
        reference_answer="0.229 (total debt $47,193M / total equity $206,223M)",
    )
    report = _run(store, agent=[case], monkeypatch=monkeypatch)

    assert not report.ok
    assert "47,193" in report.failures[0].message


def test_derived_figures_exempt_a_computed_value(monkeypatch):
    """An average legitimately appears nowhere: (5,282 + 5,159) / 2 = 5,220.5."""
    store = FakeStore({("NVDA", 2024): ["Inventories 5,282 5,159 cost of revenue 16,621"]})
    case = AgentCase(
        id="calc-nvda-2024-05",
        question="inventory turnover",
        ticker="NVDA",
        fiscal_year=2024,
        reference_answer="3.18x (COGS $16,621M / average inventory $5,221M)",
        derived_figures=["5,221"],
    )
    assert _run(store, agent=[case], monkeypatch=monkeypatch).ok


def test_warns_when_the_filing_states_the_answer_a_case_wants_computed(monkeypatch):
    """The MSFT-16 defect: the agent read a stated figure and was marked failed."""
    store = FakeStore(
        {("MSFT", 2023): ["Microsoft Cloud revenue increased 22% to $111.6 billion."]}
    )
    case = AgentCase(
        id="mixed-msft-2023-16",
        question="cloud revenue growth rate",
        ticker="MSFT",
        fiscal_year=2023,
        expected_tools=["search_10k_reports", "calculator"],
        reference_answer="22% growth to about $111B, attributed to a structural shift.",
    )
    report = _run(store, agent=[case], monkeypatch=monkeypatch)

    assert report.ok, "a stated answer is suspect, not broken"
    assert len(report.warnings) == 1
    assert "states the answer" in report.warnings[0].message


def test_does_not_warn_for_a_ratio_the_filing_never_prints(monkeypatch):
    """Ratios are computed, so their inputs being present is the normal case.

    An earlier version fired whenever a quoted input appeared in the filing,
    which flagged every legitimate calculation case -- their inputs are stated,
    and computing from them is the entire point.
    """
    store = FakeStore({("MSFT", 2023): ["Total stockholders' equity 206,223 debt 5,247 41,990"]})
    case = AgentCase(
        id="calc-msft-2023-04",
        question="debt to equity",
        ticker="MSFT",
        fiscal_year=2023,
        expected_tools=["search_10k_reports", "calculator"],
        reference_answer="0.229 (total debt $47,237M / total equity $206,223M)",
        derived_figures=["47,237"],
    )
    report = _run(store, agent=[case], monkeypatch=monkeypatch)

    assert report.ok
    assert not report.warnings


def test_flags_a_filing_missing_from_the_index(monkeypatch):
    case = RetrievalCase(
        id="ghost", query="q", ticker="AAPL", fiscal_year=1999, expect_any=["anything"]
    )
    report = _run(FakeStore({}), retrieval=[case], monkeypatch=monkeypatch)

    assert not report.ok
    assert "not in the index" in report.failures[0].message


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("22% growth to about $111B", "22%"),
        ("29.49% (2023 $106,618M vs 2022 $82,338M)", "29.49%"),
        ("-0.53% (net loss $2,722M / net sales $513,983M)", "-0.53%"),
        ("0.229 (total debt $47,237M / total equity $206,223M)", None),
        ("3.18x (COGS $16,621M / average inventory $5,221M)", None),
        ("", None),
    ],
)
def test_leading_percentage_finds_only_result_first_percentages(reference, expected):
    assert _leading_percentage(reference) == expected


def test_reference_figures_ignores_years_and_bare_numbers():
    """A thousands separator is what makes a token a reported figure.

    Without that rule "2023" counted as a figure, and the oracle-context
    selector that shares this helper pulled in fifty chunks of an unrelated
    filing.
    """
    figures = reference_figures("0.988 in 2023 (assets $143,566M / liabilities $145,308M)")
    assert figures == {"143,566", "145,308"}


def test_report_add_records_fatality():
    report = ValidationReport()
    report.add("a", "agent", "broken")
    report.add("b", "agent", "suspect", fatal=False)

    assert len(report.failures) == 1
    assert len(report.warnings) == 1
    assert not report.ok
