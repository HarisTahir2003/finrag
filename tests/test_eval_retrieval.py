"""Retrieval scoring logic, driven by a stub store.

Scoring is tested independently of any real embedding model so these run fast
and deterministically. The live path is exercised by `finrag eval retrieval
--fixtures`, which CI runs as the quality gate.
"""

from __future__ import annotations

from dataclasses import replace

from langchain_core.documents import Document

from finrag.config import get_settings
from finrag.eval.retrieval_eval import evaluate_case, evaluate_retrieval
from finrag.eval.schema import RetrievalCase


class StubStore:
    """Returns canned documents and records the filter it was asked for."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.last_filter = None

    def similarity_search(self, query, k=20, filter=None):  # noqa: A002 - matches the real signature
        self.last_filter = filter
        return self.docs[:k]


def doc(text: str, ticker: str = "AAPL", year: int = 2023) -> Document:
    return Document(page_content=text, metadata={"ticker": ticker, "year": year, "id": text[:20]})


CASE = RetrievalCase(
    id="aapl-current-assets",
    query="total current assets",
    ticker="AAPL",
    fiscal_year=2023,
    expect_any=["143,566"],
)


def test_hit_records_the_rank():
    # rerank off: this asserts the rank the *retriever* produced, and a
    # cross-encoder would reorder the two documents out from under it.
    store = StubStore([doc("nothing here"), doc("| Total current assets | 143,566 |")])
    result = evaluate_case(CASE, store=store, settings=replace(get_settings(), rerank=False))
    assert result.hit
    assert result.rank == 2
    assert result.matched == "143,566"
    assert result.reciprocal_rank == 0.5


def test_miss_when_the_figure_is_absent():
    result = evaluate_case(CASE, store=StubStore([doc("unrelated text")] * 3))
    assert not result.hit
    assert result.rank is None
    assert result.reciprocal_rank == 0.0


def test_empty_retrieval_is_a_miss_with_no_chunks():
    result = evaluate_case(CASE, store=StubStore([]))
    assert not result.hit
    assert result.retrieved == 0


def test_counts_chunks_from_the_wrong_filing():
    """The regression guard. Under the old indexing these would be tagged 2024."""
    store = StubStore([doc("143,566", year=2023), doc("other", year=2024), doc("more", year=2024)])
    result = evaluate_case(CASE, store=store)
    assert result.hit
    assert result.wrong_filing == 2


def test_search_is_filtered_to_the_requested_filing():
    store = StubStore([doc("143,566")])
    evaluate_case(CASE, store=store)
    assert store.last_filter == {"$and": [{"ticker": "AAPL"}, {"year": 2023}]}


def test_report_aggregates():
    cases = [
        CASE,
        RetrievalCase(id="miss", query="q", ticker="AAPL", fiscal_year=2023, expect_any=["zzz"]),
    ]
    report = evaluate_retrieval(cases, store=StubStore([doc("143,566")]))
    assert report.hit_rate == 0.5
    assert report.mrr == 0.5
    assert report.filter_accuracy == 1.0
    assert [r.case.id for r in report.failures()] == ["miss"]


def test_filter_accuracy_drops_when_years_are_wrong():
    store = StubStore([doc("143,566", year=2023), doc("x", year=2019)])
    report = evaluate_retrieval([CASE], store=store)
    assert report.filter_accuracy == 0.5


def test_metrics_are_rounded_for_logging():
    report = evaluate_retrieval([CASE], store=StubStore([doc("143,566")]))
    metrics = report.as_metrics()
    assert metrics["hit_rate"] == 1.0
    assert metrics["cases"] == 1
    assert metrics["empty_retrievals"] == 0


def test_query_expansion_is_off_by_default():
    """It halved hit rate on the real corpus; see retrieval.expand_query."""
    from finrag.config import Settings

    assert Settings().query_expansion is False


def test_search_uses_the_raw_query_unless_expansion_is_enabled(monkeypatch):
    """The knob has to actually reach the vector store, not just exist."""
    from finrag.config import Settings
    from finrag.retrieval import search_filing

    seen: list[str] = []

    class _Store:
        def similarity_search(self, query, k, filter):  # noqa: A002 - langchain's name
            seen.append(query)
            return []

    base = Settings()
    search_filing("total net sales", "AAPL", 2024, store=_Store(), settings=base)
    assert seen[-1] == "total net sales", "default must send the question as asked"

    search_filing(
        "total net sales",
        "AAPL",
        2024,
        store=_Store(),
        settings=replace(base, query_expansion=True),
    )
    assert seen[-1].startswith("total net sales ")
    assert "balance sheet" in seen[-1], "enabling the flag must restore the expansion"
