"""Retrieval evaluation that needs no LLM.

This is the tier that can gate CI. Every judgement is exact substring matching
against a labelled dataset, so it is deterministic, free, and fails for a
reason you can read off the output rather than a judge model's opinion.

Three metrics:

``hit_rate``
    Fraction of queries where an expected string appeared anywhere in the
    retrieved passages. The headline number.

``mrr``
    Mean reciprocal rank of the first chunk containing an expected string.
    Distinguishes "found it first" from "found it fifteenth", which hit rate
    alone hides.

``filter_accuracy``
    Fraction of retrieved chunks whose ticker and fiscal year actually match
    what was asked for. This is the direct regression guard on the metadata
    bug: with the old accession-number indexing, a query for AMZN 2022 returned
    chunks tagged 2023, and this metric would read 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import Settings, get_settings
from ..retrieval import search_filing
from .schema import RetrievalCase, load_retrieval_cases

log = logging.getLogger(__name__)


@dataclass
class CaseResult:
    case: RetrievalCase
    hit: bool
    rank: int | None
    retrieved: int
    wrong_filing: int
    matched: str | None = None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank if self.rank else 0.0


@dataclass
class RetrievalReport:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return sum(r.hit for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mrr(self) -> float:
        return (
            sum(r.reciprocal_rank for r in self.results) / len(self.results)
            if self.results
            else 0.0
        )

    @property
    def filter_accuracy(self) -> float:
        total = sum(r.retrieved for r in self.results)
        wrong = sum(r.wrong_filing for r in self.results)
        return (total - wrong) / total if total else 0.0

    @property
    def empty_retrievals(self) -> int:
        return sum(1 for r in self.results if r.retrieved == 0)

    def as_metrics(self) -> dict[str, float]:
        return {
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "filter_accuracy": round(self.filter_accuracy, 4),
            "cases": len(self.results),
            "empty_retrievals": self.empty_retrievals,
        }

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.hit]

    def format_table(self) -> str:
        lines = [f"{'case':32} {'hit':>4} {'rank':>5} {'chunks':>7} {'wrong':>6}", "-" * 58]
        for r in self.results:
            lines.append(
                f"{r.case.id:32} {'yes' if r.hit else 'NO':>4} "
                f"{r.rank if r.rank else '-':>5} {r.retrieved:>7} {r.wrong_filing:>6}"
            )
        return "\n".join(lines)


def evaluate_case(case: RetrievalCase, store=None, settings: Settings | None = None) -> CaseResult:
    settings = settings or get_settings()
    found = search_filing(case.query, case.ticker, case.fiscal_year, store=store, settings=settings)

    wrong = sum(
        1
        for d in found.documents
        if d.metadata.get("ticker") != case.ticker.upper()
        or int(d.metadata.get("year", -1)) != case.fiscal_year
    )

    for position, doc in enumerate(found.documents, start=1):
        for needle in case.expect_any:
            if needle in doc.page_content:
                return CaseResult(case, True, position, len(found.documents), wrong, needle)

    return CaseResult(case, False, None, len(found.documents), wrong)


def evaluate_retrieval(
    cases: list[RetrievalCase] | None = None,
    store=None,
    settings: Settings | None = None,
) -> RetrievalReport:
    settings = settings or get_settings()
    cases = cases if cases is not None else load_retrieval_cases()

    if store is None:
        from ..ingest.index import open_store

        store = open_store(settings)

    report = RetrievalReport()
    for case in cases:
        result = evaluate_case(case, store=store, settings=settings)
        report.results.append(result)
        log.info(
            "%-32s %s%s",
            case.id,
            "hit" if result.hit else "MISS",
            f" at rank {result.rank}" if result.rank else "",
        )
    return report
