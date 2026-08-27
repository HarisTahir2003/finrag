"""The CI gate.

Scored without an LLM so it can run on a fork with no credentials. These tests
drive it with synthetic reports rather than a live index.
"""

from __future__ import annotations

from finrag.eval.gate import Thresholds, check
from finrag.eval.retrieval_eval import CaseResult, RetrievalReport
from finrag.eval.schema import RetrievalCase

CASE = RetrievalCase(id="x", query="q", ticker="AAPL", fiscal_year=2023, expect_any=["1"])


def report_from(specs) -> RetrievalReport:
    """specs: list of (hit, rank, retrieved, wrong_filing)."""
    return RetrievalReport(
        results=[
            CaseResult(CASE, hit, rank, retrieved, wrong) for hit, rank, retrieved, wrong in specs
        ]
    )


def test_all_hits_at_rank_one_passes():
    result = check(report_from([(True, 1, 5, 0)] * 10))
    assert result.passed
    assert result.metrics["hit_rate"] == 1.0
    assert result.metrics["mrr"] == 1.0


def test_low_hit_rate_fails():
    result = check(report_from([(True, 1, 5, 0)] * 5 + [(False, None, 5, 0)] * 5))
    assert not result.passed
    assert any("hit_rate" in f for f in result.failures)


def test_wrong_filing_fails_even_when_every_query_hits():
    """The regression guard: retrieving the right text from the wrong year is still wrong."""
    result = check(report_from([(True, 1, 10, 3)] * 10))
    assert not result.passed
    assert any("filter_accuracy" in f for f in result.failures)
    assert any("wrong company or fiscal year" in f for f in result.failures)


def test_empty_retrieval_fails():
    result = check(report_from([(True, 1, 5, 0)] * 9 + [(False, None, 0, 0)]))
    assert not result.passed
    assert any("retrieved nothing" in f for f in result.failures)


def test_mrr_rewards_earlier_hits():
    early = report_from([(True, 1, 5, 0)] * 4)
    late = report_from([(True, 8, 10, 0)] * 4)
    assert early.mrr > late.mrr
    assert check(late, Thresholds(min_mrr=0.5)).passed is False


def test_thresholds_are_configurable():
    weak = report_from([(True, 1, 5, 0)] * 6 + [(False, None, 5, 0)] * 4)
    assert not check(weak).passed
    assert check(weak, Thresholds(min_hit_rate=0.5, min_mrr=0.4)).passed


def test_gate_output_names_the_breach():
    text = check(report_from([(False, None, 5, 0)] * 10)).format()
    assert "QUALITY GATE FAILED" in text
    assert "hit_rate" in text
