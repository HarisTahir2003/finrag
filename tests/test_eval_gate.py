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


def test_llm_free_runs_are_not_labelled_with_a_chat_backend():
    """The retrieval suite runs no LLM, so its record must not name one.

    search_filing is called with apply_context_budget=False specifically so the
    retrieval metrics cannot move when the chat provider changes. Recording
    llm_backend against that run asserted the dependency the code avoids, and
    named the file `retrieval-groq-openai-gpt-oss-120b-...` -- which reads as a
    retrieval score measured on Groq.
    """
    from finrag.config import get_settings
    from finrag.eval.tracking import config_params

    settings = get_settings()
    with_llm = config_params(settings, uses_llm=True)
    without = config_params(settings, uses_llm=False)

    assert "llm_backend" in with_llm
    for key in ("llm_backend", "chat_model", "resolved_model", "llm_fallbacks"):
        assert key not in without, f"{key} describes the chat model, not the retriever"

    # The retriever's own settings must survive.
    assert "retrieval_k" in without
    assert "embedding_backend" in without
    assert "chunk_strategy" in without


def test_llm_free_result_filenames_carry_no_backend_slug(tmp_path):
    from finrag.config import get_settings
    from finrag.eval.tracking import config_params, track_run

    with track_run(
        "retrieval", config_params(get_settings(), uses_llm=False), results_dir=tmp_path
    ) as record:
        record({"hit_rate": 1.0})

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    name = written[0].name
    assert name.startswith("retrieval-"), name
    for token in ("groq", "vertex", "gpt-oss", "gemini"):
        assert token not in name, f"{token!r} should not appear in an LLM-free result name"
