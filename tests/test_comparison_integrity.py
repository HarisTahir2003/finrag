"""Regression tests for ways a cross-backend benchmark can be silently wrong.

Every test here corresponds to a defect an adversarial audit of this codebase
actually found. They matter more than most tests because none of these bugs
crash: each one produces a plausible table of numbers that does not mean what
it appears to mean, which is the worst failure mode a benchmark can have.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from langchain_core.documents import Document

from finrag.config import get_settings
from finrag.eval.agent_eval import AgentCaseResult, AgentReport
from finrag.eval.compare import shared_context_budget
from finrag.eval.schema import AgentCase

CASE = AgentCase(
    id="c1",
    question="What is the current ratio?",
    ticker="AAPL",
    fiscal_year=2023,
    expected_tools=["search_10k_reports", "calculator"],
)


# ---- the context budget must not vary with the backend ----------------------


def test_context_budget_varies_by_backend_when_left_on_auto(monkeypatch):
    """The defect this guards. Documents the behaviour the sweep must override."""
    monkeypatch.setenv("FINRAG_MAX_CONTEXT_TOKENS", "auto")
    base = get_settings()
    budgets = {
        b: replace(base, llm_backend=b).max_context_tokens
        for b in ("groq", "cerebras", "github", "vertex")
    }
    assert budgets["cerebras"] == 6000
    assert budgets["groq"] == 2000
    assert budgets["vertex"] == 0
    assert len(set(budgets.values())) > 1, "auto really does differ per backend"


def test_sweep_resolves_to_the_most_restrictive_budget(monkeypatch):
    """Identical context for every row, and no request over the tightest ceiling."""
    monkeypatch.setenv("FINRAG_MAX_CONTEXT_TOKENS", "auto")
    settings = get_settings()
    budget = shared_context_budget(["groq", "cerebras", "vertex"], settings)
    assert budget == 2000, "groq's 8,000 TPM is the tightest ceiling in this sweep"

    # And with groq out, the next tightest takes over rather than going unlimited.
    assert shared_context_budget(["cerebras", "vertex"], settings) == 6000

    pinned = [
        replace(settings, llm_backend=b, max_context_tokens_raw=str(budget)).max_context_tokens
        for b in ("groq", "cerebras", "vertex")
    ]
    assert len(set(pinned)) == 1, f"rows still differ: {pinned}"


def test_budget_is_unlimited_when_no_backend_constrains_it(monkeypatch):
    monkeypatch.setenv("FINRAG_MAX_CONTEXT_TOKENS", "auto")
    assert shared_context_budget(["anthropic", "vertex"], get_settings()) == 0


def test_ollama_constrains_the_sweep_by_its_context_window(monkeypatch):
    """Ollama truncates silently rather than erroring, so an overflow looks like
    the model ignoring its context. Including it must tighten the shared budget."""
    monkeypatch.setenv("FINRAG_MAX_CONTEXT_TOKENS", "auto")
    monkeypatch.setenv("FINRAG_OLLAMA_NUM_CTX", "16384")
    settings = get_settings()
    assert replace(settings, llm_backend="ollama").max_context_tokens == 6553
    assert shared_context_budget(["anthropic", "vertex", "ollama"], settings) == 6553


def test_explicit_budget_is_respected_by_the_sweep(monkeypatch):
    monkeypatch.setenv("FINRAG_MAX_CONTEXT_TOKENS", "3000")
    assert shared_context_budget(["groq", "cerebras"], get_settings()) == 3000


# ---- attribution: a row must contain only its own backend's answers ---------


def test_sweep_disables_fallbacks_and_base_url_override(monkeypatch, tmp_path):
    """A fallback chain would let a row labelled `groq` hold another provider's
    answers; a stray base URL would collapse every preset onto one endpoint."""
    from unittest.mock import patch

    from finrag.eval.compare import compare_backends

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FINRAG_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("FINRAG_LLM_FALLBACKS", "cerebras,openrouter")
    monkeypatch.setenv("FINRAG_OPENAI_BASE_URL", "http://localhost:8000/v1")
    settings = get_settings()
    assert settings.llm_fallbacks and settings.openai_base_url, "precondition"

    seen = []

    def fake_build_agent(store=None, settings=None, verbose=False):
        seen.append(settings)

        class A:
            def invoke(self, _):
                return {"output": "0.988", "intermediate_steps": []}

        return A()

    with patch("finrag.agent.build_agent", fake_build_agent):
        compare_backends(
            ["groq", "cerebras"], suite="agent", cases=[CASE], store=object(), settings=settings
        )

    assert seen, "agent was never built"
    for s in seen:
        assert s.llm_fallbacks == "", "fallbacks must be off inside a comparison"
        assert s.openai_base_url == "", "base URL override must be cleared"


# ---- output-token parity across providers -----------------------------------


def test_ollama_receives_the_output_budget(monkeypatch):
    """Ollama calls it num_predict and defaults to ~128 tokens. Dropping
    max_tokens without translating it truncated local models to a fraction of
    the space every other backend got."""
    pytest.importorskip("langchain_ollama")
    from finrag.llm import get_chat_model

    monkeypatch.setenv("FINRAG_LLM_BACKEND", "ollama")
    monkeypatch.setenv("FINRAG_MAX_OUTPUT_TOKENS", "4000")
    monkeypatch.delenv("FINRAG_CHAT_MODEL", raising=False)
    assert get_chat_model(get_settings()).num_predict == 4000


def test_retry_policy_is_uniform(monkeypatch):
    """`errors` is a ranked column, so an uneven retry policy would score client
    libraries rather than models."""
    pytest.importorskip("langchain_groq")
    pytest.importorskip("langchain_openai")
    from finrag.llm import MAX_RETRIES, get_chat_model

    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("CEREBRAS_API_KEY", "k")
    monkeypatch.delenv("FINRAG_CHAT_MODEL", raising=False)

    monkeypatch.setenv("FINRAG_LLM_BACKEND", "groq")
    groq = get_chat_model(get_settings())
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "cerebras")
    cerebras = get_chat_model(get_settings())

    assert groq.max_retries == cerebras.max_retries == MAX_RETRIES


# ---- scoring must not reward non-answers ------------------------------------


def test_iteration_limit_is_not_a_tool_path_success():
    """The agent calls both tools, then loops until it is cut off. The trace
    looks perfect; nothing was answered."""
    r = AgentCaseResult(
        case=CASE,
        answer="Agent stopped due to iteration limit or time limit.",
        tools_called=["search_10k_reports", "calculator"],
    )
    assert r.gave_up
    assert not r.tool_path_ok
    assert not r.used_calculator_when_required
    assert not r.cited_a_figure


def test_errored_case_scores_zero_on_every_rate():
    """A case killed by a quota must not score better than one that ran."""
    r = AgentCaseResult(case=CASE, answer="", error="429 rate limited")
    assert not r.tool_path_ok
    assert not r.used_calculator_when_required, "an unreached case is not compliant"


def test_malformed_tool_calls_are_counted_not_credited():
    """handle_parsing_errors surfaces bad tool calls as a pseudo-tool. Counting
    it as a real call would hide the failure that most separates providers."""
    r = AgentCaseResult(
        case=CASE,
        answer="0.988",
        tools_called=["search_10k_reports", "_Exception", "calculator"],
    )
    assert r.malformed_tool_calls == 1
    report = AgentReport(results=[r])
    assert report.as_metrics()["malformed_tool_calls"] == 1


@pytest.mark.parametrize(
    "answer",
    [
        "Which company did you mean?",
        "Please provide the ticker symbol.",
        "I need a ticker to continue.",
    ],
)
def test_real_clarification_requests_are_detected(answer):
    assert AgentCaseResult(case=CASE, answer=answer).asked_for_clarification


@pytest.mark.parametrize(
    "answer",
    [
        "I established which company and which fiscal year were required, then answered: 0.988.",
        "The filing does not state which company acquired the subsidiary.",
    ],
)
def test_narration_is_not_mistaken_for_a_clarification_request(answer):
    """The system prompt hands the model this vocabulary, so an unanchored match
    would score verbosity rather than a failure to extract the ticker."""
    assert not AgentCaseResult(case=CASE, answer=answer).asked_for_clarification


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("The current ratio is 0.988", True),
        ("Total current assets were $143,566 million", True),
        ("R&D rose 14%", True),
        ("I could not find any figure in section 7", False),
        ("The filing does not disclose this.", False),
    ],
)
def test_figure_detection_requires_a_real_quantity(answer, expected):
    """Matching any digit made this read ~100% for every backend, including ones
    that found nothing -- no signal, yet used as a ranking tie-breaker."""
    assert AgentCaseResult(case=CASE, answer=answer).cited_a_figure is expected


# ---- the LLM-free suite must stay LLM-free ----------------------------------


class StubStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search(self, query, k=20, filter=None):  # noqa: A002
        return self.docs[:k]


def test_retrieval_metrics_do_not_move_with_the_chat_backend(monkeypatch):
    """The CI gate has no LLM in it. If the context budget applied here, a
    provider switch and a retrieval regression would look identical."""
    from finrag.eval.retrieval_eval import evaluate_case
    from finrag.eval.schema import RetrievalCase

    monkeypatch.setenv("FINRAG_MAX_CONTEXT_TOKENS", "auto")
    # Twelve fat chunks: a 6000-token budget would drop most of them.
    docs = [
        Document(page_content="x" * 3000, metadata={"ticker": "AAPL", "year": 2023, "id": f"d{i}"})
        for i in range(11)
    ] + [Document(page_content="143,566", metadata={"ticker": "AAPL", "year": 2023, "id": "hit"})]
    case = RetrievalCase(
        id="late-hit", query="q", ticker="AAPL", fiscal_year=2023, expect_any=["143,566"]
    )

    ranks = {}
    for backend in ("groq", "cerebras"):
        monkeypatch.setenv("FINRAG_LLM_BACKEND", backend)
        result = evaluate_case(case, store=StubStore(docs), settings=get_settings())
        ranks[backend] = (result.hit, result.rank)

    assert ranks["groq"] == ranks["cerebras"], f"retrieval score moved with the LLM: {ranks}"
    assert ranks["groq"][0], "the hit at rank 12 must still be found"


# ---- cache namespacing ------------------------------------------------------


def test_cache_is_namespaced_per_backend_and_model(monkeypatch, tmp_path):
    """LangChain keys the cache on whatever the client reports, and ChatOllama
    reports the same string for every model it serves. A shared cache file would
    hand one model's answers back under another model's name."""
    from finrag.cache import cache_path

    monkeypatch.setenv("FINRAG_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "ollama")
    base = get_settings()
    paths = {
        cache_path(replace(base, chat_model="qwen3:4b")),
        cache_path(replace(base, chat_model="llama3.3:70b")),
        cache_path(replace(base, llm_backend="groq", chat_model="")),
    }
    assert len(paths) == 3, "cache files collide across models or backends"


def test_ollama_clients_really_do_share_a_cache_key(monkeypatch):
    """Documents why the namespacing above is necessary rather than defensive."""
    pytest.importorskip("langchain_ollama")
    from finrag.llm import get_chat_model

    monkeypatch.setenv("FINRAG_LLM_BACKEND", "ollama")
    base = get_settings()
    a = get_chat_model(replace(base, chat_model="qwen3:4b"))._get_llm_string()
    b = get_chat_model(replace(base, chat_model="llama3.3:70b"))._get_llm_string()
    assert a == b, "if this ever differs, ChatOllama started keying by model"


# ---- checkpoints must not collide across case sets --------------------------


def test_checkpoints_separate_datasets():
    """`--dataset smoke` and the default suite are different questions; sharing
    a checkpoint file let one satisfy the other."""
    from finrag.eval.compare import _checkpoint_for

    default = _checkpoint_for("agent", "groq", "llama-3.3-70b-versatile", "default")
    smoke = _checkpoint_for("agent", "groq", "llama-3.3-70b-versatile", "smoke")
    assert default != smoke


# ---- provenance -------------------------------------------------------------


def test_table_states_the_corpus_and_case_set(monkeypatch, tmp_path):
    """A sweep over three toy fixtures produced a table indistinguishable from
    one over fifty real 10-Ks."""
    from unittest.mock import patch

    from finrag.eval.compare import compare_backends

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FINRAG_DATA_ROOT", str(tmp_path))

    class A:
        def invoke(self, _):
            return {"output": "0.988", "intermediate_steps": []}

    with patch("finrag.agent.build_agent", lambda **kw: A()):
        report = compare_backends(
            ["groq"],
            suite="agent",
            cases=[CASE],
            store=object(),
            settings=get_settings(),
            dataset="smoke",
            corpus="fixtures",
        )
    md = report.as_markdown()
    assert "dataset=smoke" in md
    assert "corpus=fixtures" in md
    assert report.config.get("llm_backend"), "full settings snapshot missing"
    assert "wall clock" in md and "rate limiting" in md, "timing must not read as provider speed"


def test_unpinned_judge_caveat_travels_with_the_table():
    """The stdout warning scrolls away; the table is what gets pasted."""
    from finrag.eval.compare import BackendResult, ComparisonReport

    report = ComparisonReport(
        suite="ragas",
        rank_key="ragas_faithfulness",
        fixed={"judge": "SELF (not pinned)"},
        results=[BackendResult("groq", "m", {"ragas_faithfulness": 0.8})],
    )
    assert "each backend scored its own answers" in report.as_markdown()


# ---- re-indexing replaces rather than accumulates ---------------------------


def test_reindexing_with_new_chunking_replaces_the_old(monkeypatch, tmp_path):
    """Chunk ids hash the chunk text, so a chunk_size change yields a disjoint
    id set and an upsert leaves the previous chunking behind -- the same filing
    indexed twice at two granularities."""
    from pathlib import Path

    # This one actually embeds, unlike the rest of the file. The unit-test CI
    # job installs only [dev] to stay light -- the quality gate is the job that
    # carries [local] -- so without this guard the whole matrix fails on a
    # missing sentence-transformers rather than on anything to do with the code.
    pytest.importorskip("langchain_huggingface")

    from finrag.ingest.index import collection_size, index_filings

    monkeypatch.setenv("FINRAG_CHUNK_STRATEGY", "recursive")
    base = replace(get_settings(), data_root=tmp_path)
    paths = sorted(
        (Path(__file__).parent / "fixtures" / "sec-edgar-filings").glob("**/full-submission.txt")
    )

    coarse = replace(base, chunk_size=3000, chunk_overlap=300)
    index_filings(paths=paths, settings=coarse)
    n_coarse = collection_size(coarse)

    fine = replace(base, chunk_size=400, chunk_overlap=60)
    index_filings(paths=paths, settings=fine)
    n_fine = collection_size(fine)

    index_filings(paths=paths, settings=fine)
    assert collection_size(fine) == n_fine, "identical re-run must be idempotent"
    assert n_fine != n_coarse + n_fine, "the coarse chunking was left behind"


# ---- resume must not launder away earlier failures --------------------------


class FlakyAgent:
    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def invoke(self, _):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("429 daily quota exhausted")
        return {"output": "0.988", "intermediate_steps": []}


def test_resume_still_discloses_earlier_failures(tmp_path):
    """`errors` counts only the current attempt. Resume a quota-starved run
    until it completes and it would otherwise publish a clean sheet -- and since
    you retry the flaky free tiers and run the reliable ones once, the column
    ended up reading backwards."""
    from finrag.eval.agent_eval import evaluate_agent

    cases = [
        AgentCase(id=f"c{i}", question=f"q{i}", ticker="AAPL", fiscal_year=2023) for i in range(4)
    ]
    path = tmp_path / "cases.jsonl"

    first = evaluate_agent(cases=cases, agent=FlakyAgent(fail_after=2), checkpoint_path=path)
    assert first.as_metrics()["errors"] == 2

    second = evaluate_agent(cases=cases, agent=FlakyAgent(), checkpoint_path=path).as_metrics()
    assert second["errors"] == 0, "this attempt genuinely had none"
    assert second["errors_all_attempts"] == 2, "but the run took two attempts to get there"
    assert second["attempts"] == 2


def test_failed_cases_are_retried_not_frozen(tmp_path):
    """Recording a failure must not mark the case done."""
    from finrag.eval.agent_eval import evaluate_agent

    cases = [
        AgentCase(id=f"c{i}", question=f"q{i}", ticker="AAPL", fiscal_year=2023) for i in range(3)
    ]
    path = tmp_path / "cases.jsonl"
    evaluate_agent(cases=cases, agent=FlakyAgent(fail_after=1), checkpoint_path=path)

    second = FlakyAgent()
    evaluate_agent(cases=cases, agent=second, checkpoint_path=path)
    assert second.calls == 2, "the two failures must run again, the success must not"


def test_error_count_is_shown_with_its_denominator():
    """`errors 6` reads very differently against 6 cases than against 20."""
    from finrag.eval.compare import BackendResult, ComparisonReport

    report = ComparisonReport(
        suite="agent",
        results=[
            BackendResult(
                "groq",
                "m",
                {
                    "tool_path_accuracy": 0.8,
                    "errors": 2,
                    "errors_all_attempts": 6,
                    "cases": 20,
                },
            )
        ],
    )
    row = [line for line in report.as_markdown().splitlines() if "groq" in line][0]
    assert "2 of 20" in row
    assert "6 all attempts" in row, "a resumed run must not read as a clean single pass"


# ---- result files must not silently overwrite each other --------------------


def test_result_files_do_not_collide_within_one_second(tmp_path):
    """A fully-checkpointed --resume run makes no LLM calls and finishes in
    milliseconds, so several backends regenerated back to back land in the same
    second. Each wrote the same filename and only the last survived."""
    from finrag.eval.tracking import track_run

    for backend in ("groq", "cerebras", "github"):
        params = {"llm_backend": backend, "resolved_model": f"{backend}-model"}
        with track_run("agent", params, results_dir=tmp_path) as record:
            record({"tool_path_accuracy": 0.5})

    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 3, f"runs overwrote each other: {[f.name for f in files]}"
    assert all(any(b in f.name for b in ("groq", "cerebras", "github")) for f in files)


def test_identical_runs_get_distinct_files(tmp_path):
    """Even the same backend twice in the same second must not clobber."""
    from finrag.eval.tracking import track_run

    params = {"llm_backend": "groq", "resolved_model": "llama-3.3-70b-versatile"}
    for _ in range(2):
        with track_run("agent", params, results_dir=tmp_path) as record:
            record({"tool_path_accuracy": 0.5})
    assert len(list(tmp_path.glob("*.json"))) == 2


# ---- the cache fix generalises beyond the client that exposed it ------------


def test_cache_namespacing_covers_every_backend(monkeypatch, tmp_path):
    """ChatOllama surfaced this, but the fix is at the file level precisely so
    it does not depend on auditing each client's cache key -- including
    ChatVertexAI, which is not installed here and so cannot be inspected."""
    from finrag.cache import cache_path

    monkeypatch.setenv("FINRAG_DATA_ROOT", str(tmp_path))
    base = get_settings()
    paths = {
        cache_path(replace(base, llm_backend=b, chat_model=""))
        for b in ("ollama", "vertex", "groq", "cerebras", "github", "anthropic")
    }
    assert len(paths) == 6, "one cache file per backend, no sharing"
