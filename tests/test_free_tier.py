"""The free-tier survival machinery: pacing, fallbacks, budgets, checkpoints, cache.

Everything here runs without a network. The live behaviour (does Cerebras
actually accept the paced run) is exercised manually and by the CI llm-smoke
job; these tests pin the mechanics.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

from finrag.config import get_settings
from finrag.eval.checkpoint import Checkpoint
from finrag.retrieval import trim_to_token_budget

# ---- rate limiting ----------------------------------------------------------


def test_free_backends_get_a_default_limiter(monkeypatch):
    pytest.importorskip("langchain_groq")
    from finrag.llm import get_chat_model

    monkeypatch.setenv("FINRAG_LLM_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("FINRAG_RPM", raising=False)
    llm = get_chat_model(get_settings())
    assert llm.rate_limiter is not None
    # 25 requests/minute, expressed as requests/second.
    assert llm.rate_limiter.requests_per_second == pytest.approx(25 / 60)


def test_finrag_rpm_overrides_the_default(monkeypatch):
    pytest.importorskip("langchain_groq")
    from finrag.llm import get_chat_model

    monkeypatch.setenv("FINRAG_LLM_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("FINRAG_RPM", "10")
    llm = get_chat_model(get_settings())
    assert llm.rate_limiter.requests_per_second == pytest.approx(10 / 60)


def test_finrag_rpm_zero_disables_pacing(monkeypatch):
    pytest.importorskip("langchain_groq")
    from finrag.llm import get_chat_model

    monkeypatch.setenv("FINRAG_LLM_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("FINRAG_RPM", "0")
    assert get_chat_model(get_settings()).rate_limiter is None


def test_paid_backends_are_unthrottled_by_default(monkeypatch):
    pytest.importorskip("langchain_anthropic")
    from finrag.llm import get_chat_model

    monkeypatch.setenv("FINRAG_LLM_BACKEND", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("FINRAG_RPM", raising=False)
    assert get_chat_model(get_settings()).rate_limiter is None


def test_github_models_preset(monkeypatch):
    """The CI backend: GITHUB_TOKEN against models.github.ai."""
    pytest.importorskip("langchain_openai")
    from finrag.llm import get_chat_model, required_api_key

    monkeypatch.setenv("FINRAG_LLM_BACKEND", "github")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.delenv("FINRAG_CHAT_MODEL", raising=False)
    monkeypatch.delenv("FINRAG_OPENAI_BASE_URL", raising=False)
    assert required_api_key(get_settings()) == "GITHUB_TOKEN"
    llm = get_chat_model(get_settings())
    assert "models.github.ai" in str(llm.openai_api_base)
    assert llm.rate_limiter is not None, "GitHub Models' ~10 RPM cap needs pacing"


# ---- fallback chains --------------------------------------------------------


def test_fallback_chain_is_built_in_order(monkeypatch):
    pytest.importorskip("langchain_groq")
    pytest.importorskip("langchain_openai")
    from finrag.llm import build_with_fallbacks

    monkeypatch.setenv("FINRAG_LLM_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("FINRAG_LLM_FALLBACKS", "openrouter")
    chain = build_with_fallbacks(get_settings())
    assert type(chain).__name__ == "RunnableWithFallbacks"
    assert len(chain.fallbacks) == 1
    assert "openrouter" in str(chain.fallbacks[0].openai_api_base)


def test_fallbacks_use_their_own_default_model(monkeypatch):
    """A Groq model id passed to OpenRouter would 404 -- ids do not transfer."""
    pytest.importorskip("langchain_groq")
    pytest.importorskip("langchain_openai")
    from finrag.llm import build_with_fallbacks

    monkeypatch.setenv("FINRAG_LLM_BACKEND", "groq")
    monkeypatch.setenv("FINRAG_CHAT_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("FINRAG_LLM_FALLBACKS", "openrouter")
    chain = build_with_fallbacks(get_settings())
    assert chain.fallbacks[0].model_name == "meta-llama/llama-3.3-70b-instruct"


def test_no_fallbacks_returns_the_bare_model(monkeypatch):
    pytest.importorskip("langchain_groq")
    from finrag.llm import build_with_fallbacks

    monkeypatch.setenv("FINRAG_LLM_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("FINRAG_LLM_FALLBACKS", raising=False)
    assert type(build_with_fallbacks(get_settings())).__name__ == "ChatGroq"


def test_primary_is_dropped_from_its_own_fallback_list(monkeypatch):
    pytest.importorskip("langchain_groq")
    from finrag.llm import build_with_fallbacks

    monkeypatch.setenv("FINRAG_LLM_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("FINRAG_LLM_FALLBACKS", "groq")
    assert type(build_with_fallbacks(get_settings())).__name__ == "ChatGroq"


# ---- context token budget ---------------------------------------------------


def doc(chars: int, tag: str) -> Document:
    return Document(page_content=tag * chars, metadata={"id": tag})


def test_budget_keeps_top_ranked_whole_chunks():
    docs = [doc(4000, "a"), doc(4000, "b"), doc(4000, "c")]
    kept = trim_to_token_budget(docs, max_tokens=2100)  # 8400 chars of budget
    assert [d.metadata["id"] for d in kept] == ["a", "b"]


def test_budget_zero_means_unlimited():
    docs = [doc(100, "a")] * 50
    assert trim_to_token_budget(docs, 0) == docs


def test_one_oversized_chunk_is_still_returned():
    """Something to answer from beats an empty context."""
    kept = trim_to_token_budget([doc(50_000, "big")], max_tokens=1000)
    assert len(kept) == 1


def test_auto_budget_follows_the_backend(monkeypatch):
    monkeypatch.setenv("FINRAG_MAX_CONTEXT_TOKENS", "auto")
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "cerebras")
    assert get_settings().max_context_tokens == 6000
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "github")
    assert get_settings().max_context_tokens == 6000
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "groq")
    assert get_settings().max_context_tokens == 0


def test_explicit_budget_beats_auto(monkeypatch):
    monkeypatch.setenv("FINRAG_MAX_CONTEXT_TOKENS", "3000")
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "groq")
    assert get_settings().max_context_tokens == 3000


# ---- checkpointing ----------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "cases.jsonl"
    first = Checkpoint(path)
    first.record("case-1", {"answer": "0.988", "tools_called": ["search_10k_reports"]})

    resumed = Checkpoint(path)
    assert len(resumed) == 1
    assert resumed.completed("case-1")["answer"] == "0.988"
    assert resumed.completed("case-2") is None


def test_checkpoint_survives_a_corrupt_line(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "ok", "answer": "1"}\nnot json at all\n', encoding="utf-8")
    checkpoint = Checkpoint(path)
    assert checkpoint.completed("ok") is not None
    assert len(checkpoint) == 1


def test_checkpoint_none_path_is_memory_only():
    checkpoint = Checkpoint(None)
    checkpoint.record("x", {"answer": "1"})
    assert checkpoint.completed("x") is not None


def test_agent_eval_resumes_without_repeating_calls(tmp_path):
    """The point of the whole mechanism: quota is spent only on unfinished cases."""
    from finrag.eval.agent_eval import evaluate_agent
    from finrag.eval.schema import AgentCase

    cases = [
        AgentCase(id=f"c{i}", question=f"q{i}", ticker="AAPL", fiscal_year=2023) for i in range(4)
    ]
    path = tmp_path / "agent-cases.jsonl"

    class CountingAgent:
        def __init__(self, fail_after: int | None = None):
            self.calls = 0
            self.fail_after = fail_after

        def invoke(self, payload):
            self.calls += 1
            if self.fail_after is not None and self.calls > self.fail_after:
                raise RuntimeError("daily quota exhausted")
            return {"output": "answer 42", "intermediate_steps": []}

    # First run: 2 cases succeed, then the "quota" dies.
    first = CountingAgent(fail_after=2)
    report = evaluate_agent(cases=cases, agent=first, checkpoint_path=path)
    assert first.calls == 4
    assert sum(1 for r in report.results if r.error) == 2
    assert len(path.read_text().splitlines()) == 2, "failed cases must not be checkpointed"

    # Resumed run: only the two unfinished cases reach the agent.
    second = CountingAgent()
    report = evaluate_agent(cases=cases, agent=second, checkpoint_path=path)
    assert second.calls == 2
    assert len(report.results) == 4
    assert not any(r.error for r in report.results)


# ---- response cache ---------------------------------------------------------


def test_cache_respects_the_kill_switch(monkeypatch):
    import finrag.cache as cache_module

    monkeypatch.setattr(cache_module, "_installed", None)
    monkeypatch.setenv("FINRAG_LLM_CACHE", "0")
    assert cache_module.enable_llm_cache(get_settings()) is False


def test_cache_enables_and_is_idempotent(monkeypatch, tmp_path):
    pytest.importorskip("langchain_community")
    import finrag.cache as cache_module

    monkeypatch.setattr(cache_module, "_installed", None)
    monkeypatch.setenv("FINRAG_LLM_CACHE", "1")
    monkeypatch.setenv("FINRAG_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "groq")
    settings = get_settings()

    assert cache_module.enable_llm_cache(settings) is True
    # The file is namespaced by backend and model, not a single shared db.
    assert cache_module.cache_path(settings).exists()
    assert cache_module.enable_llm_cache(settings) is True

    from langchain_core.globals import get_llm_cache, set_llm_cache

    assert get_llm_cache() is not None
    set_llm_cache(None)  # do not leak global state into other tests


def test_switching_backend_repoints_the_cache(monkeypatch, tmp_path):
    """A process that evaluates two backends in turn -- which is exactly what
    `finrag compare` does -- must not keep the first backend's cache installed."""
    pytest.importorskip("langchain_community")
    import finrag.cache as cache_module

    monkeypatch.setattr(cache_module, "_installed", None)
    monkeypatch.setenv("FINRAG_DATA_ROOT", str(tmp_path))

    monkeypatch.setenv("FINRAG_LLM_BACKEND", "groq")
    cache_module.enable_llm_cache(get_settings())
    first = cache_module._installed

    monkeypatch.setenv("FINRAG_LLM_BACKEND", "cerebras")
    cache_module.enable_llm_cache(get_settings())
    assert cache_module._installed != first

    from langchain_core.globals import set_llm_cache

    set_llm_cache(None)


# ---- smoke dataset ----------------------------------------------------------


def test_smoke_dataset_is_fixture_answerable():
    from finrag.eval.schema import DATASETS_DIR, load_agent_cases

    cases = load_agent_cases(DATASETS_DIR / "smoke.yaml")
    assert len(cases) == 3, "sized for GitHub Models' daily request budget"
    assert {c.ticker for c in cases} <= {"AAPL", "AMZN", "NVDA"}, "must match committed fixtures"
    assert {c.category for c in cases} == {"calc", "narrative", "mixed"}
    for case in cases:
        if case.category != "narrative":
            assert "calculator" in case.expected_tools


def test_checkpoint_records_are_json_serialisable(tmp_path):
    """Guards against a Document or dataclass sneaking into the payload."""
    path = tmp_path / "cases.jsonl"
    Checkpoint(path).record("x", {"contexts": ["a", "b"], "answer": "c"})
    line = json.loads(path.read_text().splitlines()[0])
    assert line == {"id": "x", "contexts": ["a", "b"], "answer": "c"}
