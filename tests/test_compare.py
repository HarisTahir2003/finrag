"""The cross-backend comparison harness.

A benchmark that produces plausible-but-wrong rankings is worse than no
benchmark, so these tests pin the properties that make the ranking mean what it
appears to mean: one variable changes, checkpoints cannot cross-contaminate,
ties break on something meaningful, and a dead backend does not abort the sweep.
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest

from finrag.config import get_settings
from finrag.eval.compare import (
    AGENT_RANK_KEYS,
    BackendResult,
    ComparisonReport,
    _checkpoint_for,
    compare_backends,
)
from finrag.eval.schema import AgentCase


def result(backend: str, **metrics) -> BackendResult:
    return BackendResult(backend=backend, model=f"{backend}-model", metrics=metrics)


# ---- ranking ----------------------------------------------------------------


def test_ranks_on_the_headline_metric():
    report = ComparisonReport(
        suite="agent",
        results=[
            result("a", tool_path_accuracy=0.5),
            result("b", tool_path_accuracy=0.9),
            result("c", tool_path_accuracy=0.7),
        ],
    )
    assert [r.backend for r in report.ranked()] == ["b", "c", "a"]


def test_tie_breaks_on_whether_an_answer_was_actually_given():
    """The failure this guards: 100% tool calls whose every output was
    'which company did you mean?' would otherwise tie with a real answer."""
    report = ComparisonReport(
        suite="agent",
        results=[
            result(
                "asks_back",
                tool_path_accuracy=1.0,
                answered_with_figures=0.0,
                clarification_requests=1.0,
            ),
            result(
                "answers",
                tool_path_accuracy=1.0,
                answered_with_figures=1.0,
                clarification_requests=0.0,
            ),
        ],
    )
    assert report.winner().backend == "answers"


def test_errors_break_a_full_tie():
    report = ComparisonReport(
        suite="agent",
        results=[
            result(
                "flaky",
                tool_path_accuracy=1.0,
                answered_with_figures=1.0,
                clarification_requests=0.0,
                errors=3,
            ),
            result(
                "solid",
                tool_path_accuracy=1.0,
                answered_with_figures=1.0,
                clarification_requests=0.0,
                errors=0,
            ),
        ],
    )
    assert report.winner().backend == "solid"


def test_unavailable_backends_sort_last_and_never_win():
    ok = result("ok", tool_path_accuracy=0.1)
    dead = BackendResult(backend="dead", error="ImportError: no extra")
    report = ComparisonReport(suite="agent", results=[dead, ok])
    assert [r.backend for r in report.ranked()] == ["ok", "dead"]
    assert report.winner().backend == "ok"


def test_winner_is_none_when_nothing_ran():
    report = ComparisonReport(
        suite="agent",
        results=[
            BackendResult(backend="x", error="boom"),
        ],
    )
    assert report.winner() is None


def test_rank_keys_are_directional():
    """clarification_requests and errors must count against a backend."""
    directions = dict(AGENT_RANK_KEYS)
    assert directions["tool_path_accuracy"] == 1
    assert directions["clarification_requests"] == -1
    assert directions["errors"] == -1


def test_ragas_suite_ranks_on_faithfulness():
    report = ComparisonReport(
        suite="ragas",
        rank_key="ragas_faithfulness",
        results=[
            result("a", ragas_faithfulness=0.4),
            result("b", ragas_faithfulness=0.8),
        ],
    )
    assert report.winner().backend == "b"


# ---- checkpoint isolation ---------------------------------------------------


def test_checkpoints_are_unique_per_backend_and_model():
    """Sharing a checkpoint across backends would report one model's answers as
    another's -- the single most corrupting bug this harness could have."""
    paths = {
        _checkpoint_for("agent", b, m)
        for b, m in [
            ("groq", "llama-3.3-70b-versatile"),
            ("cerebras", "gpt-oss-120b"),
            ("github", "openai/gpt-4o-mini"),
            ("vertex", "gemini-2.5-flash"),
        ]
    }
    assert len(paths) == 4


def test_checkpoint_name_survives_slashes_in_model_ids():
    path = _checkpoint_for("agent", "github", "openai/gpt-4o-mini")
    assert "/" not in path.name
    assert path.parent.name == "results"


def test_same_backend_different_model_gets_a_different_checkpoint():
    a = _checkpoint_for("agent", "groq", "llama-3.3-70b-versatile")
    b = _checkpoint_for("agent", "groq", "qwen-3-32b")
    assert a != b


# ---- the sweep --------------------------------------------------------------


class StubAgent:
    def __init__(self, tools, answer):
        self.tools, self.answer = tools, answer
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        steps = [(type("Action", (), {"tool": t})(), "obs") for t in self.tools]
        return {"output": self.answer, "intermediate_steps": steps}


@pytest.fixture
def four_cases():
    return [
        AgentCase(
            id=f"c{i}",
            question=f"q{i}",
            ticker="AAPL",
            fiscal_year=2023,
            expected_tools=["search_10k_reports", "calculator"],
        )
        for i in range(3)
    ]


@pytest.fixture
def tmp_settings(monkeypatch):
    monkeypatch.setenv("FINRAG_DATA_ROOT", tempfile.mkdtemp())
    return get_settings()


def test_sweep_runs_every_backend_and_ranks_them(four_cases, tmp_settings, monkeypatch):
    monkeypatch.chdir(tempfile.mkdtemp())  # keep results/ out of the repo
    behaviour = {
        "groq": StubAgent(["search_10k_reports", "calculator"], "the ratio is 0.988"),
        "cerebras": StubAgent(["search_10k_reports"], "the ratio is 0.988"),
    }

    def fake_build_agent(store=None, settings=None, verbose=False):
        if settings.llm_backend not in behaviour:
            raise ImportError(f"no extra for {settings.llm_backend}")
        return behaviour[settings.llm_backend]

    with patch("finrag.agent.build_agent", fake_build_agent):
        report = compare_backends(
            ["groq", "cerebras", "nonexistent"],
            suite="agent",
            cases=four_cases,
            store=object(),
            settings=tmp_settings,
        )

    assert report.winner().backend == "groq"
    assert behaviour["groq"].calls == 3
    assert behaviour["cerebras"].calls == 3
    dead = [r for r in report.results if r.backend == "nonexistent"][0]
    assert not dead.available
    assert "ImportError" in dead.error


def test_sweep_varies_only_the_backend(four_cases, tmp_settings, monkeypatch):
    """Everything except llm_backend must be identical, or the numbers are
    measuring configuration drift rather than the model."""
    monkeypatch.chdir(tempfile.mkdtemp())
    seen = []

    def fake_build_agent(store=None, settings=None, verbose=False):
        seen.append(settings)
        return StubAgent(["search_10k_reports", "calculator"], "0.988")

    with patch("finrag.agent.build_agent", fake_build_agent):
        compare_backends(
            ["groq", "cerebras"],
            suite="agent",
            cases=four_cases,
            store=object(),
            settings=tmp_settings,
        )

    assert len(seen) == 2
    a, b = seen
    assert a.llm_backend != b.llm_backend
    for attr in (
        "retrieval_k",
        "chunk_strategy",
        "chunk_size",
        "chunk_overlap",
        "embedding_backend",
        "collection_name",
        "max_output_tokens",
    ):
        assert getattr(a, attr) == getattr(b, attr), attr


def test_sweep_clears_chat_model_per_backend(four_cases, tmp_settings, monkeypatch):
    """A model id from one provider is meaningless to another; each backend must
    run its own default rather than inherit FINRAG_CHAT_MODEL."""
    monkeypatch.chdir(tempfile.mkdtemp())
    monkeypatch.setenv("FINRAG_CHAT_MODEL", "llama-3.3-70b-versatile")
    settings = get_settings()
    seen = []

    def fake_build_agent(store=None, settings=None, verbose=False):
        seen.append(settings.chat_model)
        return StubAgent(["search_10k_reports"], "x")

    with patch("finrag.agent.build_agent", fake_build_agent):
        compare_backends(
            ["groq", "cerebras"], suite="agent", cases=four_cases, store=object(), settings=settings
        )
    assert seen == ["", ""]


def test_fixed_conditions_are_recorded(four_cases, tmp_settings, monkeypatch):
    """A published table has to state what was held constant."""
    monkeypatch.chdir(tempfile.mkdtemp())
    with patch("finrag.agent.build_agent", lambda **kw: StubAgent(["search_10k_reports"], "x")):
        report = compare_backends(
            ["groq"], suite="agent", cases=four_cases, store=object(), settings=tmp_settings
        )
    assert report.fixed["retrieval_k"] == str(tmp_settings.retrieval_k)
    assert report.fixed["embeddings"] == tmp_settings.embedding_backend
    assert "chunk_strategy" in report.fixed


# ---- output -----------------------------------------------------------------


def test_markdown_table_is_wellformed():
    report = ComparisonReport(
        suite="agent",
        fixed={"retrieval_k": "6"},
        results=[
            result(
                "groq",
                tool_path_accuracy=0.85,
                calculator_compliance=1.0,
                clarification_requests=0.0,
                answered_with_figures=1.0,
                errors=0,
            ),
            BackendResult(backend="ollama", model="qwen3:4b", error="connection refused"),
        ],
    )
    md = report.as_markdown()
    lines = md.splitlines()
    assert lines[0].startswith("| backend | model |")
    assert set(lines[1]) <= {"|", "-"}
    # header + rule + 2 data rows all have the same column count
    counts = {line.count("|") for line in lines[:4]}
    assert len(counts) == 1, f"ragged table: {counts}"
    assert "85%" in md
    assert "unavailable" in md
    assert "retrieval_k=6" in md


def test_as_dict_is_json_serialisable():
    import json

    report = ComparisonReport(suite="agent", results=[result("groq", tool_path_accuracy=1.0)])
    assert json.loads(json.dumps(report.as_dict()))["results"][0]["backend"] == "groq"
