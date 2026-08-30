from __future__ import annotations

from finrag.config import get_settings


def test_defaults_need_no_environment(monkeypatch):
    """A fresh clone with no .env must still produce usable settings."""
    for var in (
        "FINRAG_DATA_ROOT",
        "FINRAG_EMBEDDINGS",
        "FINRAG_CHUNK_STRATEGY",
        "SEC_CONTACT_EMAIL",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = get_settings()
    assert settings.embedding_backend == "local", "local needs no API key, so it is the default"
    assert str(settings.data_root) == "data"


def test_index_path_separates_embedding_backends(monkeypatch):
    """Local and Google vectors have different dimensions and must not mix."""
    monkeypatch.setenv("FINRAG_EMBEDDINGS", "local")
    local = get_settings().index_dir
    monkeypatch.setenv("FINRAG_EMBEDDINGS", "google")
    assert get_settings().index_dir != local


def test_sec_contact_is_required_only_when_downloading(monkeypatch):
    monkeypatch.delenv("SEC_CONTACT_EMAIL", raising=False)
    settings = get_settings()
    assert settings.embedding_backend  # constructing settings must not raise
    try:
        settings.require_sec_contact()
    except RuntimeError as exc:
        assert "SEC_CONTACT_EMAIL" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when the contact address is missing")


def test_the_agent_is_bounded_per_question(monkeypatch):
    """Unbounded iterations let one question resend its scratchpad up to 15
    times -- ~334k tokens on Groq, more on an untrimmed backend."""
    from dataclasses import replace

    from finrag.config import Settings

    for var in ("FINRAG_AGENT_MAX_ITERATIONS", "FINRAG_ANSWER_TIMEOUT_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    settings = get_settings()
    assert settings.agent_max_iterations == 6, "the default must cap the runaway"
    assert settings.answer_timeout_seconds > 0

    # And the override is honoured.
    monkeypatch.setenv("FINRAG_AGENT_MAX_ITERATIONS", "3")
    assert replace(Settings()).agent_max_iterations == 3


def test_the_executor_is_wired_with_both_bounds():
    """A hung provider call yields no stream chunk, so the UI's between-chunk
    deadline cannot help -- max_execution_time enforces the bound inside the
    loop. Assert both reach the AgentExecutor rather than trusting the wiring."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src" / "finrag" / "agent.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    executors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AgentExecutor"
    ]
    assert executors, "agent.py no longer constructs an AgentExecutor"
    kwargs = {k.arg for k in executors[0].keywords}
    assert "max_iterations" in kwargs
    assert "max_execution_time" in kwargs, "the in-loop timeout must be set"
