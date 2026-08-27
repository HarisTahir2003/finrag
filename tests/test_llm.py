"""The chat backend is selectable, so the pipeline is not welded to one provider."""

from __future__ import annotations

import pytest

from finrag.config import get_settings
from finrag.llm import DEFAULT_MODELS, get_chat_model, required_api_key


def test_defaults_to_anthropic(monkeypatch):
    monkeypatch.delenv("FINRAG_LLM_BACKEND", raising=False)
    assert get_settings().llm_backend == "anthropic"


def test_blank_model_resolves_to_the_backend_default(monkeypatch):
    """An empty FINRAG_CHAT_MODEL must not be passed through as a model name."""
    monkeypatch.delenv("FINRAG_CHAT_MODEL", raising=False)
    assert get_settings().chat_model == ""
    assert DEFAULT_MODELS["anthropic"].startswith("claude-")
    assert DEFAULT_MODELS["google"].startswith("gemini-")


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("anthropic", "ANTHROPIC_API_KEY"), ("google", "GOOGLE_API_KEY")],
)
def test_required_key_follows_the_backend(monkeypatch, backend, expected):
    monkeypatch.setenv("FINRAG_LLM_BACKEND", backend)
    assert required_api_key(get_settings()) == expected


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "hal9000")
    with pytest.raises(ValueError, match="unknown llm backend"):
        get_chat_model(get_settings())


def test_builds_an_anthropic_client(monkeypatch):
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    monkeypatch.delenv("FINRAG_CHAT_MODEL", raising=False)
    llm = get_chat_model(get_settings())
    assert type(llm).__name__ == "ChatAnthropic"
    assert llm.model == DEFAULT_MODELS["anthropic"]


def test_explicit_model_overrides_the_default(monkeypatch):
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    monkeypatch.setenv("FINRAG_CHAT_MODEL", "claude-sonnet-5")
    assert get_chat_model(get_settings()).model == "claude-sonnet-5"


def test_overrides_reach_the_constructor(monkeypatch):
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    assert get_chat_model(get_settings(), max_tokens=123).max_tokens == 123
