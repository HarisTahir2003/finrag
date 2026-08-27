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
    [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("google", "GOOGLE_API_KEY"),
        ("groq", "GROQ_API_KEY"),
    ],
)
def test_required_key_follows_the_backend(monkeypatch, backend, expected):
    monkeypatch.setenv("FINRAG_LLM_BACKEND", backend)
    assert required_api_key(get_settings()) == expected


def test_ollama_needs_no_key(monkeypatch):
    """A keyless backend must report None, not raise or invent a variable name."""
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "ollama")
    assert required_api_key(get_settings()) is None


def test_every_backend_has_a_default_model():
    assert set(DEFAULT_MODELS) == {"anthropic", "google", "groq", "ollama"}
    assert all(v for v in DEFAULT_MODELS.values())


def test_builds_an_ollama_client_without_credentials(monkeypatch):
    """The point of the local backend: it constructs with no key present."""
    pytest.importorskip("langchain_ollama")
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "ollama")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FINRAG_CHAT_MODEL", raising=False)
    llm = get_chat_model(get_settings())
    assert type(llm).__name__ == "ChatOllama"
    assert llm.model == DEFAULT_MODELS["ollama"]
    # num_ctx must be large enough for retrieved chunks plus the scratchpad,
    # or Ollama truncates the prompt silently.
    assert llm.num_ctx >= 8192


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


# ---- OpenAI-compatible providers ----------------------------------------
# Cerebras, OpenRouter, Together, Fireworks and DeepInfra all speak the same
# protocol, so they share one client and differ only by preset.


def test_every_preset_is_complete():
    from finrag.llm import PROVIDER_PRESETS

    for name, preset in PROVIDER_PRESETS.items():
        assert preset.base_url.startswith("https://"), name
        assert preset.default_model, name
        assert preset.key_env.endswith("_API_KEY"), name


def test_all_backends_covers_both_families():
    from finrag.llm import PROVIDER_PRESETS, all_backends

    backends = set(all_backends())
    assert {"anthropic", "google", "groq", "ollama"} <= backends
    assert set(PROVIDER_PRESETS) <= backends


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("cerebras", "CEREBRAS_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("together", "TOGETHER_API_KEY"),
    ],
)
def test_preset_backends_report_their_key(monkeypatch, backend, expected):
    monkeypatch.setenv("FINRAG_LLM_BACKEND", backend)
    assert required_api_key(get_settings()) == expected


def test_missing_key_fails_with_a_useful_message(monkeypatch):
    """Silent fallback to a wrong endpoint would be far harder to debug."""
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "cerebras")
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    monkeypatch.delenv("FINRAG_LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CEREBRAS_API_KEY"):
        get_chat_model(get_settings())


def test_builds_a_client_pointed_at_the_preset_url(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "cerebras")
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
    monkeypatch.delenv("FINRAG_CHAT_MODEL", raising=False)
    monkeypatch.delenv("FINRAG_OPENAI_BASE_URL", raising=False)
    llm = get_chat_model(get_settings())
    assert type(llm).__name__ == "ChatOpenAI"
    assert "cerebras.ai" in str(llm.openai_api_base)
    assert llm.model_name == "gpt-oss-120b"


def test_base_url_override_wins(monkeypatch):
    """So the same backend can point at a self-hosted vLLM server."""
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("FINRAG_OPENAI_BASE_URL", "http://localhost:8000/v1")
    assert "localhost:8000" in str(get_chat_model(get_settings()).openai_api_base)


def test_generic_key_variable_is_a_fallback(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "together")
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.setenv("FINRAG_LLM_API_KEY", "shared-key")
    assert get_chat_model(get_settings()) is not None
