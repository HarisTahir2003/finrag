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
    """Asserted as a subset, not an exact set: adding a backend is a routine
    change and should not break an unrelated test."""
    assert {"anthropic", "google", "groq", "ollama", "vertex"} <= set(DEFAULT_MODELS)
    assert all(v for v in DEFAULT_MODELS.values())


def test_every_backend_has_an_rpm_policy():
    """A backend missing from DEFAULT_RPM silently runs unthrottled, which on a
    free tier means 429s rather than pacing."""
    from finrag.llm import DEFAULT_RPM, PROVIDER_PRESETS

    for backend in set(DEFAULT_MODELS) | set(PROVIDER_PRESETS):
        assert backend in DEFAULT_RPM, backend


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
        # github uses GITHUB_TOKEN -- the one preset whose credential is a
        # platform token rather than a provider API key.
        assert preset.key_env.endswith(("_API_KEY", "_TOKEN")), name


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


# ------------------------------------------------- provider error triage


def _groq_error(cls_name: str, status: int, message: str):
    """A real provider exception, built the way the client library builds one."""
    import groq
    import httpx

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return getattr(groq, cls_name)(
        message, response=httpx.Response(status, request=request), body=None
    )


def test_a_daily_quota_is_not_a_transient_rate_limit():
    """The distinction the classifier exists for.

    Both arrive as HTTP 429. Telling someone to wait a few seconds when the
    daily quota is spent is advice that never comes true, and on a public demo
    the difference decides whether the visitor is told to supply their own key.
    """
    from finrag.llm import classify_provider_error

    daily = _groq_error(
        "RateLimitError",
        429,
        "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` "
        "service tier `on_demand` on requests per day (RPD): Limit 1000, Used 1000",
    )
    minute = _groq_error(
        "RateLimitError",
        429,
        "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` "
        "on tokens per minute (TPM): Limit 8000, Used 7995",
    )

    assert classify_provider_error(daily) == "quota"
    assert classify_provider_error(minute) == "rate_limit"


def test_a_rejected_key_and_a_missing_key_are_told_apart():
    """Different advice: fix what you typed, versus type something at all."""
    import groq

    from finrag.llm import classify_provider_error

    rejected = _groq_error("AuthenticationError", 401, "Invalid API Key")
    absent = groq.GroqError(
        "The api_key client option must be set either by passing api_key to the "
        "client or by setting the GROQ_API_KEY environment variable"
    )

    assert classify_provider_error(rejected) == "auth"
    assert classify_provider_error(absent) == "missing_key"


def test_an_unrelated_failure_is_not_misreported_as_a_quota_problem():
    """A bug in retrieval must not tell the reader to buy more quota."""
    from finrag.llm import classify_provider_error

    assert classify_provider_error(ValueError("chroma exploded")) == "other"
    assert classify_provider_error(KeyError("id")) == "other"


def test_the_quota_message_tells_the_visitor_what_to_do():
    from finrag.presentation import failure_message

    daily = _groq_error("RateLimitError", 429, "on requests per day (RPD): Limit 1000, Used 1000")
    text = failure_message(daily, "groq")

    assert "sidebar" in text.lower(), "must point at the control that fixes it"
    assert "session" in text.lower(), "must say the key is not stored"

    minute = _groq_error("RateLimitError", 429, "on tokens per minute (TPM): Limit 8000")
    assert "sidebar" not in failure_message(minute, "groq").lower(), (
        "a transient limit must not send the visitor hunting for an API key"
    )


def test_an_unclassified_failure_still_shows_the_error():
    """Swallowing the detail would make a real bug undiagnosable."""
    from finrag.presentation import failure_message

    assert "chroma exploded" in failure_message(ValueError("chroma exploded"), "groq")
