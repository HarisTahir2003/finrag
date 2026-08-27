"""Pluggable chat model backends.

Four backends, selected by configuration, because the pipeline should not be
welded to whichever provider the author happened to have credits with:

``anthropic`` / ``google``
    Commercial APIs. Most reliable tool calling, paid per token.

``groq``
    Open-weight models (Llama, Qwen, GPT-OSS) on a free developer tier. No
    credit card and no per-token charge, but tight rate limits -- notably a
    tokens-per-minute cap that a large retrieval context will breach. Lower
    ``FINRAG_RETRIEVAL_K`` when using it.

``ollama``
    Open-weight models running locally. No key, no rate limit, no network, and
    no cost. Bounded by the machine: a 4B model quantised to 4 bits needs
    roughly 3GB of RAM, an 8B closer to 6GB, and long retrieval contexts add
    more on top.

Note the asymmetry: Anthropic publishes no embedding model, so the embedding
backend stays independent of this one. That is not a limitation in practice --
``finrag.embeddings`` defaults to local sentence-transformers, which costs
nothing and needs no key at all.
"""

from __future__ import annotations

from typing import Any

from .config import Settings, get_settings

# Sensible default per provider, chosen for cost rather than capability. Both
# are more than adequate for retrieval-grounded question answering, where the
# model is reading supplied context rather than reasoning from memory.
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "google": "gemini-2.5-flash",
    # Qwen3 is the most reliable locally-runnable series for tool calling --
    # it drops or malforms tool calls far less than comparable open models,
    # which matters more here than raw quality because a dropped tool call
    # means the agent invents a number instead of retrieving one.
    "ollama": "qwen3:4b",
    "groq": "llama-3.3-70b-versatile",
}

# Backends that need no credentials at all.
KEYLESS_BACKENDS = frozenset({"ollama"})


def get_chat_model(settings: Settings | None = None, **overrides: Any) -> Any:
    """Build the configured chat model.

    ``overrides`` are passed through to the underlying constructor, so callers
    can set temperature or token limits without going through settings.
    """
    settings = settings or get_settings()
    backend = settings.llm_backend.lower()
    model = settings.chat_model or DEFAULT_MODELS.get(backend, "")
    params = {"temperature": 0, **overrides}

    if backend == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "The Anthropic backend needs: pip install 'finrag[anthropic]'"
            ) from exc
        params.setdefault("max_tokens", settings.max_output_tokens)
        return ChatAnthropic(model=model, **params)

    if backend == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError("The Google backend needs: pip install 'finrag[google]'") from exc
        params.setdefault("max_output_tokens", settings.max_output_tokens)
        return ChatGoogleGenerativeAI(model=model, **params)

    if backend == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise ImportError("The Groq backend needs: pip install 'finrag[groq]'") from exc
        params.setdefault("max_tokens", settings.max_output_tokens)
        return ChatGroq(model=model, **params)

    if backend == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError("The Ollama backend needs: pip install 'finrag[ollama]'") from exc
        # num_ctx must cover the retrieved chunks plus the agent scratchpad.
        # Ollama silently truncates the prompt otherwise, which looks like the
        # model ignoring its context rather than never having received it.
        params.setdefault("num_ctx", settings.ollama_context_length)
        params.setdefault("base_url", settings.ollama_base_url)
        params.pop("max_tokens", None)
        return ChatOllama(model=model, **params)

    raise ValueError(f"unknown llm backend {backend!r}; expected one of {sorted(DEFAULT_MODELS)}")


def required_api_key(settings: Settings | None = None) -> str | None:
    """Environment variable the configured backend reads, or None if it needs no key."""
    settings = settings or get_settings()
    backend = settings.llm_backend.lower()
    if backend in KEYLESS_BACKENDS:
        return None
    keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
    }
    if backend not in keys:
        raise ValueError(f"unknown llm backend {backend!r}")
    return keys[backend]
