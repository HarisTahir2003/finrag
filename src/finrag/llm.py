"""Pluggable chat model backends.

Anthropic and Google are both supported and selected by configuration, because
the pipeline should not be welded to whichever provider the author happened to
have credits with.

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
}


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

    raise ValueError(f"unknown llm backend {backend!r}; expected 'anthropic' or 'google'")


def required_api_key(settings: Settings | None = None) -> str:
    """Name of the environment variable the configured backend reads."""
    settings = settings or get_settings()
    return {"anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}[
        settings.llm_backend.lower()
    ]
