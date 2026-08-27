"""Pluggable chat model backends.

Selected by FINRAG_LLM_BACKEND, because the pipeline should not be welded to
whichever provider the author happened to have credits with:

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

``vertex``
    Gemini through Google Cloud Vertex AI rather than the AI Studio API. Two
    reasons it is separate: it authenticates with Application Default
    Credentials instead of an API key, and it bills the Cloud billing account
    directly -- which means Google Cloud promotional credits are consumed
    before anything else. The AI Studio path cannot reach those credits at all
    on a prepay account, so for anyone holding promo credits this is the free
    door to the same models.

``cerebras`` / ``openrouter`` / ``together`` / ``fireworks`` / ``deepinfra`` /
``openai`` / ``github``
    OpenAI-compatible endpoints -- one client, differing only by base URL,
    default model and key variable (see PROVIDER_PRESETS). cerebras and github
    have free tiers; github additionally works inside GitHub Actions with the
    built-in GITHUB_TOKEN.

Free-tier survival is built in rather than left to the caller: a client-side
token-bucket rate limiter paces requests under each tier's published RPM
(DEFAULT_RPM / FINRAG_RPM), and build_with_fallbacks() chains providers so one
exhausted daily quota fails over to the next instead of failing the run.

Note the asymmetry: Anthropic publishes no embedding model, so the embedding
backend stays independent of this one. That is not a limitation in practice --
``finrag.embeddings`` defaults to local sentence-transformers, which costs
nothing and needs no key at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
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
    "vertex": "gemini-2.5-flash",
}

# Applied uniformly to every provider. Free tiers 429 under load even when
# paced, and the comparison table ranks on `errors`, so an uneven retry policy
# would score client libraries rather than models.
MAX_RETRIES = 6

# Backends that read no API key from the environment. ollama needs nothing at
# all; vertex authenticates with Application Default Credentials, set up once
# with `gcloud auth application-default login`.
KEYLESS_BACKENDS = frozenset({"ollama", "vertex"})

# Default client-side requests-per-minute per backend, sized a little under each
# free tier's published cap so the limit is never *hit*, only approached. Free
# tiers return 429s that burn retries and, on some providers, count against the
# daily quota anyway -- pacing beats recovering. None means unthrottled (paid
# APIs and local models). Override with FINRAG_RPM; 0 disables.
DEFAULT_RPM: dict[str, float | None] = {
    "anthropic": None,
    "google": None,
    "ollama": None,  # local: the bottleneck is the machine, not a quota
    "vertex": None,  # Cloud quotas are per-project and generous; manage in GCP
    "groq": 25,  # free tier ~30 RPM
    "cerebras": 25,  # free tier ~30 RPM
    "openrouter": 18,  # free models ~20 RPM
    "github": 8,  # free tier ~10 RPM
    "together": None,
    "fireworks": None,
    "deepinfra": None,
    "openai": None,
}


@dataclass(frozen=True)
class Preset:
    """An OpenAI-compatible endpoint serving open-weight models."""

    base_url: str
    default_model: str
    key_env: str
    note: str = ""


# Every one of these speaks the OpenAI chat-completions protocol, so they run
# through the same client and differ only in these three fields.
PROVIDER_PRESETS: dict[str, Preset] = {
    "cerebras": Preset(
        "https://api.cerebras.ai/v1",
        "gpt-oss-120b",
        "CEREBRAS_API_KEY",
        "Free tier is generous on daily tokens, which suits batch evaluation.",
    ),
    "openrouter": Preset(
        "https://openrouter.ai/api/v1",
        "meta-llama/llama-3.3-70b-instruct",
        "OPENROUTER_API_KEY",
        "One key, many models. Append ':free' to a model id for the free variants, "
        "but check tool calling works on the specific model first -- it is unreliable "
        "on several of them, and a dropped tool call means an invented number.",
    ),
    "together": Preset(
        "https://api.together.xyz/v1",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "TOGETHER_API_KEY",
        "Broadest open-weight catalogue.",
    ),
    "fireworks": Preset(
        "https://api.fireworks.ai/inference/v1",
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "FIREWORKS_API_KEY",
    ),
    "deepinfra": Preset(
        "https://api.deepinfra.com/v1/openai",
        "meta-llama/Llama-3.3-70B-Instruct",
        "DEEPINFRA_API_KEY",
    ),
    "openai": Preset("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "github": Preset(
        "https://models.github.ai/inference",
        "openai/gpt-4o-mini",
        "GITHUB_TOKEN",
        "Free with any GitHub account; ~150 requests/day on mini-class models, 8K in / "
        "4K out per request. Inside GitHub Actions the built-in GITHUB_TOKEN works once "
        "the workflow requests `permissions: models: read` -- LLM calls in CI with zero "
        "secrets. Sized for smoke tests, not full evaluation runs.",
    ),
}


def _rate_limiter(backend: str, settings: Settings):
    """Client-side throttle for the backend, or None when unthrottled.

    ``InMemoryRateLimiter`` is a token bucket that blocks the caller until a
    request slot is available. It is per-process and time-based only, which is
    exactly the shape of the problem: one CLI run pacing itself under an RPM
    cap. max_bucket_size=1 forbids bursts, since free tiers meter per minute
    but police in much smaller windows.
    """
    rpm = settings.requests_per_minute
    if rpm is None:
        rpm = DEFAULT_RPM.get(backend)
    if not rpm or rpm <= 0:
        return None

    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(
        requests_per_second=rpm / 60.0,
        check_every_n_seconds=0.25,
        max_bucket_size=1,
    )


def get_chat_model(settings: Settings | None = None, **overrides: Any) -> Any:
    """Build the configured chat model.

    ``overrides`` are passed through to the underlying constructor, so callers
    can set temperature or token limits without going through settings.
    """
    settings = settings or get_settings()
    backend = settings.llm_backend.lower()

    if backend in PROVIDER_PRESETS:
        return _openai_compatible(backend, settings, **overrides)

    model = settings.chat_model or DEFAULT_MODELS.get(backend, "")
    # max_retries is set here rather than per-branch so every provider gets the
    # same policy. Left to library defaults it varies (2 on some clients, 6 on
    # others), and since `errors` is a ranked column in the comparison table a
    # backend would be penalised for its client library's retry policy rather
    # than its own reliability.
    params = {"temperature": 0, "max_retries": MAX_RETRIES, **overrides}
    limiter = _rate_limiter(backend, settings)
    if limiter is not None:
        params.setdefault("rate_limiter", limiter)

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

    if backend == "vertex":
        try:
            from langchain_google_vertexai import ChatVertexAI
        except ImportError as exc:
            raise ImportError("The Vertex backend needs: pip install 'finrag[vertex]'") from exc
        if not settings.gcp_project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is not set. The Vertex backend bills a Cloud project, "
                "so it must know which one. Set it in .env, and authenticate once with: "
                "gcloud auth application-default login"
            )
        params.setdefault("max_output_tokens", settings.max_output_tokens)
        params.setdefault("project", settings.gcp_project)
        params.setdefault("location", settings.gcp_location)
        params.pop("max_tokens", None)
        return ChatVertexAI(model=model, **params)

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
        # Ollama names the output limit num_predict and defaults it to about
        # 128 tokens. Dropping max_tokens without translating it left local
        # models answering in a fraction of the space every other backend got
        # -- a comparison would have measured that truncation, not the model.
        params.setdefault("num_predict", settings.max_output_tokens)
        params.pop("max_tokens", None)
        # Retries are handled by the caller for local models; the parameter is
        # not part of the ChatOllama signature.
        params.pop("max_retries", None)
        return ChatOllama(model=model, **params)

    raise ValueError(f"unknown llm backend {backend!r}; expected one of {all_backends()}")


def _openai_compatible(backend: str, settings: Settings, **overrides: Any) -> Any:
    """Client for any endpoint speaking the OpenAI chat-completions protocol."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            f"The {backend} backend needs: pip install 'finrag[openai-compatible]'"
        ) from exc

    preset = PROVIDER_PRESETS[backend]
    key = os.environ.get(preset.key_env) or os.environ.get("FINRAG_LLM_API_KEY")
    if not key:
        raise RuntimeError(
            f"{preset.key_env} is not set. The {backend} backend needs it; "
            f"see {preset.base_url} for where to get a key."
        )
    params: dict[str, Any] = {
        "temperature": 0,
        "max_tokens": settings.max_output_tokens,
        "max_retries": MAX_RETRIES,
        **overrides,
    }
    limiter = _rate_limiter(backend, settings)
    if limiter is not None:
        params.setdefault("rate_limiter", limiter)
    return ChatOpenAI(
        model=settings.chat_model or preset.default_model,
        base_url=settings.openai_base_url or preset.base_url,
        api_key=key,
        **params,
    )


def build_with_fallbacks(settings: Settings | None = None, **overrides: Any) -> Any:
    """The configured chat model, falling back through FINRAG_LLM_FALLBACKS.

    Free tiers fail by quota as much as by outage, and each provider's quota is
    independent -- so a fallback chain makes the usable budget the union of the
    tiers. Each fallback backend is built with its *own* default model, never
    the primary's FINRAG_CHAT_MODEL, because model ids do not transfer between
    providers.

    Used on plain-chat paths (the RAGAS generator and judge). The tool-calling
    agent keeps a single backend: ``create_tool_calling_agent`` must call
    ``bind_tools`` on the model, and a ``RunnableWithFallbacks`` does not expose
    it, so wiring fallbacks there would mean rebuilding the agent per provider
    for marginal benefit.
    """
    settings = settings or get_settings()
    primary = get_chat_model(settings, **overrides)

    names = [n.strip().lower() for n in settings.llm_fallbacks.split(",") if n.strip()]
    names = [n for n in names if n != settings.llm_backend.lower()]
    if not names:
        return primary

    fallbacks = [
        get_chat_model(replace(settings, llm_backend=name, chat_model=""), **overrides)
        for name in names
    ]
    return primary.with_fallbacks(fallbacks)


def get_judge_model(settings: Settings | None = None, **overrides: Any) -> tuple[Any, str]:
    """The model used to SCORE answers, and a label naming it.

    Defaults to the generating backend, which is fine in isolation. For a
    cross-backend comparison it must be pinned to one provider via
    FINRAG_JUDGE_BACKEND: otherwise every backend judges its own output, and
    the resulting ranking measures judge self-preference as much as answer
    quality. The returned label is recorded alongside the metrics so a reader
    can tell which regime produced a given number.
    """
    settings = settings or get_settings()
    backend = (settings.judge_backend or settings.llm_backend).lower()
    model = settings.judge_model or (
        settings.chat_model
        if backend == settings.llm_backend.lower() and not settings.judge_backend
        else ""
    )
    judge_settings = replace(settings, llm_backend=backend, chat_model=model)
    label = f"{backend}/{model or default_model_for(backend)}"
    return get_chat_model(judge_settings, **overrides), label


def all_backends() -> list[str]:
    """Every selectable backend name."""
    return sorted(set(DEFAULT_MODELS) | set(PROVIDER_PRESETS))


def default_model_for(backend: str) -> str:
    """The model used when FINRAG_CHAT_MODEL is blank."""
    backend = backend.lower()
    if backend in PROVIDER_PRESETS:
        return PROVIDER_PRESETS[backend].default_model
    return DEFAULT_MODELS.get(backend, "")


def required_api_key(settings: Settings | None = None) -> str | None:
    """Environment variable the configured backend reads, or None if it needs no key."""
    settings = settings or get_settings()
    backend = settings.llm_backend.lower()
    if backend in KEYLESS_BACKENDS:
        return None
    if backend in PROVIDER_PRESETS:
        return PROVIDER_PRESETS[backend].key_env
    keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
    }
    if backend not in keys:
        raise ValueError(f"unknown llm backend {backend!r}")
    return keys[backend]
