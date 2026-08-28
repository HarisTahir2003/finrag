"""Response caching for LLM calls.

On a free tier the scarce resource is quota, and the most wasteful way to spend
it is re-asking a question that was already answered. With the cache on, an
evaluation re-run over unchanged cases costs zero tokens: identical
(prompt, model, bound tools) triples are served from SQLite. Change the prompt,
the model or the retrieved context and the entry no longer matches, so nothing
stale is ever returned.

This pairs with checkpointing in the eval harness: checkpoints skip *completed*
cases, the cache absorbs *repeated* calls inside cases that do rerun -- RAGAS
judge retries most of all.
"""

from __future__ import annotations

import logging
import re
import warnings

from .config import Settings, get_settings

log = logging.getLogger(__name__)

# The backend and model whose cache is currently installed, or None. Tracked so
# that switching backends mid-process re-points the cache instead of silently
# reusing the previous one.
_installed: str | None = None


def cache_path(settings: Settings):
    """Where this backend and model's cache lives.

    One file per (backend, model) rather than one shared file, because
    LangChain's cache key is whatever the client reports via
    ``_get_llm_string()`` and not every client includes the model. ChatOllama
    reports ``[('_type', 'chat-ollama'), ('stop', None)]`` for every model it
    serves, so a single shared cache would hand qwen3:4b's answers back for a
    llama3.3:70b query -- and in a backend comparison, one model's output would
    be published under another's name. Namespacing by file sidesteps the whole
    class of problem rather than trusting each client to key correctly.
    """
    from .llm import default_model_for

    model = settings.chat_model or default_model_for(settings.llm_backend)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{settings.llm_backend}-{model}".lower()).strip("-")
    return settings.data_root / f"llm_cache-{slug}.db"


def enable_llm_cache(settings: Settings | None = None) -> bool:
    """Turn on the SQLite-backed LLM cache. Returns True if active."""
    global _installed

    settings = settings or get_settings()
    if not settings.llm_cache:
        return False

    try:
        from langchain_community.cache import SQLiteCache
        from langchain_core.globals import set_llm_cache
    except ImportError:
        log.warning("langchain-community not installed; LLM cache disabled")
        return False

    path = cache_path(settings)
    if _installed == str(path):
        return True

    settings.data_root.mkdir(parents=True, exist_ok=True)

    # One warning per cache *read*, which on a fully-cached RAGAS run is
    # thousands of lines and buries the scores the run exists to produce.
    # It cannot be fixed properly from here: the warning asks callers to pass
    # `allowed_objects` to `loads()`, and SQLiteCache takes only a
    # database_path -- it calls `loads()` itself with no way through. Matched on
    # the message rather than the category so it stays narrow; every other
    # deprecation still surfaces.
    warnings.filterwarnings(
        "ignore",
        message=r".*allowed_objects.*",
    )

    set_llm_cache(SQLiteCache(database_path=str(path)))
    _installed = str(path)
    log.info("LLM cache on: %s", path)
    return True
