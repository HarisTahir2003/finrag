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

from .config import Settings, get_settings

log = logging.getLogger(__name__)

_enabled = False


def enable_llm_cache(settings: Settings | None = None) -> bool:
    """Turn on the SQLite-backed LLM cache. Idempotent. Returns True if active."""
    global _enabled
    if _enabled:
        return True

    settings = settings or get_settings()
    if not settings.llm_cache:
        return False

    try:
        from langchain_community.cache import SQLiteCache
        from langchain_core.globals import set_llm_cache
    except ImportError:
        log.warning("langchain-community not installed; LLM cache disabled")
        return False

    settings.data_root.mkdir(parents=True, exist_ok=True)
    path = settings.data_root / "llm_cache.db"
    set_llm_cache(SQLiteCache(database_path=str(path)))
    _enabled = True
    log.info("LLM cache on: %s", path)
    return True
