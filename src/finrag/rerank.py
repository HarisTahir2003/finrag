"""Cross-encoder reranking.

Retrieval scores a query and a chunk by comparing two vectors computed
independently of each other -- fast, because every chunk's vector is computed
once at index time, and lossy for the same reason: the chunk was embedded
without knowing what would be asked of it.

A cross-encoder reads the pair together and scores it directly. That cannot be
precomputed, so it is far too slow to run over a corpus, and exactly right for
reordering the fifty candidates retrieval already narrowed to.

The measured problem it targets: mrr 0.720 means the answering chunk is usually
found but ranked second to seventh, and context_precision 0.113 means roughly
two chunks in twenty are relevant. Both are ranking failures rather than recall
failures, which is what reranking is for.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import Settings, get_settings

log = logging.getLogger(__name__)

# Loading a cross-encoder takes seconds and megabytes, and a retrieval eval
# calls this thirty times. Keyed by model name so switching models in one
# process -- a sweep -- does not silently reuse the previous one.
_MODELS: dict[str, Any] = {}


def get_reranker(settings: Settings | None = None):
    """The cross-encoder named by settings, or None if it cannot be loaded.

    Returns None rather than raising: reranking is an improvement to ranking,
    and losing it should cost quality, not the query. Mirrors the degradation
    in chunking.chunk_semantic and retrieval._bm25_for_filing.
    """
    settings = settings or get_settings()
    name = settings.rerank_model

    if name in _MODELS:
        return _MODELS[name]

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        log.warning(
            "reranking needs sentence-transformers (pip install 'finrag[local]'); "
            "returning the retriever's own ordering"
        )
        return None

    try:
        model = CrossEncoder(name)
    except Exception as exc:  # noqa: BLE001 - a missing download must not fail the query
        log.warning("could not load reranker %s (%s); keeping the retriever's ordering", name, exc)
        return None

    log.info("reranker loaded: %s", name)
    _MODELS[name] = model
    return model


def rerank(query: str, documents: list, top_k: int, settings: Settings | None = None) -> list:
    """Reorder documents by cross-encoder relevance and keep the best ``top_k``.

    Returns the input order, truncated, when no reranker is available -- so a
    caller never has to check whether reranking happened.
    """
    if not documents:
        return documents

    model = get_reranker(settings)
    if model is None:
        return documents[:top_k]

    try:
        scores = model.predict([(query, d.page_content) for d in documents])
        # strict=True: one score per document is the model's contract, and a
        # silent zip truncation would drop candidates without saying so.
        order = sorted(
            zip(documents, scores, strict=True), key=lambda pair: float(pair[1]), reverse=True
        )
    except Exception as exc:  # noqa: BLE001 - degrade to the retriever's ordering
        log.warning("reranking failed (%s); keeping the retriever's ordering", exc)
        return documents[:top_k]

    if log.isEnabledFor(logging.DEBUG):
        log.debug("reranked %d candidates to %d", len(documents), min(top_k, len(order)))
    return [doc for doc, _ in order[:top_k]]
