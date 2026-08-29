"""Embedding backends.

``local`` runs sentence-transformers on the machine: no API key, no per-chunk
cost, and therefore usable in CI and by anyone who clones the repo. ``google``
uses text-embedding-004 and is higher quality but needs credentials and bills
per call.

The two produce vectors of different dimensions, so they can never share a
collection. Settings.index_dir puts the backend name in the path to make that
impossible by construction.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import Settings, get_settings

log = logging.getLogger(__name__)

# Constructing HuggingFaceEmbeddings loads a sentence-transformer -- ~90MB of
# weights and a second or two -- and nothing upstream memoises it. open_store()
# calls this on every call, and collection_size() calls open_store() just to
# count rows, so the API's readiness probe was loading an entire model every 30
# seconds, forever. Keyed by (backend, model) so a sweep that switches models in
# one process does not silently reuse the previous one -- the same contract as
# rerank._MODELS.
#
# Sharing one instance is safe because inference holds no per-call state: the
# object is only ever asked to encode, exactly as the cross-encoder is.
_MODELS: dict[tuple[str, str], Any] = {}


def get_embeddings(settings: Settings | None = None) -> Any:
    settings = settings or get_settings()
    backend = settings.embedding_backend.lower()

    if backend == "local":
        name = settings.local_embedding_model
    elif backend == "google":
        name = settings.google_embedding_model
    else:
        raise ValueError(f"unknown embedding backend {backend!r}; expected 'local' or 'google'")

    key = (backend, name)
    if key in _MODELS:
        return _MODELS[key]

    if backend == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Local embeddings need sentence-transformers. Install with: pip install 'finrag[local]'"
            ) from exc
        model = HuggingFaceEmbeddings(
            model_name=name,
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Google embeddings need langchain-google-genai. Install with: pip install 'finrag[google]'"
            ) from exc
        model = GoogleGenerativeAIEmbeddings(model=name)

    log.debug("embeddings loaded: %s/%s", backend, name)
    _MODELS[key] = model
    return model
