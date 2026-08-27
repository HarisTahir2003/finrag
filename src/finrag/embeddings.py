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

from typing import Any

from .config import Settings, get_settings


def get_embeddings(settings: Settings | None = None) -> Any:
    settings = settings or get_settings()
    backend = settings.embedding_backend.lower()

    if backend == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Local embeddings need sentence-transformers. Install with: pip install 'finrag[local]'"
            ) from exc
        return HuggingFaceEmbeddings(
            model_name=settings.local_embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

    if backend == "google":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Google embeddings need langchain-google-genai. Install with: pip install 'finrag[google]'"
            ) from exc
        return GoogleGenerativeAIEmbeddings(model=settings.google_embedding_model)

    raise ValueError(f"unknown embedding backend {backend!r}; expected 'local' or 'google'")
