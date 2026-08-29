"""Embedding backend selection and caching.

The caching is not an optimisation detail. ``open_store`` calls
``get_embeddings`` on every call and ``collection_size`` calls ``open_store``
just to count rows, so an uncached constructor made the API's readiness probe
load a whole sentence-transformer every thirty seconds. These tests fake the
provider module so they run without the [local] extra installed -- otherwise
the regression they guard would be untested in the CI job that runs pytest.
"""

from __future__ import annotations

import sys
import types

import pytest

import finrag.embeddings as emb
from finrag.config import Settings


@pytest.fixture(autouse=True)
def _empty_cache(monkeypatch):
    """Each test starts with a cold cache and leaves nothing behind."""
    monkeypatch.setattr(emb, "_MODELS", {})


def _fake_provider(monkeypatch, module: str, attribute: str) -> list[str]:
    """Install a stand-in provider and return the list it records models into."""
    constructed: list[str] = []

    class Fake:
        def __init__(self, **kwargs):
            self.name = kwargs.get("model_name") or kwargs.get("model")
            constructed.append(self.name)

    monkeypatch.setitem(sys.modules, module, types.SimpleNamespace(**{attribute: Fake}))
    return constructed


def test_the_same_model_is_constructed_once(monkeypatch):
    constructed = _fake_provider(monkeypatch, "langchain_huggingface", "HuggingFaceEmbeddings")
    settings = Settings(embedding_backend="local", local_embedding_model="tiny")

    first = emb.get_embeddings(settings)
    second = emb.get_embeddings(settings)

    assert first is second
    assert constructed == ["tiny"], "the readiness probe must not reload the model per call"


def test_a_different_model_is_a_separate_entry(monkeypatch):
    constructed = _fake_provider(monkeypatch, "langchain_huggingface", "HuggingFaceEmbeddings")

    small = emb.get_embeddings(Settings(embedding_backend="local", local_embedding_model="small"))
    large = emb.get_embeddings(Settings(embedding_backend="local", local_embedding_model="large"))

    # A sweep that switches models in one process must not silently reuse the
    # previous one -- the vectors would be incomparable.
    assert small is not large
    assert constructed == ["small", "large"]


def test_backends_do_not_share_a_cache_entry(monkeypatch):
    _fake_provider(monkeypatch, "langchain_huggingface", "HuggingFaceEmbeddings")
    _fake_provider(monkeypatch, "langchain_google_genai", "GoogleGenerativeAIEmbeddings")

    local = emb.get_embeddings(Settings(embedding_backend="local", local_embedding_model="m"))
    google = emb.get_embeddings(Settings(embedding_backend="google", google_embedding_model="m"))

    # Same model name, different backend, different vector space.
    assert local is not google


def test_unknown_backend_raises(monkeypatch):
    with pytest.raises(ValueError, match="unknown embedding backend"):
        emb.get_embeddings(Settings(embedding_backend="pinecone"))


def test_a_missing_provider_names_the_extra_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_huggingface", None)
    with pytest.raises(ImportError, match=r"finrag\[local\]"):
        emb.get_embeddings(Settings(embedding_backend="local"))
