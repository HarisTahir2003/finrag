from __future__ import annotations

from finrag.config import get_settings


def test_defaults_need_no_environment(monkeypatch):
    """A fresh clone with no .env must still produce usable settings."""
    for var in (
        "FINRAG_DATA_ROOT",
        "FINRAG_EMBEDDINGS",
        "FINRAG_CHUNK_STRATEGY",
        "SEC_CONTACT_EMAIL",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = get_settings()
    assert settings.embedding_backend == "local", "local needs no API key, so it is the default"
    assert str(settings.data_root) == "data"


def test_index_path_separates_embedding_backends(monkeypatch):
    """Local and Google vectors have different dimensions and must not mix."""
    monkeypatch.setenv("FINRAG_EMBEDDINGS", "local")
    local = get_settings().index_dir
    monkeypatch.setenv("FINRAG_EMBEDDINGS", "google")
    assert get_settings().index_dir != local


def test_sec_contact_is_required_only_when_downloading(monkeypatch):
    monkeypatch.delenv("SEC_CONTACT_EMAIL", raising=False)
    settings = get_settings()
    assert settings.embedding_backend  # constructing settings must not raise
    try:
        settings.require_sec_contact()
    except RuntimeError as exc:
        assert "SEC_CONTACT_EMAIL" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when the contact address is missing")
