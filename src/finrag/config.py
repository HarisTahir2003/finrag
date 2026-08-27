"""Runtime configuration, read from the environment.

Deliberately a plain dataclass rather than a settings framework: it has no
dependencies, it is trivial to construct in a test, and every value has a
working default so a fresh clone runs without a .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TICKERS = ("AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "JPM", "V")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Everything the pipeline needs to know about its environment."""

    data_root: Path = field(default_factory=lambda: Path(_env("FINRAG_DATA_ROOT", "./data")))

    # "local" needs no API key and no network at query time; "google" is higher
    # quality but costs a call per chunk. Local is the default so that tests, CI
    # and a fresh clone all work with no credentials.
    embedding_backend: str = field(default_factory=lambda: _env("FINRAG_EMBEDDINGS", "local"))
    local_embedding_model: str = field(
        default_factory=lambda: _env(
            "FINRAG_LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    google_embedding_model: str = field(
        default_factory=lambda: _env("FINRAG_GOOGLE_EMBED_MODEL", "models/text-embedding-004")
    )
    chat_model: str = field(default_factory=lambda: _env("FINRAG_CHAT_MODEL", "gemini-2.5-pro"))

    # "semantic" chunks on document structure via unstructured; "recursive" is
    # the faster fixed-width splitter. See finrag.chunking for the trade-off.
    chunk_strategy: str = field(default_factory=lambda: _env("FINRAG_CHUNK_STRATEGY", "semantic"))
    chunk_size: int = field(default_factory=lambda: int(_env("FINRAG_CHUNK_SIZE", "3000")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("FINRAG_CHUNK_OVERLAP", "300")))

    collection_name: str = field(default_factory=lambda: _env("FINRAG_COLLECTION", "sec_10k"))
    retrieval_k: int = field(default_factory=lambda: int(_env("FINRAG_RETRIEVAL_K", "20")))

    sec_contact_email: str = field(default_factory=lambda: _env("SEC_CONTACT_EMAIL", ""))
    sec_company_name: str = field(default_factory=lambda: _env("SEC_COMPANY_NAME", "finrag"))

    @property
    def filings_dir(self) -> Path:
        return self.data_root / "sec_filings"

    @property
    def index_dir(self) -> Path:
        # The backend is part of the path: local and Google embeddings produce
        # vectors of different dimensions and must never share a collection.
        return self.data_root / f"chroma_{self.embedding_backend}"

    def require_sec_contact(self) -> str:
        """SEC EDGAR rejects or throttles requests without a real contact address."""
        if not self.sec_contact_email:
            raise RuntimeError(
                "SEC_CONTACT_EMAIL is not set. EDGAR requires a real contact address in the "
                "User-Agent header of every request. Copy .env.example to .env and fill it in."
            )
        return self.sec_contact_email


def get_settings() -> Settings:
    return Settings()
