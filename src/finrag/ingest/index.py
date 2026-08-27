"""Building the vector index.

Two things changed from the original loop.

It called ``Chroma.from_documents(...)`` once per file, inside the loop, which
constructs a fresh client each time and appends unconditionally -- so running
the ingest twice produced two copies of every chunk with no way to tell them
apart. The store is now opened once and chunks are upserted.

Every chunk gets a deterministic ID derived from its ticker, fiscal year,
position and content. Re-running over unchanged filings therefore rewrites the
same rows instead of duplicating them, which makes ingestion resumable and
makes incremental updates possible.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_core.documents import Document

from ..chunking import chunk_filing
from ..config import Settings, get_settings
from ..embeddings import get_embeddings
from .download import list_filings
from .metadata import FilingMetadata, filing_metadata
from .parse import extract_primary_document, html_to_text

log = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 128


def chunk_id(meta: FilingMetadata, position: int, content: str) -> str:
    """Stable identity for a chunk: same filing and same text produce the same ID."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"{meta.ticker}:{meta.fiscal_year}:{position:05d}:{digest}"


def open_store(settings: Settings | None = None):
    """Open (or create) the persistent Chroma collection."""
    from langchain_chroma import Chroma

    settings = settings or get_settings()
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.index_dir),
    )


def documents_for_filing(path: str | Path, settings: Settings | None = None) -> list[Document]:
    """Parse and chunk one filing into documents carrying correct metadata.

    Returns an empty list when the primary document cannot be found. Raises if
    the fiscal year cannot be determined -- see ingest.metadata for why guessing
    is worse than failing.
    """
    settings = settings or get_settings()
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")

    meta = filing_metadata(path, text=raw)

    html = extract_primary_document(raw, form_type="10-K")
    if html is None:
        log.warning("%s: no 10-K primary document found; skipping", path)
        return []

    text = html_to_text(html)
    chunks = chunk_filing(
        html=html,
        text=text,
        strategy=settings.chunk_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    base = meta.as_chroma_metadata()
    return [
        Document(page_content=c, metadata={**base, "chunk_index": i, "id": chunk_id(meta, i, c)})
        for i, c in enumerate(chunks)
    ]


def _delete_filing(store, ticker: str, fiscal_year: int) -> int:
    """Remove every chunk already indexed for one filing. Returns how many."""
    where = {"$and": [{"ticker": ticker}, {"year": int(fiscal_year)}]}
    try:
        existing = store._collection.get(where=where, include=[])
        ids = existing.get("ids") or []
        if ids:
            store._collection.delete(ids=ids)
            log.debug("%s FY%s: removed %d stale chunks", ticker, fiscal_year, len(ids))
        return len(ids)
    except Exception as exc:  # noqa: BLE001 - a failed clean-up must not stop ingest
        log.warning("could not clear existing chunks for %s FY%s: %s", ticker, fiscal_year, exc)
        return 0


def index_filings(
    paths: list[Path] | None = None, settings: Settings | None = None
) -> dict[str, int]:
    """Index every filing, upserting so repeated runs converge instead of growing."""
    settings = settings or get_settings()
    paths = paths if paths is not None else list_filings(settings=settings)
    if not paths:
        log.warning(
            "no filings found under %s -- run the download step first", settings.filings_dir
        )
        return {"filings": 0, "chunks": 0}

    store = open_store(settings)
    total_chunks = 0
    indexed = 0

    for path in paths:
        try:
            docs = documents_for_filing(path, settings)
        except ValueError as exc:
            log.error("skipping %s: %s", path, exc)
            continue
        if not docs:
            # Loud, because this is how a whole corpus quietly becomes an empty
            # index: every filing parses, every filing yields nothing, and the
            # run still reports success.
            log.warning("%s: parsed but produced no chunks; skipping", path)
            continue

        # Replace, do not merely upsert. Chunk ids hash the chunk's text, so
        # re-indexing after a chunk_size or chunk_strategy change produces a
        # disjoint id set: the upsert adds a second chunking of the filing
        # while the first stays behind. The collection then holds the same
        # filing twice at two granularities, which inflates every retrieval and
        # is invisible except as a doubled chunk count.
        first = docs[0].metadata
        _delete_filing(store, first["ticker"], first["year"])

        for start in range(0, len(docs), EMBED_BATCH_SIZE):
            batch = docs[start : start + EMBED_BATCH_SIZE]
            store.add_documents(batch, ids=[d.metadata["id"] for d in batch])

        log.info("%s FY%s: %d chunks", first["ticker"], first["year"], len(docs))
        total_chunks += len(docs)
        indexed += 1

    if paths and not indexed:
        log.error(
            "%d filing(s) on disk but none could be indexed -- the index is empty. "
            "Try FINRAG_CHUNK_STRATEGY=recursive to rule out the chunker.",
            len(paths),
        )

    return {"filings": indexed, "chunks": total_chunks}


def collection_size(settings: Settings | None = None) -> int:
    """How many chunks are currently in the index."""
    store = open_store(settings or get_settings())
    return store._collection.count()
