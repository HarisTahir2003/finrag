"""Splitting a filing into retrievable chunks.

Two strategies, because they trade off against each other:

``semantic``
    ``unstructured`` partitions the HTML into titles, paragraphs and tables, then
    ``chunk_by_title`` groups them so a chunk boundary falls on a section
    heading rather than mid-sentence. Better retrieval, materially slower on a
    multi-megabyte 10-K.

``recursive``
    Fixed-width splitting with overlap. Fast and dependency-light, but it will
    cut a balance sheet in half without noticing.

The original notebook imported ``partition_html`` and ``chunk_by_title`` and
then never called either, splitting on fixed widths instead while describing
itself as doing semantic partitioning. Both paths are now real and selectable.
"""

from __future__ import annotations

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)


def chunk_recursive(text: str, chunk_size: int = 3000, chunk_overlap: int = 300) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def chunk_semantic(html: str, chunk_size: int = 3000, chunk_overlap: int = 300) -> list[str]:
    """Structure-aware chunking, with recursive chunking as the safety net.

    ``unstructured`` fails two ways and both have to land in the same place. It
    may be absent, because the optional extra was never installed. Or it may be
    installed and still unusable at call time: current versions fetch a spaCy
    sentence model on first use, so a machine that is offline, behind a proxy,
    or missing a CA bundle raises from deep inside ``partition_html``.

    Only the first of those used to be handled, which put the failure in the
    worst possible spot -- chunking runs after every filing has been downloaded
    and parsed, so aborting there discards the slow part of the pipeline over an
    optional improvement in chunk quality. Recursive chunking always works and
    needs nothing, so any failure degrades to it and says so.
    """
    try:
        from unstructured.chunking.title import chunk_by_title
        from unstructured.partition.html import partition_html

        elements = partition_html(text=html)
        chunks = chunk_by_title(
            elements,
            max_characters=chunk_size,
            combine_text_under_n_chars=chunk_size // 4,
            overlap=chunk_overlap,
        )
        return [str(c) for c in chunks if str(c).strip()]
    except ImportError:
        log.warning(
            "unstructured is not installed; falling back to recursive chunking. "
            "Install it with: pip install 'finrag[semantic]'"
        )
    except Exception as exc:  # noqa: BLE001 - degrade on anything, never abort the ingest
        log.warning(
            "semantic chunking failed (%s: %s); falling back to recursive chunking. "
            "Set FINRAG_CHUNK_STRATEGY=recursive to choose this deliberately.",
            type(exc).__name__,
            exc,
        )

    from .ingest.parse import html_to_text

    return chunk_recursive(html_to_text(html), chunk_size, chunk_overlap)


def chunk_filing(
    html: str,
    text: str,
    strategy: str = "semantic",
    chunk_size: int = 3000,
    chunk_overlap: int = 300,
) -> list[str]:
    """Chunk one filing using the configured strategy.

    Both the HTML and the flattened text are passed in because the two
    strategies need different inputs and neither should re-do the other's work.
    """
    if strategy == "semantic":
        return chunk_semantic(html, chunk_size, chunk_overlap)
    if strategy == "recursive":
        return chunk_recursive(text, chunk_size, chunk_overlap)
    raise ValueError(f"unknown chunk strategy {strategy!r}; expected 'semantic' or 'recursive'")
