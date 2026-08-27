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
    """Structure-aware chunking. Falls back to recursive if unstructured is absent."""
    try:
        from unstructured.chunking.title import chunk_by_title
        from unstructured.partition.html import partition_html
    except ImportError:
        log.warning(
            "unstructured is not installed; falling back to recursive chunking. "
            "Install it with: pip install 'finrag[semantic]'"
        )
        from .ingest.parse import html_to_text

        return chunk_recursive(html_to_text(html), chunk_size, chunk_overlap)

    elements = partition_html(text=html)
    chunks = chunk_by_title(
        elements,
        max_characters=chunk_size,
        combine_text_under_n_chars=chunk_size // 4,
        overlap=chunk_overlap,
    )
    return [str(c) for c in chunks if str(c).strip()]


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
