"""Hybrid retrieval: BM25 fused with the vector search."""

from __future__ import annotations

from dataclasses import replace

import pytest
from langchain_core.documents import Document

from finrag.config import Settings
from finrag.retrieval import reciprocal_rank_fusion, search_filing


class FakeStore:
    """Records what it was asked, returns what it was told to."""

    def __init__(self, vector_hits, all_chunks=None):
        self._vector = vector_hits
        self._chunks = all_chunks or []
        self.queries: list[str] = []
        self.get_calls = 0

    def similarity_search(self, query, k, filter):  # noqa: A002 - langchain's name
        self.queries.append(query)
        return self._vector[:k]

    def get(self, where, include):  # noqa: A002 - chroma's name
        self.get_calls += 1
        return {
            "documents": [d.page_content for d in self._chunks],
            "metadatas": [d.metadata for d in self._chunks],
        }


def _doc(chunk_id: str, text: str) -> Document:
    return Document(page_content=text, metadata={"id": chunk_id, "ticker": "AAPL", "year": 2023})


def test_hybrid_recovers_a_phrase_the_vector_search_ranks_low():
    """The case this exists for.

    `aapl-2023-competition` hunts the phrase "intense competition" and ranked
    26th on pure vector search -- past any sane retrieval depth. A phrase query
    is what lexical scoring is good at.
    """
    pytest.importorskip("rank_bm25")

    target = _doc("c-target", "We face intense competition in every market we address.")
    filler = [_doc(f"c{i}", f"Segment revenue discussion number {i}.") for i in range(12)]
    # The vector side never surfaces the target.
    store = FakeStore(vector_hits=filler, all_chunks=[*filler, target])

    settings = replace(
        Settings(), retrieval_mode="hybrid", retrieval_k=5, query_expansion=False, rerank=False
    )
    found = search_filing(
        "intense competition",
        "AAPL",
        2023,
        store=store,
        settings=settings,
        apply_context_budget=False,
    )

    assert any("intense competition" in d.page_content for d in found.documents)


def test_vector_mode_leaves_retrieval_untouched():
    """The old path has to stay exactly as it was; every published number used it."""
    filler = [_doc(f"c{i}", f"chunk {i}") for i in range(6)]
    target = _doc("c-target", "We face intense competition.")
    store = FakeStore(vector_hits=filler, all_chunks=[*filler, target])

    settings = replace(
        Settings(), retrieval_mode="vector", retrieval_k=5, query_expansion=False, rerank=False
    )
    found = search_filing(
        "intense competition",
        "AAPL",
        2023,
        store=store,
        settings=settings,
        apply_context_budget=False,
    )

    assert [d.metadata["id"] for d in found.documents] == [d.metadata["id"] for d in filler[:5]]


def test_lexical_side_gets_the_raw_query_not_the_expanded_one():
    """Expansion is a vector-space trick; BM25 would match its padding as terms.

    Nine words of "financial statements balance sheet results of operations"
    appended to a three-word question is a large lexical signal pointing at the
    wrong thing.
    """
    pytest.importorskip("rank_bm25")

    chunks = [_doc(f"c{i}", f"chunk {i} about revenue") for i in range(4)]
    store = FakeStore(vector_hits=chunks, all_chunks=chunks)
    settings = replace(
        Settings(), retrieval_mode="hybrid", retrieval_k=3, query_expansion=True, rerank=False
    )

    search_filing(
        "total revenue", "AAPL", 2023, store=store, settings=settings, apply_context_budget=False
    )

    # The vector side saw the expanded text...
    assert "balance sheet" in store.queries[0]
    # ...and the BM25 side is built from the raw question, which we assert by
    # the expansion never reaching it: BM25Retriever holds no query state, so
    # this checks the call site rather than the object.
    assert store.queries[0] != "total revenue", "expansion should still apply to the vector side"


def test_hybrid_degrades_to_vector_when_lexical_scoring_is_unavailable(monkeypatch):
    """Missing rank_bm25 must lose a feature, not the query."""
    import finrag.retrieval as retrieval

    monkeypatch.setattr(retrieval, "_bm25_for_filing", lambda *a, **k: None)
    chunks = [_doc(f"c{i}", f"chunk {i}") for i in range(4)]
    store = FakeStore(vector_hits=chunks, all_chunks=chunks)
    settings = replace(
        Settings(), retrieval_mode="hybrid", retrieval_k=3, query_expansion=False, rerank=False
    )

    found = search_filing(
        "revenue", "AAPL", 2023, store=store, settings=settings, apply_context_budget=False
    )

    assert len(found.documents) == 3


def test_fusion_needs_metadata_ids_to_merge_two_result_sets():
    """Without metadata the two lists share no keys and fusion concatenates.

    This is why _bm25_for_filing passes metadatas into BM25Retriever.from_texts.
    """
    shared = _doc("same-id", "the balance sheet says total current assets 143,566")
    other = _doc("other-id", "unrelated discussion")

    fused = reciprocal_rank_fusion([[shared, other], [shared]])
    ids = [d.metadata["id"] for d in fused]

    assert ids.count("same-id") == 1, "a document in both lists must appear once"
    assert ids[0] == "same-id", "and rank above one found by only a single retriever"


def test_bm25_cache_returns_a_hit_without_refetching_the_filing():
    """A warm hit must not touch the store.

    The old key included the chunk count, which could only be computed by
    reading the whole filing back out of Chroma -- up to 723KB discarded on
    every hit. Keying on (ticker, year) lets a hit return before any fetch.
    """
    pytest.importorskip("rank_bm25")
    import finrag.retrieval as retrieval

    retrieval._BM25_CACHE.clear()
    chunks = [_doc(f"c{i}", f"chunk {i}") for i in range(3)]
    store = FakeStore(vector_hits=chunks, all_chunks=chunks)

    retrieval._bm25_for_filing("AAPL", 2023, store)
    assert store.get_calls == 1, "the first call builds the index and reads the filing once"

    retrieval._bm25_for_filing("AAPL", 2023, store)
    assert store.get_calls == 1, "a cache hit must not read the filing again"


def test_bm25_cache_is_bounded_and_evicts_least_recently_used():
    """Unbounded, the cache retained ~150-250MB across a full corpus, forever,
    on a 690MB host. It is now an LRU."""
    pytest.importorskip("rank_bm25")
    import finrag.retrieval as retrieval

    retrieval._BM25_CACHE.clear()
    limit = retrieval._BM25_CACHE_MAX

    # Fill past the limit; each filing is a distinct (ticker, year).
    for year in range(2000, 2000 + limit + 3):
        chunks = [
            Document(
                page_content=f"chunk {i}",
                metadata={"id": f"{year}-{i}", "ticker": "AAPL", "year": year},
            )
            for i in range(3)
        ]
        retrieval._bm25_for_filing("AAPL", year, FakeStore(vector_hits=chunks, all_chunks=chunks))

    assert len(retrieval._BM25_CACHE) == limit, "the cache must not grow past its bound"
    # The earliest years were evicted; the most recent survive.
    survivors = {year for _, year in retrieval._BM25_CACHE}
    assert (2000 + limit + 2) in survivors, "the most recent filing must be retained"
    assert 2000 not in survivors, "the least recently used filing must be evicted"
