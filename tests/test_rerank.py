"""Cross-encoder reranking."""

from __future__ import annotations

from dataclasses import replace

from langchain_core.documents import Document

from finrag.config import Settings
from finrag.rerank import rerank
from finrag.retrieval import search_filing


def _doc(chunk_id: str, text: str) -> Document:
    return Document(page_content=text, metadata={"id": chunk_id, "ticker": "AAPL", "year": 2023})


class FakeCrossEncoder:
    """Scores by a lookup table, so the expected order is unambiguous."""

    def __init__(self, scores_by_text):
        self._scores = scores_by_text
        self.seen: list[tuple[str, str]] = []

    def predict(self, pairs):
        self.seen = list(pairs)
        return [self._scores.get(text, 0.0) for _, text in pairs]


def test_rerank_promotes_the_best_scoring_chunk(monkeypatch):
    import finrag.rerank as rr

    docs = [_doc("a", "alpha"), _doc("b", "beta"), _doc("c", "gamma")]
    monkeypatch.setattr(
        rr,
        "get_reranker",
        lambda s=None: FakeCrossEncoder({"gamma": 9.0, "alpha": 1.0, "beta": 0.5}),
    )

    out = rerank("q", docs, top_k=2)

    assert [d.metadata["id"] for d in out] == ["c", "a"]


def test_rerank_without_a_model_keeps_the_retriever_ordering(monkeypatch):
    """Losing the reranker must cost quality, not the query."""
    import finrag.rerank as rr

    monkeypatch.setattr(rr, "get_reranker", lambda s=None: None)
    docs = [_doc(str(i), f"chunk {i}") for i in range(5)]

    out = rerank("q", docs, top_k=3)

    assert [d.metadata["id"] for d in out] == ["0", "1", "2"]


def test_rerank_survives_a_model_that_raises(monkeypatch):
    import finrag.rerank as rr

    class Broken:
        def predict(self, pairs):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(rr, "get_reranker", lambda s=None: Broken())
    docs = [_doc(str(i), f"chunk {i}") for i in range(4)]

    assert [d.metadata["id"] for d in rerank("q", docs, top_k=2)] == ["0", "1"]


def test_rerank_scores_the_query_against_each_chunk(monkeypatch):
    """The point of a cross-encoder: the pair is scored jointly, not separately."""
    import finrag.rerank as rr

    model = FakeCrossEncoder({})
    monkeypatch.setattr(rr, "get_reranker", lambda s=None: model)
    docs = [_doc("a", "alpha"), _doc("b", "beta")]

    rerank("what was revenue", docs, top_k=2)

    assert model.seen == [("what was revenue", "alpha"), ("what was revenue", "beta")]


class _Store:
    def __init__(self, docs):
        self._docs = docs
        self.last_k = None

    def similarity_search(self, query, k, filter):  # noqa: A002 - langchain's name
        self.last_k = k
        return self._docs[:k]

    def get(self, where, include):  # noqa: A002 - chroma's name
        return {"documents": [], "metadatas": []}


def test_reranking_widens_the_candidate_pool(monkeypatch):
    """It can only promote what retrieval surfaced.

    Fetching k and then reranking k would reorder exactly the chunks the
    reranker is supposed to be choosing between, which is a no-op dressed up as
    a feature.
    """
    import finrag.rerank as rr

    monkeypatch.setattr(rr, "get_reranker", lambda s=None: None)
    docs = [_doc(str(i), f"chunk {i}") for i in range(60)]
    store = _Store(docs)
    settings = replace(
        Settings(),
        rerank=True,
        rerank_candidates=50,
        retrieval_k=5,
        retrieval_mode="vector",
        query_expansion=False,
    )

    found = search_filing(
        "q", "AAPL", 2023, store=store, settings=settings, apply_context_budget=False
    )

    assert store.last_k == 50, "should fetch the candidate pool, not the final k"
    assert len(found.documents) == 5, "and return only k"


def test_no_reranking_leaves_the_fetch_width_alone():
    """With reranking off, nothing should fetch 50 chunks to return 5."""
    docs = [_doc(str(i), f"chunk {i}") for i in range(60)]
    store = _Store(docs)
    settings = replace(
        Settings(), rerank=False, retrieval_k=5, retrieval_mode="vector", query_expansion=False
    )

    search_filing("q", "AAPL", 2023, store=store, settings=settings, apply_context_budget=False)

    assert store.last_k == 5


def test_rerank_runs_before_the_token_budget(monkeypatch):
    """The budget drops the lowest-ranked chunks, so it must see the final order.

    Trimming first would discard exactly the candidates reranking exists to
    promote.
    """
    import finrag.rerank as rr

    # Long chunks, so the budget can only afford the first couple.
    docs = [_doc(str(i), f"chunk {i} " + "x" * 900) for i in range(6)]
    store = _Store(docs)
    # The best chunk sits last in retrieval order.
    monkeypatch.setattr(
        rr, "get_reranker", lambda s=None: FakeCrossEncoder({docs[5].page_content: 10.0})
    )
    settings = replace(
        Settings(),
        rerank=True,
        rerank_candidates=6,
        retrieval_k=6,
        retrieval_mode="vector",
        query_expansion=False,
        max_context_tokens_raw="500",
    )

    found = search_filing(
        "q", "AAPL", 2023, store=store, settings=settings, apply_context_budget=True
    )

    assert found.documents[0].metadata["id"] == "5", "the reranked winner must survive the budget"
