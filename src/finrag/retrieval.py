"""Retrieval over the indexed filings."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document

from .config import Settings, get_settings

log = logging.getLogger(__name__)

# Appended to the user's question before embedding. A question like "what were
# revenues" is lexically nothing like the balance-sheet text that answers it, so
# nudging the query vector toward financial-statement language measurably helps.
_QUERY_EXPANSION = "financial statements balance sheet results of operations management discussion"
_NARRATIVE_HINTS = ("why", "explain", "change", "reason", "risk", "strategy", "outlook")
_NARRATIVE_EXPANSION = (
    "Management's Discussion and Analysis Liquidity and Capital Resources Risk Factors"
)


@dataclass(frozen=True)
class Retrieved:
    documents: list[Document]
    ticker: str
    fiscal_year: int

    def as_context(self) -> str:
        """Render for an LLM prompt, one labelled block per chunk."""
        if not self.documents:
            return f"No indexed filing found for {self.ticker} FY{self.fiscal_year}."
        parts = [f"--- {self.ticker} FY{self.fiscal_year} ---"]
        parts.extend(f"\n[chunk {i + 1}]\n{d.page_content}" for i, d in enumerate(self.documents))
        return "\n".join(parts)


def expand_query(query: str) -> str:
    """Append financial-statement vocabulary to a query. Off by default.

    Inherited from the notebook, where the claim that it helps was asserted
    rather than measured. Measured here on the indexed corpus, over fifteen
    cases whose target string was confirmed present in the filing first:

                            hit_rate    mrr
        with expansion         0.533   0.283
        without                1.000   0.822

    It never once improved a ranking and lost the answer completely in seven
    cases. The reason is visible in the text it produces -- "total net sales
    financial statements balance sheet results of operations management
    discussion" -- where nine words of generic boilerplate swamp three words of
    actual question, and the nearest neighbours become narrative prose that
    talks *about* the financial statements rather than the statement itself.

    Kept behind FINRAG_QUERY_EXPANSION so the experiment can be re-run rather
    than taken on faith, which is how it got in.
    """
    expanded = f"{query} {_QUERY_EXPANSION}"
    if any(hint in query.lower() for hint in _NARRATIVE_HINTS):
        expanded = f"{expanded} {_NARRATIVE_EXPANSION}"
    return expanded


def reciprocal_rank_fusion(result_sets: list[list[Document]], k: int = 60) -> list[Document]:
    """Merge several ranked lists into one.

    A document ranked highly by more than one query outranks a document ranked
    first by only one, which is what makes multi-query retrieval worth doing.
    Carried over from the FinanceBench notebook, where it was defined but never
    used by the evaluation.
    """
    scores: dict[str, float] = {}
    seen: dict[str, Document] = {}
    for docs in result_sets:
        for rank, doc in enumerate(docs):
            key = doc.metadata.get("id") or doc.page_content[:200]
            seen.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + k)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [seen[key] for key, _ in ranked]


# One BM25 index per filing, keyed by (ticker, year, chunk count). Rebuilding it
# on every query would mean re-tokenising a few hundred chunks each time; caching
# it forever would serve a stale index after a re-index, which is why the chunk
# count is part of the key -- a cheap way to notice the corpus moved underneath.
_BM25_CACHE: dict[tuple[str, int, int], object] = {}


def _bm25_for_filing(ticker: str, fiscal_year: int, store):
    """A lexical index over one filing's chunks, or None if unavailable.

    The candidate pool is one company-year, not the whole corpus: every query
    here is already filtered to a single filing, so this indexes 99 to 978
    chunks rather than 12,376. That is what makes building it on demand
    affordable.

    Returns None when rank_bm25 is not installed, so hybrid mode degrades to
    vector-only rather than failing -- the same shape as chunk_semantic's
    fallback.
    """
    where = {"$and": [{"ticker": ticker.upper()}, {"year": int(fiscal_year)}]}
    try:
        got = store.get(where=where, include=["documents", "metadatas"])
    except (AttributeError, TypeError):
        # Not every store can enumerate its contents -- a retriever-shaped
        # object may only know how to search. Lexical scoring needs the corpus
        # in hand, so without it hybrid mode is simply vector mode.
        log.debug("store does not support get(); hybrid retrieval falls back to vector search")
        return None
    except Exception as exc:  # noqa: BLE001 - a failed fetch must not fail the query
        log.warning("could not read chunks for %s FY%s: %s", ticker, fiscal_year, exc)
        return None

    texts = got.get("documents") or []
    if not texts:
        return None

    key = (ticker.upper(), int(fiscal_year), len(texts))
    if key in _BM25_CACHE:
        return _BM25_CACHE[key]

    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError:
        log.warning("langchain-community missing; hybrid retrieval falls back to vector search")
        return None

    metadatas = got.get("metadatas") or [{} for _ in texts]
    try:
        # Passing metadatas is not cosmetic: reciprocal_rank_fusion dedups on
        # metadata["id"], so without it the lexical and vector result sets share
        # no keys and the "fusion" silently concatenates two disjoint lists.
        retriever = BM25Retriever.from_texts(texts, metadatas=metadatas)
    except ImportError:
        log.warning(
            "hybrid retrieval needs rank_bm25 (pip install 'finrag[hybrid]'); "
            "falling back to vector search"
        )
        return None
    except Exception as exc:  # noqa: BLE001 - a broken index must not fail the query
        log.warning("could not build a lexical index for %s FY%s: %s", ticker, fiscal_year, exc)
        return None

    _BM25_CACHE[key] = retriever
    return retriever


def _lexical_candidates(query: str, ticker: str, fiscal_year: int, store, k: int):
    """Top-k chunks by BM25. Empty when lexical retrieval is unavailable."""
    retriever = _bm25_for_filing(ticker, fiscal_year, store)
    if retriever is None:
        return []
    try:
        retriever.k = k
        return retriever.invoke(query)
    except Exception as exc:  # noqa: BLE001 - degrade to vector-only
        log.warning("lexical retrieval failed for %s FY%s: %s", ticker, fiscal_year, exc)
        return []


def trim_to_token_budget(docs: list[Document], max_tokens: int) -> list[Document]:
    """Keep the top-ranked whole chunks that fit inside a token budget.

    Some free tiers enforce a hard request-size ceiling (Cerebras 8K, GitHub
    Models 8K-in) where an oversized prompt is a 400 error, not a truncation.
    Dropping the lowest-ranked chunks degrades recall gracefully; a failed call
    retrieves nothing at all. Chunks are never split -- half a balance-sheet
    table is worse than none. chars/4 is the usual rough token estimate, close
    enough for a budget that already carries headroom.
    """
    if max_tokens <= 0:
        return docs
    budget_chars = max_tokens * 4
    kept: list[Document] = []
    used = 0
    for doc in docs:
        cost = len(doc.page_content)
        if kept and used + cost > budget_chars:
            break
        kept.append(doc)
        used += cost
    if len(kept) < len(docs):
        log.info(
            "context trimmed to %d of %d chunks (budget %d tokens)",
            len(kept),
            len(docs),
            max_tokens,
        )
    return kept


def search_filing(
    query: str,
    ticker: str,
    fiscal_year: int,
    store=None,
    settings: Settings | None = None,
    apply_context_budget: bool = True,
) -> Retrieved:
    """Search one company's filing for one fiscal year.

    ``apply_context_budget`` trims the result to what the configured chat model
    can accept. That is right when feeding an LLM and wrong when measuring the
    retriever: the budget is derived from ``llm_backend``, so leaving it on
    would make the LLM-free retrieval metrics -- and therefore the CI quality
    gate -- move when the chat provider changed. The retrieval suite passes
    False so it measures ranking rather than configuration.
    """
    settings = settings or get_settings()
    if store is None:
        from .ingest.index import open_store

        store = open_store(settings)

    where = {"$and": [{"ticker": ticker.upper()}, {"year": int(fiscal_year)}]}
    k = settings.retrieval_k
    # Reranking can only promote what retrieval surfaced, so it needs a wider
    # pool than the caller ultimately wants. Without this the reranker would be
    # reordering the same k chunks it is meant to be choosing between.
    fetch = max(k, settings.rerank_candidates) if settings.rerank else k
    try:
        search_text = expand_query(query) if settings.query_expansion else query
        docs = store.similarity_search(search_text, k=fetch, filter=where)
    except Exception as exc:  # noqa: BLE001 - surfaced to the agent as text, not raised
        log.error("retrieval failed for %s FY%s: %s", ticker, fiscal_year, exc)
        docs = []

    if settings.retrieval_mode == "hybrid":
        # The lexical side always sees the raw question. Expansion is a
        # vector-space trick -- padding the query with financial boilerplate --
        # and BM25 would read that padding as nine more terms to match on.
        lexical = _lexical_candidates(query, ticker, fiscal_year, store, fetch)
        if lexical:
            docs = reciprocal_rank_fusion([docs, lexical])[:fetch]

    if settings.rerank:
        from .rerank import rerank

        docs = rerank(query, docs, top_k=k, settings=settings)
    else:
        docs = docs[:k]
    if apply_context_budget:
        docs = trim_to_token_budget(docs, settings.max_context_tokens)
    return Retrieved(documents=docs, ticker=ticker.upper(), fiscal_year=int(fiscal_year))
