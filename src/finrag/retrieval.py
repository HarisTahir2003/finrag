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
) -> Retrieved:
    """Search one company's filing for one fiscal year."""
    settings = settings or get_settings()
    if store is None:
        from .ingest.index import open_store

        store = open_store(settings)

    where = {"$and": [{"ticker": ticker.upper()}, {"year": int(fiscal_year)}]}
    try:
        docs = store.similarity_search(expand_query(query), k=settings.retrieval_k, filter=where)
    except Exception as exc:  # noqa: BLE001 - surfaced to the agent as text, not raised
        log.error("retrieval failed for %s FY%s: %s", ticker, fiscal_year, exc)
        docs = []
    docs = trim_to_token_budget(docs, settings.max_context_tokens)
    return Retrieved(documents=docs, ticker=ticker.upper(), fiscal_year=int(fiscal_year))
