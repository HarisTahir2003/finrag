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
    # Which provider answers questions. Anthropic and Google are interchangeable
    # here; the embedding backend above is chosen separately, because Anthropic
    # publishes no embedding model.
    llm_backend: str = field(default_factory=lambda: _env("FINRAG_LLM_BACKEND", "anthropic"))
    # Empty means "use the default for the selected backend" -- see finrag.llm.
    chat_model: str = field(default_factory=lambda: _env("FINRAG_CHAT_MODEL", ""))
    max_output_tokens: int = field(
        default_factory=lambda: int(_env("FINRAG_MAX_OUTPUT_TOKENS", "4000"))
    )

    # ---- Free-tier survival -------------------------------------------------
    # Client-side requests-per-minute. Unset -> the per-backend default in
    # finrag.llm.DEFAULT_RPM (sized just under each free tier's cap); 0 -> off.
    requests_per_minute: float | None = field(
        default_factory=lambda: (
            float(v) if (v := os.environ.get("FINRAG_RPM")) not in (None, "") else None
        )
    )
    # Comma-separated backends to fail over to when the primary errors out --
    # on a free tier that usually means its daily quota, so the usable budget
    # becomes the union of the tiers. Applied on plain-chat paths.
    llm_fallbacks: str = field(default_factory=lambda: _env("FINRAG_LLM_FALLBACKS", ""))
    # Cache identical LLM calls in SQLite so re-running an unchanged evaluation
    # costs zero tokens. On by default for eval commands; "0" disables.
    llm_cache: bool = field(default_factory=lambda: _env("FINRAG_LLM_CACHE", "1") != "0")
    # Trim retrieved context to a token budget before it reaches the model.
    # "auto" resolves per backend: free tiers with hard request-size ceilings
    # (Cerebras 8K, GitHub Models 8K-in) get 6000, everything else unlimited.
    # "hybrid" fuses BM25 with the vector search; "vector" is the old
    # behaviour. Hybrid is the default because it earned it on the 30-probe
    # corpus set -- hit_rate 0.967 -> 1.000, mrr 0.695 -> 0.720 -- which is the
    # same bar query expansion was held to and failed.
    retrieval_mode: str = field(
        default_factory=lambda: _env("FINRAG_RETRIEVAL_MODE", "hybrid").strip().lower()
    )

    # Reranking reorders a wider candidate pool with a model that scores
    # query-document pairs jointly, rather than comparing two independently
    # computed vectors. On by default: it moved mrr from 0.720 to 0.824 on the
    # 30-probe corpus set, the same bar hybrid retrieval had to clear.
    #
    # It costs about 730ms per query -- 106ms to 836ms, scoring fifty pairs
    # through a cross-encoder. That is roughly 1% of an agent answer, which
    # takes tens of seconds across several LLM calls, and would be a much larger
    # share of a bare search endpoint. Set FINRAG_RERANK=0 there.
    rerank: bool = field(
        default_factory=lambda: (
            _env("FINRAG_RERANK", "1").strip().lower() not in ("0", "false", "no", "")
        )
    )
    # ms-marco-MiniLM-L-6-v2 is ~80MB. bge-reranker-base scores better and is
    # ~1.1GB, which on an 8GB machine competes with the embedding model already
    # resident -- a real constraint here, not a hypothetical one.
    rerank_model: str = field(
        default_factory=lambda: _env("FINRAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    # How many candidates to fetch before reordering. Reranking can only
    # promote what retrieval surfaced, so this is the real recall ceiling;
    # retrieval_k is what survives to the model.
    rerank_candidates: int = field(
        default_factory=lambda: int(_env("FINRAG_RERANK_CANDIDATES", "50"))
    )

    # Off by default because it was measured, not assumed. See
    # retrieval.expand_query: on the real corpus it halved hit rate.
    query_expansion: bool = field(
        default_factory=lambda: _env("FINRAG_QUERY_EXPANSION", "").lower() in ("1", "true", "yes")
    )

    # --- spending limits for a shared key ---------------------------------
    #
    # A public demo answers with the owner's key. Groq's free tier allows 1,000
    # requests a day across everyone who finds the URL, and one agent answer
    # costs several of them -- a search, a read, sometimes a calculation. A
    # visitor holding down a starter pill can end the day for everybody without
    # meaning any harm.
    #
    # Generous enough that ordinary local use never notices. Set either to 0 to
    # disable that scope rather than to block everything.
    questions_per_session: int = field(
        default_factory=lambda: int(_env("FINRAG_QUESTIONS_PER_SESSION", "20"))
    )
    session_window_seconds: int = field(
        default_factory=lambda: int(_env("FINRAG_SESSION_WINDOW_SECONDS", "3600"))
    )
    questions_per_day: int = field(
        default_factory=lambda: int(_env("FINRAG_QUESTIONS_PER_DAY", "300"))
    )

    # How long one question may run before the reader is told it gave up. The
    # agent takes 15-45s normally; well past that is a stalled provider, and
    # waiting silently is the worst of both outcomes.
    answer_timeout_seconds: int = field(
        default_factory=lambda: int(_env("FINRAG_ANSWER_TIMEOUT_SECONDS", "120"))
    )

    # Client-side retries per provider call. Six is deliberate and stays the
    # default: llm.MAX_RETRIES is applied uniformly so the backend comparison
    # ranks models rather than their client libraries' retry policies.
    #
    # It is the wrong number for a public demo on a metered free tier, though.
    # Each retry is another request against the daily quota, so one rate-limited
    # question can cost seven. Lower it there, not here.
    max_retries: int = field(default_factory=lambda: int(_env("FINRAG_MAX_RETRIES", "6")))

    max_context_tokens_raw: str = field(
        default_factory=lambda: _env("FINRAG_MAX_CONTEXT_TOKENS", "auto")
    )

    # The model that SCORES answers in the RAGAS suite, as opposed to producing
    # them. Blank means "same as llm_backend", which is fine for a single run
    # but invalid for a cross-backend comparison: each backend would mark its
    # own homework, and any ranking would partly measure judge self-preference
    # rather than answer quality. Pin one judge across every run being compared.
    judge_backend: str = field(default_factory=lambda: _env("FINRAG_JUDGE_BACKEND", ""))
    judge_model: str = field(default_factory=lambda: _env("FINRAG_JUDGE_MODEL", ""))

    # Overrides the preset base URL for OpenAI-compatible backends. Point this
    # at a self-hosted vLLM or LM Studio server to use one.
    openai_base_url: str = field(default_factory=lambda: _env("FINRAG_OPENAI_BASE_URL", ""))

    # Google Cloud Vertex AI. Authenticates with Application Default
    # Credentials rather than an API key, and bills the Cloud billing account
    # -- so promotional credits apply, which the AI Studio API cannot use on a
    # prepay account.
    gcp_project: str = field(default_factory=lambda: _env("GOOGLE_CLOUD_PROJECT", ""))
    gcp_location: str = field(default_factory=lambda: _env("GOOGLE_CLOUD_LOCATION", "us-central1"))

    # Local models via Ollama.
    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    # Must exceed the retrieved context plus the agent scratchpad. Ollama
    # truncates the prompt without saying so, which presents as the model
    # ignoring its context rather than never having received it.
    ollama_context_length: int = field(
        default_factory=lambda: int(_env("FINRAG_OLLAMA_NUM_CTX", "16384"))
    )

    # "semantic" chunks on document structure via unstructured; "recursive" is
    # the faster fixed-width splitter. See finrag.chunking for the trade-off.
    chunk_strategy: str = field(default_factory=lambda: _env("FINRAG_CHUNK_STRATEGY", "semantic"))
    chunk_size: int = field(default_factory=lambda: int(_env("FINRAG_CHUNK_SIZE", "3000")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("FINRAG_CHUNK_OVERLAP", "300")))

    collection_name: str = field(default_factory=lambda: _env("FINRAG_COLLECTION", "sec_10k"))
    # 20, and the reasoning is worth keeping because the obvious answer is
    # wrong. Retrieval quality saturates at k=15 -- hit_rate 1.000 and mrr
    # 0.8243, flat through k=30 -- so k=15 looks like free money against 9,725
    # context tokens at k=20. Measured end to end, k=15 scored RAGAS
    # faithfulness 0.650 against k=20's 0.975 on the same ten cases.
    #
    # The retrieval metrics cannot see the difference: chunks 16-20 do not
    # change *whether* the answering passage was retrieved, which is all
    # hit_rate and mrr measure, but they do change whether the model's claims
    # are supported. Selecting k on the retrieval curve alone would have
    # shipped a regression that every retrieval metric called free.
    # See docs/k-sweep-light.png.
    retrieval_k: int = field(default_factory=lambda: int(_env("FINRAG_RETRIEVAL_K", "20")))

    sec_contact_email: str = field(default_factory=lambda: _env("SEC_CONTACT_EMAIL", ""))
    sec_company_name: str = field(default_factory=lambda: _env("SEC_COMPANY_NAME", "finrag"))

    @property
    def max_context_tokens(self) -> int:
        """Resolved context budget in tokens; 0 means unlimited.

        The 8,192-token ceilings on the Cerebras free tier and GitHub Models
        are request-size limits, not truncation -- exceed them and the call
        fails with a 400. 6000 leaves room for the system prompt, question and
        agent scratchpad inside an 8K window.
        """
        raw = self.max_context_tokens_raw.strip().lower()
        if raw != "auto":
            return int(raw)

        backend = self.llm_backend.lower()
        if backend in ("cerebras", "github"):
            return 6000
        if backend == "groq":
            # Groq meters tokens per minute, not per request: the free tier
            # allows 8,000 TPM on every tool-calling model it offers. That is a
            # tighter constraint than it looks, because an agent resends its
            # whole scratchpad on each step -- a question answered in three
            # steps pays for its retrieved context roughly three times over. A
            # single measured request came to 8,806 tokens of which only 2,204
            # were retrieved context; the rest was accumulated history. 2000
            # keeps each tool result small enough that three of them plus the
            # system prompt still fit inside one minute's budget.
            return 2000
        if backend == "ollama":
            # Ollama is the one backend whose ceiling truncates silently rather
            # than erroring, so an overflow presents as the model ignoring its
            # context. Two fifths of the window leaves room for the system
            # prompt, tool schemas, the agent scratchpad and the reply.
            return (self.ollama_context_length * 2) // 5
        return 0

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
