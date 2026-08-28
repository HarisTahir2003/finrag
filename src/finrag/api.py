"""HTTP service over the agent.

    uvicorn finrag.api:app          # or: finrag serve

Three things shape this module.

The agent is built **once**, at startup, not per request. Constructing it loads
a sentence-transformer, a cross-encoder and a Chroma collection -- seconds of
work and hundreds of megabytes -- so per-request construction would make the
first token slower than the whole answer.

Retrieval is exposed separately from answering. `/search` runs no LLM at all,
which makes it free, fast, and the honest way to inspect what the agent would
have been given before paying a provider to read it.

Every answer carries its sources. The same reasoning as the UI: a figure is
worth what the passage behind it is worth, and the two surfaces share
`finrag.presentation` so they cannot drift.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .presentation import describe_action, summarise_steps

log = logging.getLogger(__name__)

# Built at startup and shared. Everything in here is read-only per request: the
# executor holds no per-call state, Chroma serialises its own reads, and the
# cross-encoder is only ever asked to score. FastAPI runs sync endpoints in a
# threadpool, so concurrent requests share these rather than rebuilding them.
_state: dict[str, Any] = {"agent": None, "store": None, "settings": None}


def _load() -> None:
    from .ingest.index import open_store

    settings = get_settings()
    store = open_store(settings)
    _state["settings"] = settings
    _state["store"] = store

    try:
        from .agent import build_agent

        _state["agent"] = build_agent(store=store, settings=settings)
    except Exception as exc:  # noqa: BLE001 - /search and /health must still work
        # A missing provider key should cost you /ask, not the whole service.
        log.warning("agent unavailable at startup (%s); /search and /health still serve", exc)
        _state["agent"] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load()
    yield
    _state.clear()


app = FastAPI(
    title="finrag",
    version="0.3.0",
    summary="Retrieval-augmented question answering over SEC 10-K filings.",
    lifespan=lifespan,
)


# ----------------------------------------------------------------- schema


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["total current assets"])
    ticker: str = Field(..., examples=["AAPL"])
    fiscal_year: int = Field(..., examples=[2024])
    k: int | None = Field(None, ge=1, le=100, description="Defaults to the configured retrieval_k")


class Passage(BaseModel):
    text: str
    ticker: str
    fiscal_year: int


class SearchResponse(BaseModel):
    passages: list[Passage]
    count: int
    elapsed_seconds: float


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What was Apple's current ratio in 2024?"])


class Source(BaseModel):
    filing: str
    query: str
    passages: list[str]


class Calculation(BaseModel):
    expression: str
    result: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    calculations: list[Calculation]
    backend: str
    model: str
    elapsed_seconds: float


# --------------------------------------------------------------- helpers


def _require_agent():
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="The agent is unavailable -- usually a missing or rejected provider key. "
            "GET /status reports the configured backend.",
        )
    return agent


def _index_size() -> int:
    from .ingest.index import collection_size

    return collection_size(_state["settings"])


def _resolved_model(settings) -> str:
    from .llm import default_model_for

    return settings.chat_model or default_model_for(settings.llm_backend)


def _run(agent, question: str) -> tuple[str, list[dict]]:
    """Drive one agent run, collecting the answer and every tool call."""
    from .agent import answer_text

    answer, steps = "", []
    for chunk in agent.stream({"input": question}):
        for step in chunk.get("steps", []):
            steps.append(
                {
                    "tool": step.action.tool,
                    "input": step.action.tool_input,
                    "observation": str(step.observation),
                }
            )
        if "output" in chunk:
            answer = answer_text(chunk["output"])
    return answer.strip(), steps


# ------------------------------------------------------------- endpoints


@app.get("/", tags=["ops"])
def index() -> dict:
    """What this is and where to go.

    A bare 404 at the front door is a bad answer to the only URL someone
    naturally tries after starting the server. This is deliberately not a
    redirect to /docs: a browser gets its bearings either way, and a client
    gets a machine-readable list instead of a 307 with no body.
    """
    return {
        "service": "finrag",
        "version": app.version,
        "docs": "/docs",
        "endpoints": {
            "GET /health": "liveness",
            "GET /ready": "readiness — 503 when the index cannot serve",
            "GET /status": "configuration and corpus size",
            "POST /search": "retrieval only, no LLM",
            "POST /ask": "answer with sources and calculations",
            "POST /ask/stream": "the same run, as server-sent events",
        },
    }


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness. Cheap on purpose -- it must not touch the index or a provider."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready() -> dict:
    """Readiness: can this process actually answer?

    Separate from /health because they fail for different reasons and want
    different responses. A process with no index is alive and useless, and a
    load balancer should stop sending it traffic rather than watch it 500 per
    request.
    """
    from .ingest.index import index_status

    ready, size, reason = index_status(_state["settings"])
    if not ready:
        raise HTTPException(status_code=503, detail=f"{reason}; run `finrag index`")
    return {"status": "ready", "chunks": size}


@app.get("/status", tags=["ops"])
def status() -> dict:
    """Configuration and corpus, resolved rather than as written.

    `chat_model` is normally blank and means "this backend's default", so
    reporting the raw setting would answer the question nobody asked.
    """
    settings = _state["settings"]
    from .ingest.download import list_filings

    try:
        chunks = _index_size()
    except Exception:  # noqa: BLE001 - status should describe a broken index, not die with it
        chunks = -1

    return {
        "backend": settings.llm_backend,
        "model": _resolved_model(settings),
        "agent_available": _state.get("agent") is not None,
        "embeddings": settings.embedding_backend,
        "chunk_strategy": settings.chunk_strategy,
        "retrieval_mode": settings.retrieval_mode,
        "rerank": settings.rerank,
        "retrieval_k": settings.retrieval_k,
        "chunks_indexed": chunks,
        "filings_on_disk": len(list_filings(settings=settings)),
    }


@app.post("/search", response_model=SearchResponse, tags=["retrieval"])
def search(request: SearchRequest) -> SearchResponse:
    """Retrieval only. No LLM, no provider key, no cost."""
    from dataclasses import replace

    from .retrieval import search_filing

    settings = _state["settings"]
    if request.k:
        settings = replace(settings, retrieval_k=request.k)

    started = time.perf_counter()
    found = search_filing(
        request.query,
        request.ticker,
        request.fiscal_year,
        store=_state["store"],
        settings=settings,
        # The budget is a property of the chat model, and this endpoint calls
        # none -- trimming here would make retrieval results depend on which
        # provider happened to be configured.
        apply_context_budget=False,
    )
    return SearchResponse(
        passages=[
            Passage(
                text=d.page_content,
                ticker=str(d.metadata.get("ticker", "")),
                fiscal_year=int(d.metadata.get("year", 0)),
            )
            for d in found.documents
        ],
        count=len(found.documents),
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )


@app.post("/ask", response_model=AskResponse, tags=["agent"])
def ask(request: AskRequest) -> AskResponse:
    agent = _require_agent()
    settings = _state["settings"]
    started = time.perf_counter()
    try:
        answer, steps = _run(agent, request.question)
    except Exception as exc:  # noqa: BLE001 - a provider failure is not our bug
        raise HTTPException(status_code=502, detail=f"agent run failed: {exc}") from exc

    grouped = summarise_steps(steps)
    return AskResponse(
        answer=answer or "The agent returned nothing.",
        sources=[Source(**s) for s in grouped["sources"]],
        calculations=[Calculation(**c) for c in grouped["calculations"]],
        backend=settings.llm_backend,
        model=_resolved_model(settings),
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )


@app.post("/ask/stream", tags=["agent"])
def ask_stream(request: AskRequest) -> StreamingResponse:
    """The same run as /ask, narrated as it happens.

    An answer takes tens of seconds across several tool calls. Sent as
    server-sent events: `step` per tool call, then one `answer` carrying the
    same payload /ask returns, so a client need not accumulate anything.

    The generator is synchronous deliberately -- Starlette iterates a sync
    generator in a threadpool, whereas an async one calling the blocking agent
    would stall the event loop for every other request.
    """
    agent = _require_agent()
    settings = _state["settings"]

    def events():
        from .agent import answer_text

        started = time.perf_counter()
        answer, steps = "", []
        try:
            for chunk in agent.stream({"input": request.question}):
                for action in chunk.get("actions", []):
                    payload = {
                        "text": describe_action(action.tool, action.tool_input),
                        "tool": action.tool,
                    }
                    yield f"event: step\ndata: {json.dumps(payload)}\n\n"
                for step in chunk.get("steps", []):
                    steps.append(
                        {
                            "tool": step.action.tool,
                            "input": step.action.tool_input,
                            "observation": str(step.observation),
                        }
                    )
                if "output" in chunk:
                    answer = answer_text(chunk["output"])
        except Exception as exc:  # noqa: BLE001 - the stream is already open, so report in-band
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            return

        grouped = summarise_steps(steps)
        final = {
            "answer": answer.strip() or "The agent returned nothing.",
            **grouped,
            "backend": settings.llm_backend,
            "model": _resolved_model(settings),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        yield f"event: answer\ndata: {json.dumps(final)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Proxies that buffer will happily hold a 45-second stream to the end.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
