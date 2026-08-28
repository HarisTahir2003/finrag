"""The HTTP service.

Nothing here calls a provider or opens the real index. The agent and the store
are stubbed at their source modules, because api._load() builds both at startup
and the point of these tests is the endpoint contract, not the agent's answers.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from finrag import api  # noqa: E402

CONTEXT = (
    "--- AAPL FY2024 ---\n\n"
    "[chunk 1]\nTotal current assets 152,987\n\n"
    "[chunk 2]\nTotal current liabilities 176,392\n"
)


class _Action:
    def __init__(self, tool, tool_input):
        self.tool, self.tool_input = tool, tool_input


class _Step:
    def __init__(self, action, observation):
        self.action, self.observation = action, observation


class _StubAgent:
    def __init__(self, fail=False):
        self._fail = fail

    def stream(self, _inputs):
        if self._fail:
            raise RuntimeError("provider rejected the key")
        search = _Action(
            "search_10k_reports",
            {"ticker": "AAPL", "fiscal_year": 2024, "query": "total current assets"},
        )
        calc = _Action("calculator", {"expression": "152987/176392"})
        yield {"actions": [search]}
        yield {"steps": [_Step(search, CONTEXT)]}
        yield {"actions": [calc]}
        yield {"steps": [_Step(calc, "0.8673")]}
        yield {"output": "Apple's current ratio was 0.87."}


def _client(monkeypatch, *, agent=None, chunks=12376, filings=50):
    """A client whose startup installs stubs instead of loading the real thing.

    index_status is stubbed at its own module because /ready resolves it lazily
    -- the same helper the Streamlit sidebar uses, so both surfaces agree about
    what "no index" means.
    """
    import finrag.ingest.index
    from finrag.config import get_settings

    settings = get_settings()

    def fake_load():
        api._state["settings"] = settings
        api._state["store"] = object()
        api._state["agent"] = agent

    status = (True, chunks, "ready") if chunks else (False, 0, "index is empty")
    monkeypatch.setattr(api, "_load", fake_load)
    monkeypatch.setattr(api, "_index_size", lambda: chunks)
    monkeypatch.setattr(finrag.ingest.index, "index_status", lambda *a, **k: status)
    monkeypatch.setattr("finrag.ingest.download.list_filings", lambda **kw: [None] * filings)
    return TestClient(api.app)


def test_root_points_at_the_docs(monkeypatch):
    """The only URL anyone tries after starting the server used to 404."""
    with _client(monkeypatch, agent=None, chunks=0) as client:
        response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["docs"] == "/docs"
    assert "POST /ask" in body["endpoints"]


def test_health_is_cheap_and_always_ok(monkeypatch):
    """Liveness must not touch the index or a provider."""
    with _client(monkeypatch, agent=None, chunks=0) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_is_503_when_the_index_is_empty(monkeypatch):
    """Alive and useless is a real state, and it deserves its own answer.

    A process with no index should stop receiving traffic rather than 500 once
    per request.
    """
    with _client(monkeypatch, agent=_StubAgent(), chunks=0) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert "empty" in response.json()["detail"]


def test_ready_reports_the_chunk_count(monkeypatch):
    with _client(monkeypatch, agent=_StubAgent()) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["chunks"] == 12376


def test_status_reports_the_resolved_model(monkeypatch):
    """chat_model is normally blank and means "this backend's default"."""
    with _client(monkeypatch, agent=_StubAgent()) as client:
        body = client.get("/status").json()

    assert body["model"], "a blank setting must not surface as a blank model"
    assert body["agent_available"] is True
    assert body["chunks_indexed"] == 12376
    assert body["filings_on_disk"] == 50
    assert set(("backend", "retrieval_mode", "rerank", "retrieval_k")) <= set(body)


def test_search_runs_no_llm(monkeypatch):
    """The endpoint exists to be free: it must work with no agent at all."""
    from langchain_core.documents import Document

    from finrag.retrieval import Retrieved

    captured = {}

    def fake_search(
        query, ticker, fiscal_year, store=None, settings=None, apply_context_budget=True
    ):
        captured["k"] = settings.retrieval_k
        captured["budget"] = apply_context_budget
        return Retrieved(
            documents=[
                Document(
                    page_content="Total current assets 152,987",
                    metadata={"ticker": "AAPL", "year": 2024},
                )
            ],
            ticker=ticker,
            fiscal_year=fiscal_year,
        )

    monkeypatch.setattr("finrag.retrieval.search_filing", fake_search)
    with _client(monkeypatch, agent=None) as client:
        response = client.post(
            "/search",
            json={"query": "total current assets", "ticker": "AAPL", "fiscal_year": 2024, "k": 3},
        )

    assert response.status_code == 200, "no agent must not break retrieval"
    body = response.json()
    assert body["count"] == 1
    assert body["passages"][0]["ticker"] == "AAPL"
    assert captured["k"] == 3, "k in the request should override the configured default"
    assert captured["budget"] is False, "a context budget belongs to a chat model, not to search"


def test_search_rejects_a_malformed_body(monkeypatch):
    with _client(monkeypatch, agent=None) as client:
        assert client.post("/search", json={"query": "x"}).status_code == 422
        assert (
            client.post(
                "/search", json={"query": "", "ticker": "AAPL", "fiscal_year": 2024}
            ).status_code
            == 422
        )


def test_ask_returns_the_answer_with_its_provenance(monkeypatch):
    with _client(monkeypatch, agent=_StubAgent()) as client:
        body = client.post("/ask", json={"question": "current ratio?"}).json()

    assert "0.87" in body["answer"]
    assert body["sources"][0]["filing"] == "AAPL FY2024"
    assert len(body["sources"][0]["passages"]) == 2
    assert body["calculations"] == [{"expression": "152987/176392", "result": "0.8673"}]
    assert body["elapsed_seconds"] >= 0


def test_ask_is_503_without_an_agent(monkeypatch):
    """A missing provider key should cost /ask, not the whole service."""
    with _client(monkeypatch, agent=None) as client:
        response = client.post("/ask", json={"question": "anything"})

    assert response.status_code == 503
    assert "/status" in response.json()["detail"]


def test_ask_is_502_when_the_provider_fails(monkeypatch):
    """Their outage is not our bug, and 500 would say it was."""
    with _client(monkeypatch, agent=_StubAgent(fail=True)) as client:
        response = client.post("/ask", json={"question": "anything"})

    assert response.status_code == 502
    assert "provider rejected the key" in response.json()["detail"]


def test_ask_stream_emits_steps_then_one_final_answer(monkeypatch):
    with (
        _client(monkeypatch, agent=_StubAgent()) as client,
        client.stream("POST", "/ask/stream", json={"question": "current ratio?"}) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = "".join(response.iter_text())

    events = [line for line in raw.splitlines() if line.startswith("event:")]
    assert events.count("event: step") == 2, "one per tool call, as it starts"
    assert events.count("event: answer") == 1, "exactly one terminal event"

    final = json.loads(raw.split("event: answer\ndata: ")[1].split("\n\n")[0])
    assert "0.87" in final["answer"]
    assert final["sources"][0]["filing"] == "AAPL FY2024"
    assert final["calculations"][0]["expression"] == "152987/176392"


def test_ask_stream_reports_a_failure_in_band(monkeypatch):
    """The response has already begun, so an error cannot be a status code."""
    with (
        _client(monkeypatch, agent=_StubAgent(fail=True)) as client,
        client.stream("POST", "/ask/stream", json={"question": "x"}) as response,
    ):
        assert response.status_code == 200
        raw = "".join(response.iter_text())

    assert "event: error" in raw
    assert "provider rejected the key" in raw
