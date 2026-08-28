"""The Streamlit front end and the display helpers behind it.

Worth testing because the app had never once been run: it is the only surface
in the repo CI does not exercise, and a front end that imports cleanly can
still render nothing useful.

The agent is stubbed. These assert the UI's own behaviour -- what it does with
a stream of tool calls -- not the agent's, which the eval suites cover and which
would cost an API call per test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finrag.presentation import calculator_expression, describe_action, parse_passages

APP = str(Path(__file__).resolve().parents[1] / "app.py")

CONTEXT = (
    "--- AAPL FY2024 ---\n\n"
    "[chunk 1]\nTotal current assets 152,987 143,566\n\n"
    "[chunk 2]\nTotal current liabilities 176,392 145,308\n"
)


# ------------------------------------------------------------- helpers


def test_parse_passages_recovers_the_filing_and_its_chunks():
    label, passages = parse_passages(CONTEXT)

    assert label == "AAPL FY2024"
    assert len(passages) == 2
    assert "152,987" in passages[0]


def test_parse_passages_survives_an_unrecognised_observation():
    """An unparsed observation is still worth showing."""
    assert parse_passages("no filing header here") == ("filing", ["no filing header here"])


def test_parse_passages_handles_an_empty_observation():
    assert parse_passages("") == ("filing", [])


def test_calculator_expression_strips_the_envelope():
    """Rendering the dict puts `{'expression': ...}` in front of the reader."""
    assert calculator_expression({"expression": "152987/176392"}) == "152987/176392"
    assert calculator_expression("1+1") == "1+1"


def test_describe_action_names_the_call_in_words():
    search = describe_action(
        "search_10k_reports", {"ticker": "AAPL", "fiscal_year": 2024, "query": "net sales"}
    )
    assert "AAPL FY2024" in search and "net sales" in search
    assert "152987/176392" in describe_action("calculator", {"expression": "152987/176392"})
    assert "mystery" in describe_action("mystery", {})


# ----------------------------------------------------------------- app

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


class _Action:
    def __init__(self, tool, tool_input):
        self.tool = tool
        self.tool_input = tool_input


class _Step:
    def __init__(self, action, observation):
        self.action = action
        self.observation = observation


class _StubAgent:
    """Yields the shape AgentExecutor.stream() produces."""

    def __init__(self, output="Apple's current ratio was 0.87.", fail=False):
        self._output = output
        self._fail = fail

    def stream(self, _inputs):
        if self._fail:
            raise RuntimeError("provider unavailable")
        search = _Action(
            "search_10k_reports",
            {"ticker": "AAPL", "fiscal_year": 2024, "query": "total current assets"},
        )
        calc = _Action("calculator", {"expression": "152987/176392"})
        yield {"actions": [search]}
        yield {"steps": [_Step(search, CONTEXT)]}
        yield {"actions": [calc]}
        yield {"steps": [_Step(calc, "0.8673125765340832")]}
        yield {"output": self._output}


def _stubbed_app(monkeypatch, agent, *, index=(True, 12376, "ready")):
    """AppTest runs app.py in its own namespace, so stubs go in at the source.

    app.py resolves build_agent and index_status lazily, so patching the
    modules they live in reaches it. st.cache_resource would otherwise hold the
    first stub for the whole process.

    The index is stubbed because a UI test should not need a 134MB corpus --
    CI has no index, and without this the chat input is disabled and nothing
    renders at all, which is how these first failed.
    """
    import streamlit as st

    import finrag.agent
    import finrag.ingest.index

    # A keyless backend, so the test does not depend on a provider key existing.
    # Without this the default is anthropic, the app correctly refuses to run
    # without ANTHROPIC_API_KEY, and every assertion sees an empty page -- which
    # passed locally off a .env and failed in CI, where there is none.
    # setenv wins over .env either way: load_dotenv never overrides.
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "ollama")
    monkeypatch.setattr(finrag.agent, "build_agent", lambda **kwargs: agent)
    monkeypatch.setattr(finrag.ingest.index, "index_status", lambda *a, **k: index)
    # load_agent opens the store before it builds the agent, and opening the
    # store loads a sentence-transformer. Stubbing only build_agent left the
    # embedding model as a hard requirement of a UI test -- fine locally, and
    # an ImportError in CI, which installs no [local] extra.
    monkeypatch.setattr(finrag.ingest.index, "open_store", lambda *a, **k: object())
    st.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=60)


def test_app_loads_without_error(monkeypatch):
    at = _stubbed_app(monkeypatch, None)
    at.run()

    assert not at.exception
    assert "SEC filing analyst" in [t.value for t in at.title]


def test_no_index_disables_the_chat_and_says_why(monkeypatch):
    """Alive with no corpus: refuse at the door rather than deep in retrieval."""
    at = _stubbed_app(monkeypatch, None, index=(False, 0, "no index found"))
    at.run()

    assert at.chat_input[0].disabled
    assert any("No index found" in str(e.value) for e in at.error)


def test_answer_and_sources_are_rendered(monkeypatch):
    """The whole point of the UI: the answer, and what backed it."""
    at = _stubbed_app(monkeypatch, _StubAgent())
    at.run()
    at.chat_input[0].set_value("What was Apple's current ratio in fiscal 2024?").run()

    assert not at.exception
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "0.87" in rendered
    assert "Searching AAPL FY2024" in rendered, "progress is the only feedback for ~45s"
    assert "152987/176392" in rendered
    assert "{'expression'" not in rendered
    assert any("passage" in e.label for e in at.expander), "provenance must be shown"
    assert len(at.text_area) == 2


def test_a_failing_provider_is_reported_not_raised(monkeypatch):
    """A dead backend should read as a message, not a stack trace."""
    at = _stubbed_app(monkeypatch, _StubAgent(fail=True))
    at.run()
    at.chat_input[0].set_value("anything").run()

    assert not at.exception
    assert "provider unavailable" in " ".join(str(m.value) for m in at.markdown)


def test_an_empty_answer_does_not_render_a_blank_bubble(monkeypatch):
    at = _stubbed_app(monkeypatch, _StubAgent(output="   "))
    at.run()
    at.chat_input[0].set_value("anything").run()

    assert "returned nothing" in " ".join(str(m.value) for m in at.markdown)


# ------------------------------------------------------ rendering and memory


def test_escape_dollars_stops_currency_becoming_latex():
    """Two dollar figures in a sentence made everything between them maths."""
    from finrag.presentation import escape_dollars

    out = escape_dollars("net sales of $574,785 million, compared to $383,285 million")

    assert out.count("\\$") == 2
    assert "$574,785" not in out.replace("\\$", "$@")  # no unescaped delimiter left


def test_escape_dollars_leaves_code_spans_alone():
    """A backslash inside code renders as a backslash."""
    from finrag.presentation import escape_dollars

    assert escape_dollars("use `$5` here and $10 there") == "use `$5` here and \\$10 there"


def test_escape_dollars_is_idempotent():
    from finrag.presentation import escape_dollars

    once = escape_dollars("costs $5")
    assert escape_dollars(once) == once


def test_follow_up_questions_receive_the_prior_turns(monkeypatch):
    """The agent's prompt always had a chat_history slot; the UI never filled it.

    Without this, "explain why apple's was higher" straight after a comparison
    got "please specify which year" -- the front end forgetting, read by the
    user as the agent being dim.
    """
    seen = {}

    class _HistoryAgent(_StubAgent):
        def stream(self, inputs):
            seen["history"] = inputs.get("chat_history")
            yield from super().stream(inputs)

    at = _stubbed_app(monkeypatch, _HistoryAgent())
    at.run()
    at.chat_input[0].set_value("first question").run()
    assert seen["history"] == [], "the opening turn has no history"

    at.chat_input[0].set_value("and why is that?").run()
    contents = [m.content for m in seen["history"]]
    assert "first question" in contents, "the follow-up must see what came before"
    assert len(seen["history"]) == 2, "one user turn and one answer"


def test_history_carries_the_unescaped_text(monkeypatch):
    """Escaping is for the renderer. The agent should not read backslashes."""
    seen = {}

    class _HistoryAgent(_StubAgent):
        def stream(self, inputs):
            seen["history"] = inputs.get("chat_history")
            yield from super().stream(inputs)

    at = _stubbed_app(monkeypatch, _HistoryAgent(output="Net sales were $574,785 million."))
    at.run()
    at.chat_input[0].set_value("net sales?").run()
    at.chat_input[0].set_value("and the year before?").run()

    answers = [m.content for m in seen["history"] if "574,785" in m.content]
    assert answers, "the prior answer should be in history"
    assert "\\$" not in answers[0]
