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


def _stubbed_app(monkeypatch, agent):
    """AppTest runs app.py in its own namespace, so the stub goes in at the source.

    app.py resolves build_agent lazily inside load_agent, so patching
    finrag.agent.build_agent reaches it. st.cache_resource would otherwise hold
    the first stub for the whole process.
    """
    import streamlit as st

    import finrag.agent

    monkeypatch.setattr(finrag.agent, "build_agent", lambda **kwargs: agent)
    st.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=60)


def test_app_loads_without_error():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()

    assert not at.exception
    assert "SEC filing analyst" in [t.value for t in at.title]


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
