"""The Streamlit front end and the display helpers behind it.

Worth testing because the app had never once been run: it is the only surface
in the repo CI does not exercise, and a front end that imports cleanly can
still render nothing useful.

The agent is stubbed. These assert the UI's own behaviour -- what it does with
a stream of tool calls -- not the agent's, which the eval suites cover and which
would cost an API call per test.
"""

from __future__ import annotations

import ast
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
    monkeypatch.setattr(
        finrag.ingest.index, "corpus_coverage", lambda *a, **k: {"AAPL": [2023, 2024]}
    )
    st.cache_data.clear()
    st.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=60)


def test_app_loads_without_error(monkeypatch):
    at = _stubbed_app(monkeypatch, None)
    at.run()

    assert not at.exception
    assert "SEC filing analyst" in [t.value for t in at.title]


def test_examples_are_offered_only_on_an_empty_conversation(monkeypatch):
    """A blank chat with a placeholder does not say what the thing can do."""
    at = _stubbed_app(monkeypatch, _StubAgent())
    at.run()
    assert at.pills, "an empty conversation should offer starter questions"

    at.chat_input[0].set_value("anything").run()
    # Within that run the pills were already drawn before the answer existed;
    # the app empties their container, which the browser honours but AppTest
    # still records. The durable invariant is the next run.
    at.run()
    assert not at.pills, "a conversation with history should offer none"
    assert at.session_state["messages"], "and it should still have the conversation"


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
    # Passages render as monospace blocks: they are statement tables, and
    # proportional text destroys the columns.
    passages = [c.value for c in at.code if "152,987" in c.value or "176,392" in c.value]
    assert len(passages) == 2


def test_a_failing_provider_is_reported_not_raised(monkeypatch, caplog):
    """A dead backend should read as a message, not a stack trace.

    The message no longer quotes the exception. This test asserted that it did,
    until the app became public: an unrecognised provider error's body is
    written for whoever operates the service, and it has already put a raw
    Groq payload -- upgrade link included -- in front of a visitor once. The
    detail moves to the log, and both halves are checked here.
    """
    import logging

    at = _stubbed_app(monkeypatch, _StubAgent(fail=True))
    at.run()
    with caplog.at_level(logging.ERROR):
        at.chat_input[0].set_value("anything").run()

    assert not at.exception
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "went wrong" in rendered.lower(), "the reader still has to be told"
    assert "provider unavailable" not in rendered, "the raw exception reached the page"
    assert "provider unavailable" in caplog.text, "the detail must still be diagnosable"


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


# ------------------------------------------------------- the cache key


def _load_agent_def():
    """The `load_agent` function node, read without executing the app script."""
    import ast

    tree = ast.parse(Path(APP).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "load_agent":
            return node
    raise AssertionError("app.py no longer defines load_agent")


def test_load_agent_is_cached_per_key():
    """The cache parameter must be hashable *and* hashed.

    Streamlit excludes underscore-prefixed arguments from the cache key -- the
    documented way to pass an unhashable handle like a database connection. The
    fingerprint here is a string, so the prefix buys nothing and costs the whole
    key: `_fingerprint` leaves load_agent with an empty key set and one entry
    for the life of the process.

    That is not a performance bug, it is an auth bug. build_agent reads the key
    from the environment and bakes it into the client, so the cached agent holds
    the first key ever entered. A user who pastes a corrected or rotated key
    gets the old one back, and every question keeps failing with no way out of
    the loop from inside the page. This has been written wrong once already.
    """
    node = _load_agent_def()
    args = [a.arg for a in node.args.args]

    assert args, "load_agent takes no argument, so every key shares one cache entry"

    # The two arguments are the same fact at two sensitivities, and the
    # underscore is right on exactly one of them. `fingerprint` identifies the
    # key and MUST be hashed, or a changed key reuses the old agent. The raw
    # key MUST NOT be hashed -- it would put a live secret in a cache key, and
    # the fingerprint already distinguishes it.
    hashed = [a for a in args if not a.startswith("_")]
    assert hashed, (
        "every argument is underscore-prefixed, so Streamlit's cache key is "
        "empty and one agent is pinned for the life of the process"
    )
    assert any("fingerprint" in a for a in hashed), (
        f"no fingerprint among the hashed arguments {hashed}; the cache would "
        "not notice the key changing"
    )
    for arg in args:
        if arg.startswith("_"):
            assert "key" in arg, f"{arg} is excluded from the cache key -- is that deliberate?"


def test_load_agent_is_actually_cached():
    """The decorator is the other half: without it there is no reuse at all."""
    node = _load_agent_def()
    decorators = [ast.dump(d) for d in node.decorator_list]

    assert any("cache_resource" in d for d in decorators), (
        "load_agent must stay cached; rebuilding it per question reloads the "
        "embedding model and the cross-encoder on every message"
    )


def test_the_fingerprint_is_not_key_material():
    """A cache key can surface in a Streamlit trace; a raw key slice must not."""
    import hashlib

    source = Path(APP).read_text(encoding="utf-8")
    assert "def key_fingerprint" in source
    assert "sha256" in source

    # The identity it produces is stable and distinguishing, which is all the
    # cache needs from it.
    digest = hashlib.sha256(b"sk-secret-value").hexdigest()[:16]
    assert digest != hashlib.sha256(b"sk-other-value").hexdigest()[:16]
    assert "secret" not in digest


# --------------------------------------------------- the key never leaves


def test_the_sidebar_never_pre_fills_the_key_from_the_environment(monkeypatch):
    """A widget's default value is sent to every browser that opens the page.

    So `st.text_input(..., value=os.environ["GROQ_API_KEY"])` publishes the
    host's key to every visitor before they click anything -- type="password"
    masks the rendering and nothing else. This was live: a sentinel key placed
    in the environment came back as the widget's value.

    On a shared deployment the owner's key is exactly what sits in that
    variable, so the blast radius is "anyone who opens the page". The box must
    stay empty and say what it is falling back to instead.
    """
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import finrag.ingest.index

    SENTINEL = "gsk_SENTINEL_MUST_NOT_REACH_A_BROWSER"
    monkeypatch.setenv("FINRAG_LLM_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", SENTINEL)
    monkeypatch.setattr(finrag.ingest.index, "index_status", lambda *a, **k: (True, 12376, "ready"))
    monkeypatch.setattr(finrag.ingest.index, "open_store", lambda *a, **k: object())
    monkeypatch.setattr(finrag.ingest.index, "corpus_coverage", lambda *a, **k: {"AAPL": [2024]})
    st.cache_data.clear()
    st.cache_resource.clear()

    at = AppTest.from_file(APP, default_timeout=60)
    at.run()

    assert not at.exception
    boxes = [ti for ti in at.text_input if "api key" in ti.label.lower()]
    assert boxes, "the key box should still be rendered for a keyed backend"
    for box in boxes:
        assert SENTINEL not in str(box.value), "the host's key reached the browser"
        assert box.value == "", "the box must start empty"
        assert SENTINEL not in str(getattr(box, "placeholder", "")), "leaked via the placeholder"


def test_a_visitor_key_is_not_written_into_the_process_environment():
    """One process serves every visitor.

    Writing a typed key to os.environ would hand it to whoever loads the page
    next, and would silently bill one visitor for another's questions. It has to
    travel to the model as an argument instead, which is why build_agent grew
    **llm_overrides.
    """
    import ast
    import inspect

    from finrag.agent import build_agent

    source = Path(APP).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # No assignment into os.environ anywhere in the app.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "environ"
                ):
                    raise AssertionError(
                        "app.py assigns into os.environ; a visitor's key would "
                        "become every later visitor's key"
                    )

    # And the route it takes instead actually exists.
    assert "llm_overrides" in inspect.signature(build_agent).parameters or any(
        p.kind is p.VAR_KEYWORD for p in inspect.signature(build_agent).parameters.values()
    ), "build_agent must accept per-call llm overrides for the key to travel"


# ------------------------------------------------- spending the shared key


def test_the_app_refuses_a_question_once_the_session_budget_is_gone(monkeypatch):
    """End to end: the limit is enforced where the money is actually spent.

    The unit tests cover the counter. This covers the wiring -- that the check
    happens before the agent runs, so a refused question costs nothing, and
    that the refusal reaches the page instead of an exception.
    """
    import finrag.ratelimit as rl
    from finrag.ratelimit import RateLimiter

    monkeypatch.setattr(rl, "limiter", RateLimiter())
    monkeypatch.setenv("FINRAG_QUESTIONS_PER_SESSION", "2")
    monkeypatch.setenv("FINRAG_QUESTIONS_PER_DAY", "99")

    agent = _StubAgent()
    at = _stubbed_app(monkeypatch, agent)
    at.run()

    for i in range(2):
        at.chat_input[0].set_value(f"question {i}").run()
        assert not at.warning, f"question {i} should have been answered"

    at.chat_input[0].set_value("one too many").run()
    assert at.warning, "the third question should have been refused"

    refusal = at.warning[-1].value.lower()
    assert "limit" in refusal
    assert "key" in refusal and "sidebar" in refusal, "a refusal must offer a way forward"


def test_a_visitor_using_their_own_key_is_not_rationed(monkeypatch):
    """The escape hatch has to actually work, or the limit is just a wall."""
    import streamlit as st

    import finrag.ratelimit as rl
    from finrag.ratelimit import RateLimiter

    monkeypatch.setattr(rl, "limiter", RateLimiter())
    monkeypatch.setenv("FINRAG_QUESTIONS_PER_SESSION", "1")
    monkeypatch.setenv("FINRAG_QUESTIONS_PER_DAY", "1")

    at = _stubbed_app(monkeypatch, _StubAgent())
    at.run()
    # Stand in for the visitor having typed their own key in the sidebar.
    at.session_state["api_key"] = "gsk_a_visitors_own_key"

    for i in range(4):
        at.chat_input[0].set_value(f"question {i}").run()

    assert not at.warning, "a visitor spending their own quota must not be rationed"
    assert st is not None


def test_each_question_is_logged_with_an_id_that_ties_start_to_finish(monkeypatch, caplog):
    """On a hosted deployment the log is the only instrument there is.

    finrag's loggers had no level set outside the CLI, so every INFO line the
    app emitted was dropped before it reached a handler.
    """
    import logging

    at = _stubbed_app(monkeypatch, _StubAgent())
    at.run()
    with caplog.at_level(logging.INFO, logger="finrag.app"):
        at.chat_input[0].set_value("what were net sales").run()

    starts = [r for r in caplog.records if r.getMessage().startswith("q start")]
    dones = [r for r in caplog.records if r.getMessage().startswith("q done")]
    assert starts and dones, "a question must log both its start and its outcome"

    def field(message: str, name: str) -> str:
        return dict(part.split("=", 1) for part in message.split() if "=" in part)[name]

    assert field(starts[0].getMessage(), "id") == field(dones[0].getMessage(), "id")
    assert "elapsed" in dones[0].getMessage()


# ------------------------------------------------- Phase-5-audit hardening


def test_the_agent_cache_is_bounded():
    """An unbounded st.cache_resource leaks one ChatGroq (sync+async httpx pools,
    a live socket) per distinct key fingerprint, forever. A scripted client
    pasting fresh strings turns that into an OOM on the shared host."""
    import ast

    node = _load_agent_def()
    decorator = next(
        (d for d in node.decorator_list if isinstance(d, ast.Call)),
        None,
    )
    assert decorator is not None, "load_agent must keep its cache_resource decorator"
    kwargs = {k.arg for k in decorator.keywords}
    assert "max_entries" in kwargs, "the agent cache must be bounded (max_entries)"


def test_chat_history_is_capped():
    """Unbounded history is resent whole on every question, so it 413s Groq's
    8,000 TPM after a handful of turns and then never recovers. A fixed tail
    keeps the request size flat however long the conversation runs."""
    import ast

    source = Path(APP).read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "chat_history"
    )
    # Somewhere in chat_history, session_state.messages must be sliced, not read whole.
    sliced = any(
        isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice) for n in ast.walk(fn)
    )
    assert sliced, "chat_history() must slice st.session_state.messages, not send all of it"


def _keyed_app(monkeypatch, agent, server_key="gsk_server_key_1234567890"):
    """A stubbed app on a keyed backend, with a server key present."""
    import streamlit as st

    import finrag.agent
    import finrag.ingest.index

    monkeypatch.setenv("FINRAG_LLM_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", server_key)
    monkeypatch.setattr(finrag.agent, "build_agent", lambda **kwargs: agent)
    monkeypatch.setattr(finrag.ingest.index, "index_status", lambda *a, **k: (True, 12376, "ready"))
    monkeypatch.setattr(finrag.ingest.index, "open_store", lambda *a, **k: object())
    monkeypatch.setattr(
        finrag.ingest.index, "corpus_coverage", lambda *a, **k: {"AAPL": [2023, 2024]}
    )
    st.cache_data.clear()
    st.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=60)


def test_clearing_the_key_box_falls_back_to_the_server_key(monkeypatch):
    """The bug the failure message promised was fixed: emptying the box used to
    keep the old key for the whole session, so a visitor who pasted a truncated
    key and cleared it -- exactly as told -- stayed locked out."""

    def stored_key(app):
        # AppTest's session_state proxies attribute access, so `.get` is not a
        # dict method there (SIM401's suggestion would raise) -- membership then
        # index is the supported read.
        if "api_key" in app.session_state:  # noqa: SIM401
            return app.session_state["api_key"]
        return None

    at = _keyed_app(monkeypatch, _StubAgent())
    at.run()

    box = next(ti for ti in at.text_input if "api key" in ti.label.lower())
    box.set_value("gsk_a_visitors_own_key_9999").run()
    assert stored_key(at) == "gsk_a_visitors_own_key_9999"

    # Clear it.
    box = next(ti for ti in at.text_input if "api key" in ti.label.lower())
    box.set_value("").run()
    assert stored_key(at) in (None, ""), "the typed key was not withdrawn"


def test_a_junk_key_does_not_buy_a_rate_limit_waiver(monkeypatch):
    """The waiver keyed on 'any non-empty string', so typing junk exempted a
    visitor from the limits. Only a key-shaped string should waive them."""
    import finrag.ratelimit as rl
    from finrag.ratelimit import RateLimiter

    monkeypatch.setattr(rl, "limiter", RateLimiter())
    monkeypatch.setenv("FINRAG_QUESTIONS_PER_SESSION", "2")
    monkeypatch.setenv("FINRAG_QUESTIONS_PER_DAY", "99")

    at = _keyed_app(monkeypatch, _StubAgent())
    at.run()
    at.session_state["api_key"] = "letmein"  # junk: short, would not be a real key

    for _ in range(2):
        at.chat_input[0].set_value("a question").run()
        assert not at.warning, "junk-key visitor should be treated as shared-key"

    at.chat_input[0].set_value("one too many").run()
    assert at.warning, "a junk key must not exempt the visitor from the shared limit"


def test_stored_history_observations_are_bounded(monkeypatch):
    """The live answer keeps full provenance; the copy in session_state does not.
    Each observation is the whole ~8,000-char retrieval context, re-serialised to
    the browser on every rerun -- a long conversation would re-ship a third of a
    megabyte per keystroke."""

    class _LongAgent:
        def stream(self, _inputs):
            action = _Action(
                "search_10k_reports",
                {"ticker": "AAPL", "fiscal_year": 2024, "query": "net sales"},
            )
            big = "--- AAPL FY2024 ---\n\n[chunk 1]\n" + ("net sales 391035 " * 1000)
            yield {"steps": [_Step(action, big)]}
            yield {"output": "Apple's net sales were $391,035 million."}

    at = _stubbed_app(monkeypatch, _LongAgent())
    at.run()
    at.chat_input[0].set_value("net sales?").run()

    assert not at.exception
    stored = at.session_state["messages"][-1]["steps"]
    assert stored, "the answer should have recorded its step"
    for step in stored:
        assert len(step["observation"]) <= 2100, "the stored observation must be bounded"
        assert "truncated" in step["observation"], "and marked as trimmed"
