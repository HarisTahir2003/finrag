"""Streamlit chat interface over the finrag agent.

A thin front end: configuration, retrieval, tools and the agent all live in the
`finrag` package, so this file holds only what is genuinely about the UI.

The one thing it does beyond relaying text is show its working. An answer like
"$391,035 million" is worth exactly as much as the passage behind it, and an
agent that quietly invented the number looks identical to one that retrieved it.
Every tool call is surfaced as it happens, and the retrieved passages are kept
and shown under the answer.

    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path

import streamlit as st
from dotenv import find_dotenv, load_dotenv

# A platform-as-a-service clones the repository, installs a requirements file
# and runs this script from the repository root -- it does not `pip install`
# the project, so `finrag` is not importable yet. Adding src/ costs nothing
# where the package IS installed properly: the same files resolve either way.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from finrag.agent import answer_text  # noqa: E402
from finrag.bootstrap import ensure_index  # noqa: E402
from finrag.config import get_settings  # noqa: E402
from finrag.ingest.download import list_filings  # noqa: E402
from finrag.llm import (  # noqa: E402
    classify_provider_error,
    default_model_for,
    fits_multi_filing_question,
    required_api_key,
)
from finrag.logconfig import configure_logging  # noqa: E402
from finrag.presentation import (  # noqa: E402
    calculator_expression,
    describe_action,
    escape_dollars,
    failure_message,
    limit_message,
    parse_passages,
)
from finrag.ratelimit import spend_check  # noqa: E402

# Same search order as the CLI: the .env beside the working directory wins, and
# a real exported variable beats both.
load_dotenv(find_dotenv(usecwd=True))
load_dotenv()

# gRPC logs one INFO line per file descriptor after a fork, and the embedding
# model forks on load. Harmless, but it buries the server log.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

# Streamlit installs the root handler; what was missing is the level on
# finrag's own loggers, without which every INFO line below is dropped. On a
# hosted deployment these logs are the only instrument there is.
configure_logging()

# Named under "finrag." so the package's level applies; __name__ here is
# "__main__", which inherits the root's WARNING and would drop every INFO line.
log = logging.getLogger("finrag.app")


def session_id() -> str:
    """A stable, anonymous identifier for this browser session.

    Streamlit gives each connected browser its own session_state, so a value
    stored here is exactly "one visitor" for as long as the tab is open. It is
    random and never leaves the process -- it identifies a rate-limit bucket
    and a log line, not a person.
    """
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex[:12]
    return st.session_state["session_id"]


st.set_page_config(page_title="finrag — SEC filing analyst", page_icon="📈", layout="wide")

# Hosted Streamlit puts secrets in st.secrets, not the environment, and every
# module in `finrag` reads os.environ. Copying them across here rather than
# teaching each module about Streamlit keeps the package usable from the CLI,
# the API and the tests. setdefault, so a real environment variable still wins
# and a local .env is not overridden by a stale secrets file.
#
# Scalars, not just strings, and str()-ed on the way through. TOML parses an
# unquoted `FINRAG_RERANK = false` as a boolean, and an `isinstance(_,str)`
# filter dropped it silently -- so the single most important switch on the
# deployment reverted to its default (rerank ON, ~1.1GB on a 690MB host, a
# silent OOM) if the operator wrote the natural TOML instead of the quoted
# form. str(False) -> "False" -> config lower-cases it to "false", which it
# already handles. Nested tables (dict/list values) are still skipped: those
# are not environment variables.
try:
    for _name, _value in dict(st.secrets).items():
        if isinstance(_value, (str, bool, int, float)):
            os.environ.setdefault(_name, str(_value))
except Exception:  # noqa: BLE001 - no secrets file is the normal local case
    pass

settings = get_settings()

# On a host that only gives us a git checkout there is no 134MB index, because
# 134MB does not go in git -- a 45MB archive of it does. Unpacks once per
# container, then returns immediately forever after.
try:
    if ensure_index(settings):
        st.toast("Unpacked the search index", icon="📦")
except Exception:  # noqa: BLE001 - the index gate below reports this properly
    # The exception text carries host filesystem paths, which a public page
    # should not show. logging.exception puts the traceback in the server log,
    # where the person who can act on it is looking.
    logging.getLogger(__name__).exception("could not unpack the index")
    st.error("Could not prepare the search index. The server log has the details.")


@st.cache_data(show_spinner=False)
def coverage() -> dict[str, list[int]]:
    """Cached: reading every chunk's metadata takes seconds on a full corpus."""
    from finrag.ingest.index import corpus_coverage

    return corpus_coverage(settings)


def looks_like_api_key(value: str | None) -> bool:
    """A cheap shape check: does this look like a real provider key at all?

    Used only to decide whether the rate-limit waiver applies. A visitor with a
    real key spends their own quota, so waiving the limit is right; but the
    waiver keyed on "any non-empty string" let a scripted client type junk to
    become exempt. A key is 20+ characters of the key alphabet (gsk_…, sk-…,
    AIza…); "asdf" and "let me in" are not. This is not validation -- only the
    provider can say if a key is real -- just enough that nonsense is rationed
    like the no-key path instead of buying a free pass.
    """
    if not value:
        return False
    value = value.strip()
    return len(value) >= 20 and bool(re.fullmatch(r"[A-Za-z0-9_.\-]+", value))


# ----------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("Settings")

    model_name = settings.chat_model or default_model_for(settings.llm_backend)
    key_var = required_api_key(settings)

    if key_var is None:
        api_key = "no-key-required"
        # ollama runs on this machine; vertex authenticates with Application
        # Default Credentials. Neither takes an API key, and only one of them is
        # local -- the old copy called both "local", which is wrong for vertex.
        where = (
            "on this machine" if settings.llm_backend == "ollama" else "with ambient credentials"
        )
        st.info(f"`{settings.llm_backend}` authenticates {where} — no API key needed.")
    else:
        # NEVER pre-fill this from the environment. A widget's default value is
        # sent to every browser that opens the page, so `value=os.environ[...]`
        # publishes the host's key to every visitor before they click anything;
        # type="password" masks the rendering and nothing else. Confirmed by
        # reading the widget back with a sentinel key in the environment. A
        # placeholder says a key is configured without being one.
        configured = bool(os.environ.get(key_var))
        typed = st.text_input(
            f"{settings.llm_backend.title()} API key",
            type="password",
            placeholder=(
                "Using the key configured on the server" if configured else f"Paste a {key_var}"
            ),
            help=(
                f"Optional while the server provides {key_var}. A key you enter "
                "is kept for your session only and is never written to disk."
            ),
        )
        # Session state rather than os.environ: one process serves every
        # visitor, so writing a key into the environment would hand it to the
        # next person to load the page. Theirs wins over the server's for them
        # alone.
        #
        # The else is not optional. Without it, emptying the box keeps the old
        # key for the whole session -- so a visitor who pastes a truncated key,
        # is told (correctly, by failure_message) to "leave the box empty to
        # fall back to the server", clears it, and asks again, gets the same
        # rejection with no way back short of a hard reload.
        if typed:
            st.session_state["api_key"] = typed
        else:
            st.session_state.pop("api_key", None)
        api_key = st.session_state.get("api_key") or os.environ.get(key_var, "")

    st.caption(f"Model: `{model_name}`")

    st.divider()
    st.caption("Retrieval")
    st.code(
        f"embeddings  {settings.embedding_backend}\n"
        f"chunking    {settings.chunk_strategy}\n"
        f"mode        {settings.retrieval_mode}"
        f"{' + rerank' if settings.rerank else ''}\n"
        f"top k       {settings.retrieval_k}",
        language=None,
    )

    from finrag.ingest.index import index_status

    index_ready, size, reason = index_status(settings)
    if index_ready:
        st.success(f"{size:,} chunks indexed")
    else:
        st.error(reason.capitalize())

    if not index_ready:
        st.caption(f"{len(list_filings(settings=settings))} filings downloaded. Build one with:")
        st.code("finrag download\nfinrag index", language="bash")
    else:
        # "Ask about any company and fiscal year in the index" is not usable
        # advice unless something says what is in it.
        with st.expander("What is indexed"):
            for ticker, years in coverage().items():
                st.markdown(f"**{ticker}** &nbsp; FY{years[0]}–{years[-1]}")

    if st.session_state.get("messages"):
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def key_fingerprint(api_key: str) -> str:
    """A stable identifier for a key that is not itself key material."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


# max_entries bounds the cache. The default is unbounded, and one entry -- an
# AgentExecutor holding a ChatGroq with its own sync and async httpx pools and
# a live TLS socket -- is created per distinct key fingerprint and never freed.
# On a shared host that is a slow leak a scripted client can turn into an OOM by
# pasting fresh strings; eight covers the server key plus a handful of
# bring-your-own visitors, and ttl reclaims the rest.
@st.cache_resource(show_spinner=False, max_entries=8, ttl=3600)
def load_agent(fingerprint: str, _api_key: str | None = None):
    """Build the agent once per API key.

    The fingerprint is a cache parameter rather than a closure variable on
    purpose: an earlier version closed over it, so the first key entered was
    pinned and changing it silently had no effect.

    It must NOT be named with a leading underscore. Streamlit deliberately
    excludes underscore-prefixed arguments from the cache key -- the convention
    for passing unhashable things like a database handle -- so `_fingerprint`
    leaves this function with an empty key set and one entry forever, which is
    the exact bug the paragraph above claims to have fixed. It was named that
    way once and reinstated it. The key is read from the environment inside
    build_agent and baked into the client at construction, so a stale entry
    means a corrected or rotated key never takes effect and every question
    keeps failing against the old one.
    """
    from finrag.agent import build_agent
    from finrag.ingest.index import open_store

    # Underscore-prefixed *here* on purpose, and only here: Streamlit excludes
    # it from the cache key, which is exactly right for the secret itself --
    # `fingerprint` above already identifies it, and the raw key has no business
    # in a cache key. The two arguments are the same fact at two sensitivities.
    overrides = {"api_key": _api_key} if (_api_key and required_api_key(settings)) else {}
    return build_agent(store=open_store(settings), settings=settings, **overrides)


# ------------------------------------------------------------- provenance


# An old answer's provenance is kept for scroll-back, not for re-reading the
# whole filing. Bounding each stored observation keeps session_state and the
# per-rerun payload flat as a conversation grows, while still showing the first
# passages behind a historical answer.
_HISTORY_OBSERVATION_CHARS = 2000


def _trim_steps_for_history(steps: list[dict]) -> list[dict]:
    """A copy of ``steps`` with each observation bounded for storage."""
    trimmed = []
    for step in steps:
        observation = step["observation"]
        if len(observation) > _HISTORY_OBSERVATION_CHARS:
            observation = observation[:_HISTORY_OBSERVATION_CHARS] + "\n… (truncated)"
        trimmed.append({**step, "observation": observation})
    return trimmed


def render_sources(steps: list[dict]) -> None:
    """Show what the answer was built from: passages retrieved, sums computed."""
    searches = [s for s in steps if s["tool"] == "search_10k_reports"]
    sums = [s for s in steps if s["tool"] == "calculator"]
    if not searches and not sums:
        return

    total = sum(len(parse_passages(s["observation"])[1]) for s in searches)
    label = f"{total} passage{'s' if total != 1 else ''} from {len(searches)} search"
    label += "es" if len(searches) != 1 else ""
    if sums:
        label += f" · {len(sums)} calculation{'s' if len(sums) != 1 else ''}"

    with st.expander(label):
        for step in sums:
            st.markdown(f"**`{calculator_expression(step['input'])}`** = `{step['observation']}`")
        if sums and searches:
            st.divider()
        for step in searches:
            filing, passages = parse_passages(step["observation"])
            query = step["input"].get("query", "") if isinstance(step["input"], dict) else ""
            st.markdown(f"**{filing}** — searched for *{query}*")
            for passage in passages:
                # Monospace, because these are statement tables rendered as
                # pipe-delimited rows and proportional text destroys the
                # columns. wrap_lines keeps a wide row inside the expander
                # instead of adding a second scrollbar.
                st.code(passage, language=None, wrap_lines=True)


# ---------------------------------------------------------------- the chat

st.title("SEC filing analyst")
st.caption(
    "Ask about any company and fiscal year in the index. "
    "Figures are retrieved from the filing and computed, not recalled."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(escape_dollars(message["content"]))
        if message.get("steps"):
            render_sources(message["steps"])
        if message.get("elapsed"):
            st.caption(f"{message['elapsed']:.0f}s · {settings.llm_backend} · {model_name}")


# How many prior turns to carry. Six -- three exchanges -- is enough to resolve
# "why was it higher?" against the question before it, which is the whole reason
# history exists here.
#
# The cap is not a nicety. The agent resends its entire scratchpad on every LLM
# round trip, and one search already spends ~2,700 real tokens against Groq's
# 8,000-per-minute free tier. Unbounded history meant a two-search question
# began failing with a 413 after ~8 turns and then failed *forever*, because
# history only grows -- and the error told the visitor to "ask about one
# company", which is not what was wrong. A fixed tail keeps the request size
# flat no matter how long the conversation runs.
MAX_HISTORY_TURNS = 6


def chat_history():
    """The most recent turns as LangChain messages -- see MAX_HISTORY_TURNS.

    The agent's prompt has always carried a `chat_history` placeholder and the
    UI never filled it, so every question started from nothing: asking "explain
    why apple's net income was higher" straight after a comparison got "please
    specify which year", which reads as the agent being dim rather than the
    front end forgetting.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    messages = []
    for turn in st.session_state.messages[-MAX_HISTORY_TURNS:]:
        role = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(role(content=turn["content"]))
    return messages


# One filing each. Deliberately: a two-company comparison sends both filings in
# one request, which is ~8,600 tokens against Groq's free 8,000/minute, so it
# fails every time it is offered. A suggested question that reliably fails is
# worse than one fewer suggestion -- the visitor reads it as the system being
# broken, and on the evidence in front of them they are right.
EXAMPLES = [
    "What was Apple's total net sales in fiscal 2024?",
    "What was Amazon's net income in 2023?",
    "What is Microsoft's debt-to-equity ratio for 2023?",
    "What cybersecurity risks does JP Morgan disclose in 2023?",
]

# Offered only where the context budget allows it -- Vertex has no such cap.
if fits_multi_filing_question(settings):
    EXAMPLES.insert(2, "Compare Apple and Amazon's net income in 2023")

# An empty chat with only a placeholder does not say what the thing can do.
# These disappear once a conversation starts rather than sitting under it.
picked = None
examples_slot = st.container()
if index_ready and not st.session_state.messages:
    with examples_slot:
        st.caption("Try one of these, or ask your own:")
        picked = st.pills("Examples", EXAMPLES, label_visibility="collapsed")

placeholder = "Ask about a company and fiscal year…"
prompt = st.chat_input(placeholder, disabled=not index_ready) or picked

if prompt:
    if key_var is not None and not api_key:
        st.warning(f"Enter a {settings.llm_backend.title()} API key in the sidebar first.")
    else:
        # Drawn at the top of this same run, before the conversation existed.
        # Emptying the slot withdraws them without a rerun, which would take the
        # progress log and the elapsed time with it.
        examples_slot.empty()

        # A visitor's own key is their own budget; only the shared one is
        # rationed. Checked before any work is done, so a refused question
        # costs nothing. The waiver requires a key that at least *looks* like a
        # key -- otherwise typing any junk string bought exemption from the
        # limits, and combined with the (now bounded) agent cache that was a
        # free resource loop.
        on_the_house = not looks_like_api_key(st.session_state.get("api_key"))
        verdict = spend_check(settings, session_id(), shared_key=on_the_house)
        if not verdict.allowed:
            st.warning(limit_message(verdict, settings))
            st.stop()

        history = chat_history()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(escape_dollars(prompt))

        with st.chat_message("assistant"):
            steps: list[dict] = []
            answer = ""
            started = time.perf_counter()
            # Streaming rather than invoke(): a question takes tens of seconds
            # across several tool calls, and a bare spinner for that long is
            # indistinguishable from a hang.
            request_id = uuid.uuid4().hex[:8]
            deadline = started + settings.answer_timeout_seconds
            timed_out = False
            log.info(
                "q start id=%s session=%s backend=%s chars=%d",
                request_id,
                session_id(),
                settings.llm_backend,
                len(prompt),
            )
            with st.status("Reading filings…", expanded=True) as status:
                try:
                    agent = load_agent(key_fingerprint(api_key), api_key)
                    for chunk in agent.stream({"input": prompt, "chat_history": history}):
                        # Checked between steps rather than with a signal: the
                        # agent runs in Streamlit's own script thread, where a
                        # SIGALRM is not available and killing the thread would
                        # leave the provider client mid-request. Breaking here
                        # stops the next tool call, which is where the time
                        # actually goes.
                        if time.perf_counter() > deadline:
                            timed_out = True
                            log.warning(
                                "q timeout id=%s after=%.0fs steps=%d",
                                request_id,
                                time.perf_counter() - started,
                                len(steps),
                            )
                            break
                        for action in chunk.get("actions", []):
                            st.write(describe_action(action.tool, action.tool_input))
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
                    status.update(
                        label=f"Answered in {time.perf_counter() - started:.0f}s",
                        state="complete",
                        expanded=False,
                    )
                except Exception as exc:  # noqa: BLE001 - shown rather than a stack trace
                    answer = failure_message(exc, settings.llm_backend)
                    log.warning(
                        "q failed id=%s after=%.0fs kind=%s",
                        request_id,
                        time.perf_counter() - started,
                        classify_provider_error(exc),
                    )
                    status.update(label="Failed", state="error", expanded=False)

            if timed_out and not answer:
                answer = (
                    f"**That took longer than {settings.answer_timeout_seconds} seconds, so I "
                    "stopped waiting.** The provider may be busy. Try again, or ask something "
                    "narrower -- a single figure from a single filing is much quicker than a "
                    "question that needs several searches."
                )

            answer = answer.strip() or "The agent returned nothing."
            elapsed = time.perf_counter() - started
            log.info(
                "q done id=%s elapsed=%.1fs steps=%d answer_chars=%d timed_out=%s",
                request_id,
                elapsed,
                len(steps),
                len(answer),
                timed_out,
            )
            st.markdown(escape_dollars(answer))
            render_sources(steps)
            st.caption(f"{elapsed:.0f}s · {settings.llm_backend} · {model_name}")
            # The just-rendered answer keeps its full provenance; the copy kept in
            # session_state does not. Each observation is the whole ~8,000-char
            # retrieval context, stored per step, and every historical turn is
            # re-rendered and re-serialised to the browser on every rerun -- a
            # 20-turn conversation would re-ship a third of a megabyte of passage
            # text on each keystroke. The trimmed copy still shows the leading
            # passages of an old answer, which is all a scroll-back needs.
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "steps": _trim_steps_for_history(steps),
                    "elapsed": elapsed,
                }
            )
