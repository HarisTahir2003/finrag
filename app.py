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
import os
import time

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from finrag.agent import answer_text
from finrag.config import get_settings
from finrag.ingest.download import list_filings
from finrag.llm import default_model_for, required_api_key
from finrag.presentation import (
    calculator_expression,
    describe_action,
    escape_dollars,
    parse_passages,
)

# Same search order as the CLI: the .env beside the working directory wins, and
# a real exported variable beats both.
load_dotenv(find_dotenv(usecwd=True))
load_dotenv()

# gRPC logs one INFO line per file descriptor after a fork, and the embedding
# model forks on load. Harmless, but it buries the server log.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

st.set_page_config(page_title="finrag — SEC filing analyst", page_icon="📈", layout="wide")

settings = get_settings()


@st.cache_data(show_spinner=False)
def coverage() -> dict[str, list[int]]:
    """Cached: reading every chunk's metadata takes seconds on a full corpus."""
    from finrag.ingest.index import corpus_coverage

    return corpus_coverage(settings)


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
        api_key = st.text_input(
            f"{settings.llm_backend.title()} API key",
            type="password",
            value=os.environ.get(key_var, ""),
            help=f"Read from {key_var}.",
        )
        if api_key:
            os.environ[key_var] = api_key

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


@st.cache_resource(show_spinner=False)
def load_agent(fingerprint: str):
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

    return build_agent(store=open_store(settings), settings=settings)


# ------------------------------------------------------------- provenance


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


def chat_history():
    """Prior turns as LangChain messages.

    The agent's prompt has always carried a `chat_history` placeholder and the
    UI never filled it, so every question started from nothing: asking "explain
    why apple's net income was higher" straight after a comparison got "please
    specify which year", which reads as the agent being dim rather than the
    front end forgetting.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    messages = []
    for turn in st.session_state.messages:
        role = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(role(content=turn["content"]))
    return messages


EXAMPLES = [
    "What was Apple's total net sales in fiscal 2024?",
    "Compare Apple and Amazon's net income in 2023",
    "What is Microsoft's debt-to-equity ratio for 2023?",
    "What cybersecurity risks does JP Morgan disclose in 2023?",
]

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
    if key_var is not None and not os.environ.get(key_var):
        st.warning(f"Enter a {settings.llm_backend.title()} API key in the sidebar first.")
    else:
        # Drawn at the top of this same run, before the conversation existed.
        # Emptying the slot withdraws them without a rerun, which would take the
        # progress log and the elapsed time with it.
        examples_slot.empty()

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
            with st.status("Reading filings…", expanded=True) as status:
                try:
                    agent = load_agent(key_fingerprint(api_key))
                    for chunk in agent.stream({"input": prompt, "chat_history": history}):
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
                    answer = f"Something went wrong: {exc}"
                    status.update(label="Failed", state="error", expanded=False)

            answer = answer.strip() or "The agent returned nothing."
            elapsed = time.perf_counter() - started
            st.markdown(escape_dollars(answer))
            render_sources(steps)
            st.caption(f"{elapsed:.0f}s · {settings.llm_backend} · {model_name}")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "steps": steps, "elapsed": elapsed}
            )
