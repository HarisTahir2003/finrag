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

import os

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from finrag.agent import answer_text
from finrag.config import get_settings
from finrag.ingest.download import list_filings
from finrag.llm import default_model_for, required_api_key
from finrag.presentation import calculator_expression, describe_action, parse_passages

# Same search order as the CLI: the .env beside the working directory wins, and
# a real exported variable beats both.
load_dotenv(find_dotenv(usecwd=True))
load_dotenv()

# gRPC logs one INFO line per file descriptor after a fork, and the embedding
# model forks on load. Harmless, but it buries the server log.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

st.set_page_config(page_title="finrag — SEC filing analyst", page_icon="📈", layout="wide")

settings = get_settings()

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


@st.cache_resource(show_spinner=False)
def load_agent(_key_fingerprint: str):
    """Build the agent once per API key.

    The key fingerprint is a cache parameter rather than a closure variable on
    purpose: the previous version closed over it, so the first key entered in a
    session was pinned and changing it silently had no effect.
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
            for i, passage in enumerate(passages, start=1):
                st.text_area(
                    f"{filing} passage {i}",
                    passage,
                    height=120,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"src-{id(step)}-{i}",
                )


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
        st.markdown(message["content"])
        if message.get("steps"):
            render_sources(message["steps"])

placeholder = "Compare Apple and Amazon's current ratio in fiscal 2023"
prompt = st.chat_input(placeholder, disabled=not index_ready)

if prompt:
    if key_var is not None and not os.environ.get(key_var):
        st.warning(f"Enter a {settings.llm_backend.title()} API key in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            steps: list[dict] = []
            answer = ""
            # Streaming rather than invoke(): a question takes tens of seconds
            # across several tool calls, and a bare spinner for that long is
            # indistinguishable from a hang.
            with st.status("Reading filings…", expanded=True) as status:
                try:
                    agent = load_agent(api_key[-8:])
                    for chunk in agent.stream({"input": prompt}):
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
                    status.update(label="Done", state="complete", expanded=False)
                except Exception as exc:  # noqa: BLE001 - shown rather than a stack trace
                    answer = f"Something went wrong: {exc}"
                    status.update(label="Failed", state="error", expanded=False)

            answer = answer.strip() or "The agent returned nothing."
            st.markdown(answer)
            render_sources(steps)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "steps": steps}
            )
