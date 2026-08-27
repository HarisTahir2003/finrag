"""Streamlit chat interface over the finrag agent.

A thin front end: configuration, retrieval, tools and the agent all live in the
`finrag` package, so this file holds only what is genuinely about the UI. It is
scheduled for a fuller rewrite when the API layer lands.

    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from finrag.config import get_settings
from finrag.ingest.download import list_filings
from finrag.llm import default_model_for, required_api_key

load_dotenv()

st.set_page_config(page_title="finrag — SEC filing analyst", page_icon="📈", layout="wide")

settings = get_settings()

with st.sidebar:
    st.title("Settings")

    model_name = settings.chat_model or default_model_for(settings.llm_backend)
    key_var = required_api_key(settings)
    if key_var is None:
        # Ollama runs locally and needs no credentials.
        api_key = "local"
        st.info(f"Running locally via {settings.llm_backend} — no API key needed.")
        st.caption(f"Model: {model_name}")
    else:
        api_key = st.text_input(
            f"{settings.llm_backend.title()} API key",
            type="password",
            value=os.environ.get(key_var, ""),
            help=f"Read from {key_var}. Model: {model_name}",
        )
        if api_key:
            os.environ[key_var] = api_key

    st.divider()
    st.caption("Index")
    st.code(
        f"backend   {settings.embedding_backend}\nchunking  {settings.chunk_strategy}\npath      {settings.index_dir}",
        language=None,
    )

    if settings.index_dir.exists():
        try:
            from finrag.ingest.index import collection_size

            st.success(f"{collection_size(settings):,} chunks indexed")
        except Exception as exc:  # noqa: BLE001 - the sidebar must not take the page down
            st.warning(f"Index unreadable: {exc}")
    else:
        st.error("No index found")
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


st.title("SEC filing analyst")
st.caption(
    "Ask about any company and fiscal year in the index. Figures are retrieved from the filing and computed, not recalled."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

placeholder = "Compare Apple and Amazon's current ratio in fiscal 2023"
if prompt := st.chat_input(placeholder):
    if key_var is not None and not os.environ.get(key_var):
        st.warning(f"Enter a {settings.llm_backend.title()} API key in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"), st.spinner("Reading filings…"):
            try:
                agent = load_agent(api_key[-8:])
                result = agent.invoke({"input": prompt})
                answer = result["output"]
                if isinstance(answer, list):
                    # Gemini can return a multi-part response; stitch it back together.
                    answer = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in answer
                    )
                answer = str(answer).strip()
            except Exception as exc:  # noqa: BLE001 - shown to the user rather than a stack trace
                answer = f"Something went wrong: {exc}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
