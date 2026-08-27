"""
Financial Analyst AI Streamlit Application

Description:
This Streamlit app provides an interactive interface for querying and analyzing
financial data from SEC 10-K filings using an agentic AI powered by Google Gemini.

Key Features:
1. User Interface:
   - Sidebar to input Google API Key and check database availability.
   - Main chat interface for asking financial questions

2. Data Access:
   - Connects to a local Chroma vector database containing pre-embedded 10-K filings.
   - Uses GoogleGenerativeAIEmbeddings for semantic search over financial reports.

3. Agentic Reasoning:
   - Defines a structured tool (`search_10k_reports`) to query specific ticker and year data.
   - Provides a Python REPL tool (`python_calculator`) for performing financial calculations.
   - Creates a tool-calling agent that can reason, combine data retrieval, and perform math.

4. Chat Interaction:
   - Maintains chat history within the session state.
   - Supports iterative queries, comparison of multiple companies, and detailed explanations.
"""

import streamlit as st
import os
import re
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool, Tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool
from pydantic import BaseModel, Field


st.set_page_config(page_title="Financial Analyst AI", page_icon="📈", layout="wide")

with st.sidebar:
    st.title("Settings")
    google_api_key = st.text_input("Google API Key", type="password")
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    
    st.markdown("---")
    data_root = os.environ.get("FINRAG_DATA_ROOT", "./data")
    project_folder = os.path.join(data_root, "Financial_Analyzer_Project")
    db_path = os.path.join(project_folder, "chroma_db_financial_semantic")
    
    if os.path.exists(db_path):
        st.success("Database Found ✅")
    else:
        st.error("Database Not Found ❌")

def process_agent_response(response_output):
    """
    Ensures that multi-part Gemini responses are stitched together
    into a single full narrative.
    """
    if isinstance(response_output, list):
        parts = []
        for block in response_output:
            if isinstance(block, dict):
                content = block.get('text') or block.get('content')
                if content:
                    parts.append(str(content))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    
    return str(response_output).strip()

@st.cache_resource
def initialize_agent():
    if not google_api_key:
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

    def query_10k(query: str, ticker: str, year: int):
        filter_dict = {"$and": [{"ticker": ticker}, {"year": year}]}
        results = vectorstore.similarity_search(query, k=15, filter=filter_dict)
        context = f"--- DATA FOR {ticker} ({year}) ---\n"
        for doc in results:
            context += f"\n{doc.page_content}\n"
        return context

    class SearchInput(BaseModel):
        query: str = Field(description="The financial data point to find.")
        ticker: str = Field(description="Ticker (e.g. AAPL).")
        year: int = Field(description="Year (e.g. 2023).")

    financial_tool = StructuredTool.from_function(
        func=query_10k, 
        name="search_10k_reports", 
        description="Finds data in 10-K filings. Required: ticker and year.",
        args_schema=SearchInput
    )

    repl = PythonREPLTool()
    python_tool = Tool(name="python_calculator", func=repl.run, description="Math calculations.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, max_output_tokens=4000)
    
    system_prompt = """You are a Senior Financial Research Agent.
    When comparing two companies, analyze them sequentially.
    1. Retrieve data for Company A. 2. Retrieve data for Company B. 3. Perform math. 
    Explain the results thoroughly. Do not be brief."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, [financial_tool, python_tool], prompt)
    return AgentExecutor(agent=agent, tools=[financial_tool, python_tool], 
                         verbose=True, handle_parsing_errors=True, max_iterations=15)


st.title("🤖 Agentic Financial Analyst")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Compare Apple and Amazon's inventory turnover in 2023"):
    if not google_api_key:
        st.warning("Enter API Key in sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            agent_exec = initialize_agent()
            with st.spinner("Analyzing multiple reports..."):
                response = agent_exec.invoke({"input": prompt})
                
                full_text = process_agent_response(response["output"])
                
                st.markdown(full_text)
                st.session_state.messages.append({"role": "assistant", "content": full_text})