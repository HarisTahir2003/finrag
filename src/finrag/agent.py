"""The tool-calling agent: retrieval plus arithmetic."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from .calculator import calculate_as_text
from .config import Settings, get_settings
from .retrieval import search_filing

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial research agent working from SEC 10-K filings.

Procedure:
1. Work out which company, which fiscal year, and which figures the question needs.
2. Call search_10k_reports once per company-year involved. It requires a ticker and a
   fiscal year; if the question does not name one, ask the user rather than guessing.
3. Call calculator for every arithmetic step. Do not compute in your head.
4. State the figures you retrieved and the formula you used before giving the result.

Fiscal years are labelled by the year the fiscal period ends, which is how the filings
themselves are indexed.

Common formulas:
- Current ratio      = total current assets / total current liabilities
- Acid test          = (total current assets - inventories) / total current liabilities
- Net profit margin  = net income / total revenue
- Debt-to-equity     = total debt / total shareholders' equity
- Inventory turnover = cost of goods sold / average inventory
"""


class SearchInput(BaseModel):
    query: str = Field(description="The financial data point or topic to find.")
    ticker: str = Field(description="Stock ticker, for example AAPL.")
    fiscal_year: int = Field(description="Fiscal year, for example 2023.")


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="A single arithmetic expression, for example (143566 - 6331) / 145308"
    )


def build_tools(store=None, settings: Settings | None = None) -> list:
    """Retrieval and calculator tools, bound to one vector store."""
    from langchain_core.tools import StructuredTool

    settings = settings or get_settings()

    def _search(query: str, ticker: str, fiscal_year: int) -> str:
        return search_filing(
            query, ticker, fiscal_year, store=store, settings=settings
        ).as_context()

    return [
        StructuredTool.from_function(
            func=_search,
            name="search_10k_reports",
            description=(
                "Search an indexed SEC 10-K filing. Requires a ticker and a fiscal year. "
                "Returns the most relevant passages, including financial statement tables."
            ),
            args_schema=SearchInput,
        ),
        StructuredTool.from_function(
            func=lambda expression: calculate_as_text(expression),
            name="calculator",
            description=(
                "Evaluate one arithmetic expression and return the result. Use for every "
                "calculation. Supports + - * / ** and abs, round, min, max, sum, sqrt, log."
            ),
            args_schema=CalculatorInput,
        ),
    ]


def build_agent(store=None, settings: Settings | None = None, verbose: bool = False):
    """Assemble the agent executor.

    Needs whichever provider extra matches settings.llm_backend, and that
    provider's API key. See finrag.llm.
    """
    try:
        from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as exc:
        raise ImportError(
            "The agent needs langchain-classic: pip install 'finrag[anthropic]' "
            "or pip install 'finrag[google]'"
        ) from exc

    from .llm import get_chat_model

    settings = settings or get_settings()
    tools = build_tools(store=store, settings=settings)

    llm = get_chat_model(settings)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=15,
        return_intermediate_steps=True,
    )
