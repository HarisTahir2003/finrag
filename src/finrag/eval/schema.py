"""Evaluation cases, loaded from YAML rather than hardcoded in a notebook cell.

The original suite lived inside a code cell, which made it invisible to version
control diffs and impossible to run without executing the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATASETS_DIR = Path(__file__).parent / "datasets"


@dataclass(frozen=True)
class RetrievalCase:
    """One retrieval probe: a query, the filing it should hit, and proof it did."""

    id: str
    query: str
    ticker: str
    fiscal_year: int
    # Retrieval succeeded if any retrieved chunk contains one of these. Plain
    # substrings keep the check deterministic and LLM-free.
    expect_any: list[str] = field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class AgentCase:
    """One end-to-end question, with the tools it ought to use."""

    id: str
    question: str
    ticker: str
    fiscal_year: int
    expected_tools: list[str] = field(default_factory=list)
    reference_answer: str = ""
    category: str = ""
    note: str = ""


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_retrieval_cases(path: str | Path | None = None) -> list[RetrievalCase]:
    data = _load_yaml(Path(path) if path else DATASETS_DIR / "retrieval.yaml")
    return [RetrievalCase(**case) for case in data["cases"]]


def load_agent_cases(path: str | Path | None = None) -> list[AgentCase]:
    data = _load_yaml(Path(path) if path else DATASETS_DIR / "agent.yaml")
    return [AgentCase(**case) for case in data["cases"]]
