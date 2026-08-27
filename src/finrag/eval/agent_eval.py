"""End-to-end agent evaluation: did it use the right tools, and get the right answer.

The original suite reported a single "functional tool call accuracy" of 60%. The
dominant failure was the agent replying "I need a stock ticker -- which company
are you interested in?" for a ticker the question already named, because the
harness passed only the question text and the agent had no way to recover it.
That is scored explicitly here as ``clarification_requests`` rather than being
folded into a general failure count, because it is a harness and prompt problem
rather than a reasoning one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..config import Settings, get_settings
from .schema import AgentCase, load_agent_cases

log = logging.getLogger(__name__)

# Phrases the agent uses when it has failed to extract the company or year.
_CLARIFICATION = re.compile(
    r"(which company|what is the ticker|provide the ticker|need a (stock )?ticker|"
    r"which fiscal year|please provide)",
    re.I,
)


@dataclass
class AgentCaseResult:
    case: AgentCase
    answer: str
    tools_called: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def asked_for_clarification(self) -> bool:
        return bool(_CLARIFICATION.search(self.answer))

    @property
    def tool_path_ok(self) -> bool:
        """Every expected tool was called at least once."""
        return all(t in self.tools_called for t in self.case.expected_tools)

    @property
    def used_calculator_when_required(self) -> bool:
        """Arithmetic must be computed, not recalled."""
        if "calculator" not in self.case.expected_tools:
            return True
        return "calculator" in self.tools_called

    @property
    def cited_a_figure(self) -> bool:
        """A weak but LLM-free signal that the answer is grounded in numbers."""
        return bool(re.search(r"\d", self.answer))


@dataclass
class AgentReport:
    results: list[AgentCaseResult] = field(default_factory=list)

    def _rate(self, predicate) -> float:
        return sum(predicate(r) for r in self.results) / len(self.results) if self.results else 0.0

    def as_metrics(self) -> dict[str, float]:
        return {
            "tool_path_accuracy": round(self._rate(lambda r: r.tool_path_ok), 4),
            "calculator_compliance": round(
                self._rate(lambda r: r.used_calculator_when_required), 4
            ),
            "clarification_requests": round(self._rate(lambda r: r.asked_for_clarification), 4),
            "answered_with_figures": round(self._rate(lambda r: r.cited_a_figure), 4),
            "errors": sum(1 for r in self.results if r.error),
            "cases": len(self.results),
        }

    def format_table(self) -> str:
        lines = [f"{'case':28} {'tools':>6} {'calc':>5} {'clarify':>8}  tools called", "-" * 78]
        for r in self.results:
            lines.append(
                f"{r.case.id:28} {'ok' if r.tool_path_ok else 'FAIL':>6} "
                f"{'ok' if r.used_calculator_when_required else 'no':>5} "
                f"{'YES' if r.asked_for_clarification else '-':>8}  {','.join(r.tools_called) or '-'}"
            )
        return "\n".join(lines)


def _tools_from_steps(intermediate_steps) -> list[str]:
    names = []
    for step in intermediate_steps or []:
        action = step[0] if isinstance(step, (tuple, list)) else step
        name = getattr(action, "tool", None)
        if name:
            names.append(name)
    return names


def _flatten(output) -> str:
    """Gemini can return a list of content parts rather than a string."""
    if isinstance(output, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in output).strip()
    return str(output).strip()


def evaluate_agent(
    cases: list[AgentCase] | None = None,
    agent=None,
    store=None,
    settings: Settings | None = None,
    limit: int | None = None,
) -> AgentReport:
    """Run the agent over the question set. Requires GOOGLE_API_KEY."""
    settings = settings or get_settings()
    cases = cases if cases is not None else load_agent_cases()
    if limit:
        cases = cases[:limit]

    if agent is None:
        from ..agent import build_agent
        from ..ingest.index import open_store

        agent = build_agent(store=store or open_store(settings), settings=settings)

    report = AgentReport()
    for case in cases:
        # The ticker and fiscal year are stated in the prompt as well as being
        # present in the question, so a failure to use them is the agent's,
        # not the harness's.
        prompt = f"{case.question}\n\n(Company: {case.ticker}. Fiscal year: {case.fiscal_year}.)"
        try:
            raw = agent.invoke({"input": prompt})
            result = AgentCaseResult(
                case=case,
                answer=_flatten(raw.get("output", "")),
                tools_called=_tools_from_steps(raw.get("intermediate_steps")),
            )
        except Exception as exc:  # noqa: BLE001 - one bad case must not end the run
            log.error("%s: %s", case.id, exc)
            result = AgentCaseResult(case=case, answer="", error=str(exc))
        report.results.append(result)
        log.info("%-28s tools=%s", case.id, result.tools_called or "none")
    return report
