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
# Anchored to interrogative or imperative forms so a model *narrating* its
# reasoning ("I established which company and which fiscal year") is not scored
# as having asked for information it already had. The system prompt hands the
# model that exact vocabulary, so an unanchored match measures verbosity.
_CLARIFICATION = re.compile(
    r"(?:^|[.?!]\s+|\n)\s*(?:"
    r"which company|what (?:is|was) the ticker|please provide|could you (?:provide|specify)|"
    r"i need (?:a|the) (?:stock )?ticker|what ticker|which fiscal year"
    r")",
    re.I,
)

# LangChain returns this verbatim when an agent exhausts max_iterations. It is
# not an answer, but it arrives in the same field as one -- and the agent will
# have called its tools on the way, so without this check a model that thrashed
# for fifteen turns and converged on nothing scores a full tool-path success on
# the headline ranking metric.
_NO_ANSWER_SENTINELS = (
    "agent stopped due to iteration limit",
    "agent stopped due to max iterations",
)

# handle_parsing_errors=True surfaces malformed tool calls as a pseudo-tool of
# this name. Counting it as a real call would hide exactly the failure mode that
# most distinguishes these providers: reliability of tool-call formatting.
_PARSE_ERROR_TOOL = "_Exception"


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
    def gave_up(self) -> bool:
        """The agent hit its iteration limit instead of producing an answer."""
        lowered = self.answer.lower()
        return any(sentinel in lowered for sentinel in _NO_ANSWER_SENTINELS)

    @property
    def malformed_tool_calls(self) -> int:
        """Tool calls the model emitted in a shape the framework could not parse."""
        return sum(1 for t in self.tools_called if t == _PARSE_ERROR_TOOL)

    @property
    def tool_path_ok(self) -> bool:
        """Every expected tool was called, and the run produced an answer.

        The second half matters: calling both tools and then looping until the
        iteration limit is a failure, not a success, however good the trace
        looks.
        """
        if self.error is not None or self.gave_up:
            return False
        return all(t in self.tools_called for t in self.case.expected_tools)

    @property
    def used_calculator_when_required(self) -> bool:
        """Arithmetic must be computed, not recalled."""
        if self.error is not None or self.gave_up:
            return False
        if "calculator" not in self.case.expected_tools:
            return True
        return "calculator" in self.tools_called

    @property
    def cited_a_figure(self) -> bool:
        """A weak but LLM-free signal that the answer is grounded in numbers.

        Requires something shaped like a reported financial quantity -- a
        multi-digit or decimal number, a percentage, or a currency amount --
        rather than any digit at all. Matching a bare digit made this read
        95-100% for every backend, including ones that said they found nothing,
        leaving it with no discriminating signal while acting as a ranking
        tie-breaker.
        """
        if self.error is not None or self.gave_up:
            return False
        return bool(re.search(r"\d[\d,]*\.?\d*\s*%|[$£€]\s*\d|\d[\d,]{2,}|\d+\.\d+", self.answer))


@dataclass
class AgentReport:
    results: list[AgentCaseResult] = field(default_factory=list)
    # Failures across every attempt that wrote to this run's checkpoint, and
    # how many attempts that was. Reported because `errors` alone counts only
    # the current attempt: resume a quota-starved run until it completes and it
    # would otherwise publish a clean sheet.
    historic_errors: int = 0
    attempts: int = 1

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
            "gave_up": round(self._rate(lambda r: r.gave_up), 4),
            "malformed_tool_calls": sum(r.malformed_tool_calls for r in self.results),
            "errors": sum(1 for r in self.results if r.error),
            "errors_all_attempts": max(
                self.historic_errors, sum(1 for r in self.results if r.error)
            ),
            "attempts": self.attempts,
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
    checkpoint_path=None,
) -> AgentReport:
    """Run the agent over the question set.

    Needs whichever provider key settings.llm_backend requires. With
    ``checkpoint_path`` set, each successful case is persisted as it completes
    and a rerun skips it -- which is what lets a run larger than a free tier's
    daily quota finish across days instead of restarting from zero.
    """
    from .checkpoint import Checkpoint

    settings = settings or get_settings()
    cases = cases if cases is not None else load_agent_cases()
    if limit:
        cases = cases[:limit]

    checkpoint = Checkpoint(checkpoint_path)
    pending = [c for c in cases if checkpoint.completed(c.id) is None]
    report = AgentReport(historic_errors=checkpoint.historic_errors, attempts=checkpoint.attempts)

    if agent is None and pending:
        from ..agent import build_agent
        from ..ingest.index import open_store

        agent = build_agent(store=store or open_store(settings), settings=settings)

    for case in cases:
        done = checkpoint.completed(case.id)
        if done is not None:
            report.results.append(
                AgentCaseResult(
                    case=case,
                    answer=done.get("answer", ""),
                    tools_called=list(done.get("tools_called", [])),
                )
            )
            continue

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
        # A failed case is recorded but not marked done: it retries on the next
        # attempt, while its failure still counts toward errors_all_attempts.
        if result.error is None:
            checkpoint.record(
                case.id, {"answer": result.answer, "tools_called": result.tools_called}
            )
        else:
            checkpoint.record_failure(case.id, result.error)
            report.historic_errors = checkpoint.historic_errors
        log.info("%-28s tools=%s", case.id, result.tools_called or "none")
    return report
