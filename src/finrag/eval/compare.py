"""Run one evaluation suite across several backends and rank them.

The comparison is only meaningful if every backend sees identical inputs, so
this deliberately varies exactly one thing -- ``llm_backend`` -- and holds the
corpus, the chunking, the retrieval depth, the case set and the context budget
fixed across every run.

Two properties make the ranking trustworthy:

* The agent suite scores with string and tool-trace checks only, no LLM judge,
  so no model can favour its own family.
* Where a judge *is* involved (the RAGAS suite), FINRAG_JUDGE_BACKEND pins one
  judge across every run. Without that, each backend marks its own homework.

A backend that cannot run at all -- missing key, missing extra, no local server
-- is recorded as unavailable rather than aborting the sweep, because a
half-finished comparison is worth more than none.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings, get_settings

log = logging.getLogger(__name__)

# Metrics worth ranking on, in the order they belong in a table. tool_path
# accuracy leads because a dropped tool call is the failure that matters here:
# the agent answers from memory and invents a figure.
AGENT_COLUMNS = (
    ("tool_path_accuracy", "tool path"),
    ("calculator_compliance", "calc use"),
    ("clarification_requests", "asked back"),
    ("answered_with_figures", "gave figures"),
    ("errors", "errors"),
)
RAGAS_COLUMNS = (
    ("ragas_faithfulness", "faithful"),
    ("ragas_answer_relevancy", "relevancy"),
    ("ragas_context_precision", "ctx precision"),
)

# Ranking keys in priority order, each with the direction that counts as better.
# One metric is not enough: a backend can call both tools correctly on every
# question and still be useless if what it produced was "which company did you
# mean?" -- tool_path_accuracy would read 100% while the agent answered nothing.
# So ties on the headline metric break on whether an answer with actual figures
# came back, then on how often the model asked for information it already had,
# then on outright errors.
AGENT_RANK_KEYS = (
    ("tool_path_accuracy", 1),
    ("answered_with_figures", 1),
    ("clarification_requests", -1),
    ("errors", -1),
)
RAGAS_RANK_KEYS = (
    ("ragas_faithfulness", 1),
    ("ragas_answer_relevancy", 1),
    ("ragas_context_precision", 1),
)


@dataclass
class BackendResult:
    backend: str
    model: str = ""
    metrics: dict[str, float | str] = field(default_factory=dict)
    seconds: float = 0.0
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.error is None

    def score(self, key: str) -> float | None:
        """The metric, or None if it is absent or not a real number.

        RAGAS emits NaN when its judge calls fail -- exactly the free-tier 429
        case this project is built around. NaN compares False against
        everything, so left in place it would sit wherever the sort happened to
        put it and could crown a backend that produced no valid score at all.
        It is also not valid JSON. Treated as missing on both counts.
        """
        value = self.metrics.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        return None if math.isnan(value) or math.isinf(value) else float(value)


@dataclass
class ComparisonReport:
    suite: str
    results: list[BackendResult] = field(default_factory=list)
    rank_key: str = "tool_path_accuracy"
    # Held constant across every run; recorded so the table can state them.
    fixed: dict[str, str] = field(default_factory=dict)

    @property
    def columns(self):
        return AGENT_COLUMNS if self.suite == "agent" else RAGAS_COLUMNS

    @property
    def rank_keys(self):
        return AGENT_RANK_KEYS if self.suite == "agent" else RAGAS_RANK_KEYS

    def sort_key(self, result: BackendResult) -> tuple[float, ...]:
        """Ordered key: headline metric first, then the tie-breakers."""
        return tuple(direction * (result.score(key) or 0.0) for key, direction in self.rank_keys)

    def ranked(self) -> list[BackendResult]:
        runnable = [r for r in self.results if r.available and r.score(self.rank_key) is not None]
        runnable.sort(key=self.sort_key, reverse=True)
        return runnable + [r for r in self.results if r not in runnable]

    def winner(self) -> BackendResult | None:
        ranked = self.ranked()
        return ranked[0] if ranked and ranked[0].available else None

    def as_markdown(self) -> str:
        """A table ready to paste into the README."""
        cols = self.columns
        head = "| backend | model | " + " | ".join(label for _, label in cols) + " | wall clock |"
        rule = "|---" * (len(cols) + 3) + "|"
        lines = [head, rule]

        for r in self.ranked():
            if not r.available:
                lines.append(
                    f"| `{r.backend}` | {r.model or '—'} | "
                    + " | ".join(["—"] * len(cols))
                    + f" | unavailable: {r.error[:60]} |"
                )
                continue
            cells = []
            for key, _ in cols:
                raw = r.metrics.get(key)
                value = r.score(key)
                if value is None:
                    # Distinguishes "the judge failed on this metric" from
                    # "this metric was never collected".
                    cells.append("n/a" if isinstance(raw, float) else "—")
                elif isinstance(raw, int) and not isinstance(raw, bool):
                    cells.append(str(raw))
                else:
                    cells.append(f"{value:.0%}" if 0 <= value <= 1 else f"{value:.3f}")
            lines.append(
                f"| `{r.backend}` | {r.model} | " + " | ".join(cells) + f" | {r.seconds:.0f}s |"
            )

        if self.fixed:
            lines += [
                "",
                "Held constant across every run: "
                + ", ".join(f"`{k}={v}`" for k, v in sorted(self.fixed.items()))
                + ".",
            ]
        # Stated because the number invites the wrong reading: most of it is
        # this project's own client-side rate limiter, which paces each backend
        # to its free tier's RPM -- github at 8/min sleeps three times longer
        # than groq at 25/min for identical work. It is not provider latency.
        lines += [
            "",
            "`wall clock` includes client-side rate limiting and cache hits, so it measures "
            "this harness under free-tier pacing rather than provider speed.",
        ]
        if self.suite == "ragas" and self.fixed.get("judge", "").startswith("SELF"):
            lines += [
                "",
                "**Caveat: no judge was pinned, so each backend scored its own answers.** "
                "These numbers partly measure judge self-preference. Set `FINRAG_JUDGE_BACKEND` "
                "and re-run before drawing any conclusion from the ranking.",
            ]
        return "\n".join(lines)

    # Full settings snapshot, from the same config_params() a single eval uses.
    # Without it the comparison artifact -- the durable record behind a
    # published table -- carried less provenance than an individual run.
    config: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "suite": self.suite,
            "rank_key": self.rank_key,
            "fixed": self.fixed,
            "config": self.config,
            "results": [
                {
                    "backend": r.backend,
                    "model": r.model,
                    # NaN is not valid JSON; a file that cannot be parsed is
                    # worse than a metric recorded as null.
                    "metrics": {
                        k: (v if r.score(k) is not None or not isinstance(v, float) else None)
                        for k, v in r.metrics.items()
                    },
                    "seconds": round(r.seconds, 1),
                    "error": r.error,
                }
                for r in self.ranked()
            ],
        }


def _checkpoint_for(suite: str, backend: str, model: str, dataset: str = "default") -> Path:
    """Checkpoint file for one (suite, backend, model, case set).

    The case set has to be part of the key. `--dataset smoke` asks three
    fixture questions with ids like `smoke-calc-aapl-2023`; the default suite
    asks twenty real ones. Keying on backend and model alone let a resumed run
    of one silently satisfy cases of the other whenever ids happened to match,
    and left a stale file to be mistaken for a completed sweep.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", f"{backend}-{model}-{dataset}".lower()).strip("-")
    return Path("results") / f"{suite}-{slug}-cases.jsonl"


def shared_context_budget(backends: list[str], settings: Settings) -> int:
    """The context budget every backend in the sweep must share.

    Settings.max_context_tokens is derived from llm_backend when set to "auto"
    -- 6000 tokens on cerebras and github, unlimited elsewhere. Since a sweep
    varies llm_backend, leaving it on "auto" silently varies the *context* too:
    cerebras would answer from eight chunks while groq answered from twenty,
    under a table announcing "held constant". Whatever such a comparison
    measured, it would not be the models.

    Resolving to the most restrictive budget in the sweep fixes both halves of
    the problem at once: every backend sees identical context, and no request
    exceeds the tightest provider ceiling and fails with a 400.
    """
    budgets = []
    for backend in backends:
        value = replace(settings, llm_backend=backend).max_context_tokens
        if value > 0:
            budgets.append(value)
    return min(budgets) if budgets else 0


def compare_backends(
    backends: list[str],
    suite: str = "agent",
    cases=None,
    store=None,
    settings: Settings | None = None,
    limit: int | None = None,
    resume: bool = False,
    dataset: str = "default",
    corpus: str = "real",
) -> ComparisonReport:
    """Run ``suite`` once per backend and collect the results.

    Every backend gets its own checkpoint file keyed by backend and resolved
    model, so a sweep interrupted by one provider's daily quota can be resumed
    without any chance of one backend inheriting another's answers.
    """
    from ..llm import default_model_for

    settings = settings or get_settings()

    # Pin everything that would otherwise vary as a side effect of switching
    # llm_backend. Without this the sweep changes several things at once and
    # the table's "held constant" footer is simply false.
    budget = shared_context_budget(backends, settings)

    report = ComparisonReport(
        suite=suite,
        rank_key="tool_path_accuracy" if suite == "agent" else "ragas_faithfulness",
        fixed={
            "retrieval_k": str(settings.retrieval_k),
            "chunk_strategy": settings.chunk_strategy,
            "chunk_size": str(settings.chunk_size),
            "embeddings": settings.embedding_backend,
            "context_tokens": str(budget) if budget else "unlimited",
            # Both matter to a reader and neither was recorded before. A sweep
            # over three toy fixture filings with three questions produced a
            # table indistinguishable from one over fifty real 10-Ks.
            "dataset": dataset,
            "corpus": corpus,
            "cases": str(limit) if limit else "all",
            "run": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        },
    )
    if suite == "ragas":
        report.fixed["judge"] = settings.judge_backend or "SELF (not pinned)"

    from .tracking import config_params

    report.config = config_params(settings)

    for backend in backends:
        run_settings = replace(
            settings,
            llm_backend=backend,
            # A model id from one provider is meaningless to another, so each
            # backend runs its own default rather than inheriting one.
            chat_model="",
            # Resolved once above, so every row sees the same context.
            max_context_tokens_raw=str(budget),
            # Fallbacks would let a row labelled "groq" contain answers a
            # different provider generated after a 429 -- the attribution the
            # whole table rests on. Off for the duration of a comparison.
            llm_fallbacks="",
            # A leftover base-URL override would silently collapse every
            # OpenAI-compatible preset onto one endpoint while the table went
            # on naming them separately.
            openai_base_url="",
        )
        model = default_model_for(backend)
        result = BackendResult(backend=backend, model=model)
        checkpoint = _checkpoint_for(suite, backend, model, dataset)
        if not resume and checkpoint.exists():
            checkpoint.unlink()

        log.info("=== %s (%s) ===", backend, model)
        started = time.monotonic()
        try:
            if suite == "agent":
                from .agent_eval import evaluate_agent

                sub = evaluate_agent(
                    cases=cases,
                    store=store,
                    settings=run_settings,
                    limit=limit,
                    checkpoint_path=checkpoint,
                )
            else:
                from .ragas_eval import evaluate_ragas

                sub = evaluate_ragas(
                    cases=cases,
                    store=store,
                    settings=run_settings,
                    limit=limit,
                    checkpoint_path=checkpoint,
                )
            result.metrics = sub.as_metrics()
        except Exception as exc:  # noqa: BLE001 - one dead backend must not end the sweep
            result.error = f"{type(exc).__name__}: {exc}"
            log.error("%s unavailable: %s", backend, result.error)
        result.seconds = time.monotonic() - started
        report.results.append(result)

    return report
