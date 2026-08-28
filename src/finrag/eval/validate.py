"""Checks that the evaluation cases assert things the corpus can actually support.

Every defect found in these datasets so far shared one shape: an expectation the
filings contradict, which produced a plausible wrong number instead of an error.

- ``amzn-2022-fulfillment`` expected the word "regionalization". It appears in no
  Amazon 10-K of any year -- it was shareholder-letter language -- so the case
  could not pass at any retrieval depth, and quietly held hit_rate at 8/9.
- ``mixed-meta-2023-15`` asks about the "Year of Efficiency" and the "Family of
  Apps operating margin". Neither phrase is in META's FY2023 filing.
- ``mixed-msft-2023-16`` demands the calculator for a figure the filing states
  outright: "Microsoft Cloud revenue increased 22% to $111.6 billion". The agent
  read it instead of recomputing, which is correct, and was marked failed.

Three defects in twenty-nine hand-authored cases. None raised an error; each one
depressed a headline metric. They are all mechanically detectable, so they are
detected here rather than by whoever next reads the numbers carefully.

Two tiers, because they need different things:

``structural``
    Needs nothing but the YAML. Runs in CI, on a fresh clone, offline.

``corpus``
    Needs the downloaded and indexed filings, so it runs locally. This is the
    tier that catches the defects above.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..config import DEFAULT_TICKERS, Settings, get_settings
from .schema import load_agent_cases, load_retrieval_cases, reference_figures

log = logging.getLogger(__name__)

# A 10-K exists for a year in roughly this range; anything outside is a typo
# rather than a filing we simply have not downloaded.
_PLAUSIBLE_YEARS = range(1994, 2100)


@dataclass
class Finding:
    """One problem with one case. ``fatal`` distinguishes broken from suspect."""

    case_id: str
    dataset: str
    message: str
    fatal: bool = True

    def format(self) -> str:
        return f"{'FAIL' if self.fatal else 'warn'}  {self.dataset}/{self.case_id}: {self.message}"


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)
    checked: int = 0

    def add(self, case_id: str, dataset: str, message: str, fatal: bool = True) -> None:
        self.findings.append(Finding(case_id, dataset, message, fatal))

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.fatal]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if not f.fatal]

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_metrics(self) -> dict[str, float]:
        return {
            "cases_checked": self.checked,
            "failures": len(self.failures),
            "warnings": len(self.warnings),
        }

    def format_report(self) -> str:
        if not self.findings:
            return f"{self.checked} cases checked, nothing to report."
        lines = [f.format() for f in self.findings]
        lines.append("")
        lines.append(
            f"{self.checked} cases checked: "
            f"{len(self.failures)} failing, {len(self.warnings)} suspect."
        )
        return "\n".join(lines)


_LEADING_PERCENTAGE = re.compile(r"^\s*(-?\d+(?:\.\d+)?%)")


def _leading_percentage(reference: str) -> str | None:
    """The percentage a reference answer leads with, if it leads with one.

    References are written result-first -- "22% growth to about $111B",
    "29.49% (2023 $106,618M vs 2022 $82,338M)" -- so the head of the string is
    the answer and the rest is its working.
    """
    match = _LEADING_PERCENTAGE.match(reference or "")
    return match.group(1) if match else None


def _known_tool_names(settings: Settings) -> set[str]:
    """The tools the agent actually registers.

    An `expected_tools` entry naming something `build_tools` does not create can
    never be satisfied, so the case fails on every run for a reason no metric
    explains.
    """
    from ..agent import build_tools

    return {t.name for t in build_tools(settings=settings)}


def validate_structure(settings: Settings | None = None) -> ValidationReport:
    """Checks that need only the YAML. Safe for CI on a fresh clone."""
    settings = settings or get_settings()
    report = ValidationReport()

    try:
        tool_names = _known_tool_names(settings)
    except Exception as exc:  # noqa: BLE001 - missing extras must not fail validation
        log.debug("could not enumerate tools (%s); skipping the tool-name check", exc)
        tool_names = set()

    tickers = {t.upper() for t in DEFAULT_TICKERS}
    seen: dict[str, str] = {}

    for dataset, cases in (
        ("retrieval", load_retrieval_cases()),
        ("agent", load_agent_cases()),
    ):
        for case in cases:
            report.checked += 1

            if case.id in seen:
                report.add(case.id, dataset, f"duplicate id, already used in {seen[case.id]}")
            seen[case.id] = dataset

            if case.ticker.upper() not in tickers:
                report.add(
                    case.id,
                    dataset,
                    f"ticker {case.ticker!r} is not in DEFAULT_TICKERS, so it is never downloaded",
                )
            if int(case.fiscal_year) not in _PLAUSIBLE_YEARS:
                report.add(case.id, dataset, f"implausible fiscal_year {case.fiscal_year}")

            if dataset == "retrieval":
                if not case.query.strip():
                    report.add(case.id, dataset, "empty query")
                if not case.expect_any or not any(e.strip() for e in case.expect_any):
                    report.add(
                        case.id,
                        dataset,
                        "no expect_any, so the case can never fail and measures nothing",
                    )
            else:
                if not case.question.strip():
                    report.add(case.id, dataset, "empty question")
                if not case.reference_answer.strip():
                    report.add(case.id, dataset, "no reference_answer")
                unknown = [t for t in case.expected_tools if tool_names and t not in tool_names]
                if unknown:
                    report.add(
                        case.id,
                        dataset,
                        f"expected_tools names tools the agent does not register: {unknown}",
                    )

    return report


def validate_against_corpus(store=None, settings: Settings | None = None) -> ValidationReport:
    """Checks that need the indexed filings. The tier that catches real defects."""
    settings = settings or get_settings()
    report = ValidationReport()

    if store is None:
        from ..ingest.index import open_store

        store = open_store(settings)

    # One fetch per filing, reused across every case touching it.
    cache: dict[tuple[str, int], list[str]] = {}

    def chunks_for(ticker: str, year: int) -> list[str]:
        key = (ticker.upper(), int(year))
        if key not in cache:
            where = {"$and": [{"ticker": key[0]}, {"year": key[1]}]}
            cache[key] = store.get(where=where, include=["documents"])["documents"]
        return cache[key]

    for dataset, cases in (
        ("retrieval", load_retrieval_cases()),
        ("agent", load_agent_cases()),
    ):
        for case in cases:
            report.checked += 1

            documents = chunks_for(case.ticker, case.fiscal_year)
            if not documents:
                report.add(
                    case.id,
                    dataset,
                    f"{case.ticker} FY{case.fiscal_year} is not in the index -- "
                    "download and index it, or the case is untestable",
                )
                continue

            if dataset == "retrieval":
                # expect_any is a disjunction -- evaluate_case returns a hit on
                # the first needle that matches -- so the case is broken only
                # when *none* of them exist. Requiring all of them was this
                # checker's own version of the bug it exists to catch: it
                # flagged amzn-2022-net-loss, whose reference offers "(2,722)"
                # and "2,722" precisely because the sign convention varies, and
                # which passes on the second.
                present = [
                    e for e in case.expect_any if any(e.lower() in d.lower() for d in documents)
                ]
                if not present:
                    report.add(
                        case.id,
                        dataset,
                        f"none of expect_any appears in the filing: {case.expect_any} -- "
                        "the probe cannot pass at any retrieval depth",
                    )
                continue

            # Agent cases: a reference quoting figures the filing never reports
            # is asserting something the corpus cannot support. Figures the
            # answer *computes* are exempt -- an average or a sum legitimately
            # appears nowhere -- which is what derived_figures declares.
            figures = reference_figures(case.reference_answer)
            quoted = figures - set(case.derived_figures)
            if quoted:
                missing = [f for f in quoted if not any(f in d for d in documents)]
                if missing:
                    report.add(
                        case.id,
                        dataset,
                        f"reference quotes figures absent from the filing: {missing} "
                        "(add them to derived_figures if the answer computes them)",
                    )

            # The MSFT defect: the filing states the *answer* outright -- "Microsoft
            # Cloud revenue increased 22% to $111.6 billion" -- so reading it is
            # correct and demanding arithmetic marks correct behaviour as failure.
            #
            # The signal is the headline result, not the inputs. An earlier version
            # of this check fired whenever a quoted input was present, which flagged
            # every legitimate ratio case: their inputs are stated, and computing
            # from them is exactly the point. Restricted to a leading percentage,
            # because that is the form a filing reports directly ("increased 22%")
            # while a ratio like 0.229 or 3.18x is essentially never printed.
            if "calculator" in case.expected_tools:
                headline = _leading_percentage(case.reference_answer)
                if headline and any(headline in d for d in documents):
                    report.add(
                        case.id,
                        dataset,
                        f"expects the calculator, but the filing states the answer "
                        f"({headline}) verbatim -- reading it is correct behaviour "
                        "and would score as a failure",
                        fatal=False,
                    )

    return report


def validate_all(store=None, settings: Settings | None = None) -> ValidationReport:
    """Both tiers, running only the corpus one when there is a corpus.

    An absent index does not raise: ``open_store`` happily returns an empty
    collection, so the corpus tier would report every case as "not in the
    index" and fail the build for the opposite of the intended reason. CI never
    downloads filings, which makes this the normal path there rather than an
    edge case.
    """
    settings = settings or get_settings()
    report = validate_structure(settings)

    try:
        if store is None:
            from ..ingest.index import collection_size, open_store

            if collection_size(settings) == 0:
                log.info(
                    "no indexed filings, so only the structural checks ran; "
                    "`finrag download && finrag index` enables the rest"
                )
                return report
            store = open_store(settings)

        corpus = validate_against_corpus(store=store, settings=settings)
    except Exception as exc:  # noqa: BLE001 - structural results still worth having
        log.warning("corpus checks skipped (%s); run `finrag index` first", exc)
        return report

    report.findings.extend(corpus.findings)
    report.checked += corpus.checked
    return report
