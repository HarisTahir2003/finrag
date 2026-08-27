"""Quality thresholds for CI.

The gate runs the retrieval evaluation, which needs no LLM and no API key, so a
pull request from a fork can be checked for free and deterministically. A gate
that required a paid judge model would be a gate nobody could run.

Thresholds are deliberately a little below current measured performance: the
purpose is to catch regressions, not to fail on noise.
"""

from __future__ import annotations

from dataclasses import dataclass

from .retrieval_eval import RetrievalReport


@dataclass(frozen=True)
class Thresholds:
    min_hit_rate: float = 0.80
    min_mrr: float = 0.50
    # Anything below 1.0 means chunks from the wrong company or year are being
    # returned, which is exactly the bug this project started with.
    min_filter_accuracy: float = 1.0
    max_empty_retrievals: int = 0


@dataclass
class GateResult:
    passed: bool
    failures: list[str]
    metrics: dict[str, float]

    def format(self) -> str:
        head = "QUALITY GATE PASSED" if self.passed else "QUALITY GATE FAILED"
        lines = [head, ""]
        lines += [f"  {k:22} {v}" for k, v in self.metrics.items()]
        if self.failures:
            lines += ["", "  Breaches:"] + [f"    - {f}" for f in self.failures]
        return "\n".join(lines)


def check(report: RetrievalReport, thresholds: Thresholds | None = None) -> GateResult:
    t = thresholds or Thresholds()
    metrics = report.as_metrics()
    failures: list[str] = []

    if report.hit_rate < t.min_hit_rate:
        failures.append(f"hit_rate {report.hit_rate:.3f} is below {t.min_hit_rate}")
    if report.mrr < t.min_mrr:
        failures.append(f"mrr {report.mrr:.3f} is below {t.min_mrr}")
    if report.filter_accuracy < t.min_filter_accuracy:
        failures.append(
            f"filter_accuracy {report.filter_accuracy:.3f} is below {t.min_filter_accuracy} "
            "-- chunks from the wrong company or fiscal year are being retrieved"
        )
    if report.empty_retrievals > t.max_empty_retrievals:
        failures.append(
            f"{report.empty_retrievals} queries retrieved nothing (limit {t.max_empty_retrievals})"
        )

    return GateResult(passed=not failures, failures=failures, metrics=metrics)
