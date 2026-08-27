"""Experiment tracking, if MLflow is installed.

Every evaluation run records the configuration that produced it -- embedding
backend, chunk strategy, chunk size, retrieval depth, model. Without that, a
metric is an anecdote: the original 0.18 -> 0.74 faithfulness improvement cannot
be reproduced from the repository because nothing recorded what changed between
the two runs.

MLflow is optional. When it is missing, this degrades to a no-op that still
prints the run, so evaluation never fails for want of a tracking server.
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _mlflow():
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


def config_params(settings: Any) -> dict[str, Any]:
    """The settings worth recording alongside a metric."""
    raw = asdict(settings) if is_dataclass(settings) else dict(settings)
    keep = (
        "llm_backend",
        "embedding_backend",
        "local_embedding_model",
        "google_embedding_model",
        "chat_model",
        "chunk_strategy",
        "chunk_size",
        "chunk_overlap",
        "retrieval_k",
        "max_output_tokens",
        "collection_name",
        "requests_per_minute",
        "llm_fallbacks",
    )
    params = {k: str(raw[k]) for k in keep if k in raw}

    # chat_model is usually "" -- it means "use whatever this backend defaults
    # to". Recording only the blank leaves a result file that cannot say which
    # model produced it, which would make a cross-backend comparison table
    # unreadable. Resolve and record the actual model name.
    backend = raw.get("llm_backend", "")
    if backend:
        try:
            from ..llm import default_model_for

            params["resolved_model"] = raw.get("chat_model") or default_model_for(backend)
        except Exception:  # noqa: BLE001 - tracking must never break a run
            pass

    # Also resolved, because "auto" means different budgets on different
    # backends -- 6000 tokens on cerebras/github, unlimited elsewhere. Two runs
    # recording max_context_tokens_raw="auto" are not comparable.
    if hasattr(settings, "max_context_tokens"):
        params["max_context_tokens"] = str(settings.max_context_tokens)

    return params


@contextmanager
def track_run(name: str, params: dict[str, Any], results_dir: Path | None = None):
    """Record one evaluation run.

    Yields a callable that accepts a metrics dict. Metrics are always written to
    a timestamped JSON file so there is a durable record with or without MLflow.
    """
    mlflow = _mlflow()
    captured: dict[str, Any] = {}

    def record(metrics: dict[str, Any]) -> None:
        captured.update(metrics)

    if mlflow is None:
        log.info("mlflow not installed; recording to disk only (pip install mlflow)")
        yield record
    else:
        mlflow.set_experiment("finrag")
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            yield record
            numeric = {k: v for k, v in captured.items() if isinstance(v, (int, float))}
            if numeric:
                mlflow.log_metrics(numeric)

    # Second-resolution stamps collide: a fully-checkpointed --resume run makes
    # no LLM calls at all (the agent is never even constructed) and finishes in
    # milliseconds, so regenerating several backends from existing checkpoints
    # lands them in the same second. Each would write the same filename and only
    # the last would survive -- with no error and no gap in the sequence to show
    # the others were lost. Microseconds plus the run identity, and never
    # overwrite.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    backend = str(params.get("llm_backend", "")).strip()
    model = str(params.get("resolved_model", "")).strip()
    slug = re.sub(r"[^a-z0-9]+", "-", f"{backend}-{model}".lower()).strip("-")
    stem = f"{name}-{slug}-{stamp}" if slug else f"{name}-{stamp}"

    out_dir = results_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.json"
    suffix = 2
    while path.exists():
        path = out_dir / f"{stem}-{suffix}.json"
        suffix += 1
    path.write_text(
        json.dumps(
            {"run": name, "timestamp": stamp, "params": params, "metrics": captured}, indent=2
        ),
        encoding="utf-8",
    )
    log.info("results written to %s", path)
