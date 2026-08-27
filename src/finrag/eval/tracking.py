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
        "embedding_backend",
        "local_embedding_model",
        "google_embedding_model",
        "chat_model",
        "chunk_strategy",
        "chunk_size",
        "chunk_overlap",
        "retrieval_k",
        "collection_name",
    )
    return {k: str(raw[k]) for k in keep if k in raw}


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

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = results_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}-{stamp}.json"
    path.write_text(
        json.dumps(
            {"run": name, "timestamp": stamp, "params": params, "metrics": captured}, indent=2
        ),
        encoding="utf-8",
    )
    log.info("results written to %s", path)
