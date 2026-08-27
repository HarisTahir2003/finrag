"""Per-case checkpointing for evaluation runs.

Free tiers meter by the day as well as by the minute. A 20-case evaluation that
dies at case 14 when a daily quota runs out should cost 6 calls tomorrow, not
20 -- so each case's result is appended to a JSONL file the moment it
completes, and a resumed run skips the cases already on disk.

Failures are recorded too, but never treated as complete: a case that errored
-- which on a free tier usually means the quota itself -- must run again on the
next attempt. They are written down so the *count* survives across attempts.
Without that, `errors` reported only the current attempt while the successes
accumulated, so a backend retried until its quota cleared published `errors 0`
and a backend run once published `errors 6`. In practice you retry the flaky
free tiers and run the reliable ones once, which made the column read
backwards.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class Checkpoint:
    """Append-only JSONL store of per-case results, keyed by case id."""

    def __init__(self, path: Path | None):
        self.path = path
        self._done: dict[str, dict[str, Any]] = {}
        self._failures: list[dict[str, Any]] = []
        self._attempts = 1
        if path and path.exists():
            self._attempts = 2  # this file already holds at least one prior run
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("skipping corrupt checkpoint line in %s", path)
                    continue
                if "id" not in record:
                    continue
                if record.get("error"):
                    self._failures.append(record)
                else:
                    # Later lines win, so a re-recorded case supersedes itself,
                    # and a success after a failure clears it.
                    self._done[record["id"]] = record
            if self._done:
                log.info("resuming: %d case(s) already complete in %s", len(self._done), path)

    def completed(self, case_id: str) -> dict[str, Any] | None:
        """The stored result, or None if the case still needs to run.

        A recorded failure returns None so it is retried, while still counting
        toward historic_errors.
        """
        record = self._done.get(case_id)
        if record is None or record.get("error"):
            return None
        return record

    def record(self, case_id: str, payload: dict[str, Any]) -> None:
        record = {"id": case_id, **payload}
        self._done[case_id] = record
        self._append(record)

    def record_failure(self, case_id: str, error: str) -> None:
        """Note a failed attempt without marking the case done."""
        record = {"id": case_id, "error": error}
        # Kept out of _done so a later success can overwrite it cleanly.
        self._failures.append(record)
        self._append(record)

    @property
    def historic_errors(self) -> int:
        """Failed attempts across every run that wrote to this checkpoint."""
        return len(self._failures)

    @property
    def attempts(self) -> int:
        """How many times this checkpoint has been opened for a run."""
        return self._attempts

    def _append(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._done)
