"""Per-case checkpointing for evaluation runs.

Free tiers meter by the day as well as by the minute. A 20-case evaluation that
dies at case 14 when a daily quota runs out should cost 6 calls tomorrow, not
20 -- so each case's result is appended to a JSONL file the moment it
completes, and a resumed run skips the cases already on disk.

Only *successful* cases are checkpointed. A case that errored -- which on a
free tier usually means the quota itself -- must run again, because recording
the failure would freeze it into the report.
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
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("skipping corrupt checkpoint line in %s", path)
                    continue
                if "id" in record:
                    # Later lines win, so a re-recorded case supersedes itself.
                    self._done[record["id"]] = record
            if self._done:
                log.info("resuming: %d case(s) already complete in %s", len(self._done), path)

    def completed(self, case_id: str) -> dict[str, Any] | None:
        return self._done.get(case_id)

    def record(self, case_id: str, payload: dict[str, Any]) -> None:
        record = {"id": case_id, **payload}
        self._done[case_id] = record
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._done)
