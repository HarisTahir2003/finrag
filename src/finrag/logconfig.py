"""One logging setup, shared by every entry point.

The CLI configured logging and nothing else did, so the Streamlit app ran with
whatever the root logger happened to be -- which on a hosted deployment meant
finrag's own INFO lines were dropped and the only diagnostics left were
tracebacks. That is precisely backwards: on a machine you can attach a debugger
to, logs are a convenience; on someone else's host they are the only instrument
you have.
"""

from __future__ import annotations

import logging

_FORMAT = "%(levelname)-7s %(name)s: %(message)s"


def configure_logging(verbose: bool = False) -> None:
    """finrag's own logs at INFO, everybody else's at WARNING.

    basicConfig sets the level on the *root* logger, which every library
    inherits. At INFO that meant a plain `finrag index` opened with forty lines
    of httpx traffic from the embedding model checking its cache, and buried
    its own output underneath. ``verbose`` restores the full firehose.

    Adds a handler only when nothing else has. basicConfig(force=True) was the
    obvious way to make this work under Streamlit, which installs handlers
    before a script runs -- and it is wrong twice: it closes handlers the host
    installed to capture output, and under pytest it removes caplog's, so the
    records a test is asserting on quietly stop arriving. A test caught that.

    Nothing is lost by being gentle. Records propagate to the root handler
    whoever installed it, so the only thing this has to get right is the level
    on finrag's own loggers, which is what was actually missing.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format=_FORMAT,
        )
    logging.getLogger("finrag").setLevel(logging.DEBUG if verbose else logging.INFO)

    # The app logs one line per question at INFO; without a level on this
    # logger it inherits the root's WARNING and the record is dropped.
    logging.getLogger("finrag.app").setLevel(logging.DEBUG if verbose else logging.INFO)
