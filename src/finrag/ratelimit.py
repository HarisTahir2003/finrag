"""Spending limits for a shared key.

A public demo answers questions with the owner's API key. Groq's free tier
allows 1,000 requests a day across everyone who finds the URL, and one agent
answer costs several of them -- a search, a read, sometimes a calculation, each
a round trip. A single visitor holding down a starter pill can therefore end
the day for everybody, without meaning any harm and without seeing that they
did it.

Two limits, because they fail differently:

``session``
    What one browser may spend. Stops the ordinary case: someone curious,
    clicking repeatedly.

``global``
    What the whole deployment may spend before the daily reset. Stops the case
    the session limit cannot see -- many browsers, or one person who reloads.

Both are deliberately crude. A fixed window and an in-memory counter, no store
and no dependency, because the process this runs in is the same process for
every visitor and the numbers only have to be right enough to protect a free
tier. A counter that resets when the container sleeps is not a flaw here: the
quota it guards resets daily anyway, and a sleeping container is spending
nothing.

What this is not is authentication. It slows an accident down; it does not stop
anyone determined, and nothing in a public demo with a shared key could.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

__all__ = ["Decision", "RateLimiter", "Window"]


@dataclass(frozen=True)
class Decision:
    """Whether a request may proceed, and what to say if not."""

    allowed: bool
    scope: str = ""
    retry_after: float = 0.0
    used: int = 0
    limit: int = 0

    @property
    def retry_after_human(self) -> str:
        """A duration a person can read, rounded the way people speak."""
        seconds = max(0.0, self.retry_after)
        if seconds < 90:
            return f"{seconds:.0f} seconds"
        minutes = seconds / 60
        if minutes < 90:
            return f"{minutes:.0f} minutes"
        return f"{minutes / 60:.0f} hours"


@dataclass
class Window:
    """A fixed window: at most ``limit`` events per ``seconds``."""

    limit: int
    seconds: float
    started: float = field(default_factory=time.monotonic)
    used: int = 0

    def take(self, now: float) -> tuple[bool, float]:
        """Consume one event. Returns (allowed, seconds until the window resets)."""
        if now - self.started >= self.seconds:
            self.started = now
            self.used = 0
        remaining = self.seconds - (now - self.started)
        if self.used >= self.limit:
            return False, remaining
        self.used += 1
        return True, remaining


class RateLimiter:
    """Fixed-window counters keyed by scope, safe across Streamlit's threads.

    Streamlit runs each browser session's script in its own thread inside one
    process, so a module-level instance of this is shared by every visitor --
    which is exactly what the global limit needs, and exactly why the counters
    need a lock.

    A limit of 0 or less disables that scope rather than blocking everything,
    so a misconfigured environment variable degrades to "no limit" instead of
    to "no service".
    """

    def __init__(self) -> None:
        self._windows: dict[str, Window] = {}
        self._lock = threading.Lock()

    def check(self, scope: str, key: str, limit: int, seconds: float) -> Decision:
        """Account for one request against ``scope``/``key``."""
        if limit <= 0:
            return Decision(allowed=True, scope=scope)

        now = time.monotonic()
        identity = f"{scope}:{key}"
        with self._lock:
            window = self._windows.get(identity)
            if window is None or window.seconds != seconds or window.limit != limit:
                # Reconfigured between calls -- start the window again rather
                # than reinterpreting a count taken under different rules.
                window = Window(limit=limit, seconds=seconds)
                self._windows[identity] = window
            allowed, remaining = window.take(now)
            used = window.used

        return Decision(
            allowed=allowed,
            scope=scope,
            retry_after=0.0 if allowed else remaining,
            used=used,
            limit=limit,
        )

    def reset(self) -> None:
        """Forget every counter. For tests, and for a deliberate operator reset."""
        with self._lock:
            self._windows.clear()


# The instance the app uses. Module level on purpose: one process serves every
# visitor, so this is how the global limit sees them all.
limiter = RateLimiter()


def spend_check(settings, session_key: str, shared_key: bool = True) -> Decision:
    """Whether this session may ask another question on the shared key.

    ``shared_key=False`` waives both limits: a visitor who supplied their own
    key is spending their own quota, and rationing it would be rationing
    somebody else's money. That is also what makes the limits tolerable -- when
    the shared budget runs out there is still a way to keep using the demo.

    The session limit is checked first so its message names the smaller,
    sooner-resetting window when both would refuse.
    """
    if not shared_key:
        return Decision(allowed=True, scope="own-key")

    session = limiter.check(
        "session",
        session_key,
        settings.questions_per_session,
        float(settings.session_window_seconds),
    )
    if not session.allowed:
        return session

    # 86400s rather than "since midnight": the provider's own daily window
    # resets on its clock, not ours, and guessing the offset would produce a
    # limit that is wrong twice a day. A rolling day is never wrong by more
    # than the window.
    return limiter.check("global", "all", settings.questions_per_day, 86_400.0)
