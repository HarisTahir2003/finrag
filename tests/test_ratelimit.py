"""Spending limits for a shared key.

These guard a real, exhaustible resource: Groq's free tier is 1,000 requests a
day across everyone who finds the public URL, and one agent answer costs
several. The failure they prevent is silent -- the demo simply stops working
for everybody, hours before the owner notices.
"""

from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from finrag.config import Settings
from finrag.presentation import limit_message
from finrag.ratelimit import Decision, RateLimiter, spend_check


@pytest.fixture(autouse=True)
def _fresh_limiter(monkeypatch):
    """Counters are module-level and shared; tests must not inherit each other's."""
    import finrag.ratelimit as rl

    monkeypatch.setattr(rl, "limiter", RateLimiter())


def _settings(**over) -> Settings:
    base = Settings(
        questions_per_session=3,
        session_window_seconds=3600,
        questions_per_day=5,
    )
    return replace(base, **over) if over else base


# ------------------------------------------------------------- the window


def test_it_allows_up_to_the_limit_then_refuses():
    limiter = RateLimiter()
    verdicts = [limiter.check("s", "a", limit=3, seconds=60) for _ in range(4)]

    assert [v.allowed for v in verdicts] == [True, True, True, False]
    assert verdicts[-1].retry_after > 0, "a refusal must say when to come back"


def test_separate_keys_do_not_share_a_budget():
    """One visitor exhausting theirs must not refuse the next arrival."""
    limiter = RateLimiter()
    for _ in range(3):
        limiter.check("session", "visitor-a", limit=3, seconds=60)

    assert limiter.check("session", "visitor-a", 3, 60).allowed is False
    assert limiter.check("session", "visitor-b", 3, 60).allowed is True


def test_the_window_rolls_over():
    limiter = RateLimiter()
    for _ in range(3):
        limiter.check("s", "a", limit=3, seconds=60)
    assert limiter.check("s", "a", 3, 60).allowed is False

    # Reach into the window rather than sleeping 60s in a unit test.
    window = limiter._windows["s:a"]
    window.started -= 61

    assert limiter.check("s", "a", 3, 60).allowed is True


def test_a_limit_of_zero_disables_rather_than_blocks():
    """A misconfigured variable should cost the limit, not the service.

    FINRAG_QUESTIONS_PER_DAY="0" read as "zero allowed" would take the whole
    demo down, which is the opposite of what someone setting it to zero means.
    """
    limiter = RateLimiter()
    assert all(limiter.check("s", "a", limit=0, seconds=60).allowed for _ in range(50))


def test_counting_is_safe_across_threads():
    """Streamlit runs each browser session in its own thread, one process.

    Without the lock the counter loses increments and the limit leaks.
    """
    limiter = RateLimiter()
    allowed: list[bool] = []
    lock = threading.Lock()

    def ask():
        verdict = limiter.check("global", "all", limit=50, seconds=60)
        with lock:
            allowed.append(verdict.allowed)

    threads = [threading.Thread(target=ask) for _ in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(allowed) == 50, f"expected exactly 50 through, got {sum(allowed)}"


# -------------------------------------------------------------- the policy


def test_a_visitor_with_their_own_key_is_not_rationed():
    """Rationing someone's own quota would be rationing their money.

    It is also what makes the limits tolerable: when the shared budget is gone
    there is still a way to use the demo.
    """
    settings = _settings()
    for _ in range(100):
        assert spend_check(settings, "visitor", shared_key=False).allowed


def test_the_session_limit_bites_before_the_daily_one():
    """Its message names the smaller, sooner-resetting window."""
    settings = _settings()
    for _ in range(3):
        assert spend_check(settings, "visitor").allowed

    verdict = spend_check(settings, "visitor")
    assert verdict.allowed is False
    assert verdict.scope == "session"


def test_the_daily_limit_catches_what_the_session_limit_cannot():
    """Many browsers, or one person who reloads to get a new session."""
    settings = _settings()
    outcomes = [spend_check(settings, f"visitor-{i}").allowed for i in range(8)]

    assert outcomes[:5] == [True] * 5
    assert outcomes[5:] == [False] * 3, "the shared budget must bound the total"
    assert spend_check(settings, "visitor-99").scope == "global"


# ------------------------------------------------------------- the message


def test_a_refusal_always_offers_a_way_forward():
    """A refusal with no next step reads as a broken app."""
    settings = _settings()
    for scope in ("session", "global"):
        text = limit_message(Decision(False, scope=scope, retry_after=900, limit=3), settings)
        assert "own" in text.lower() and "key" in text.lower()
        assert "sidebar" in text.lower()


def test_the_wait_is_phrased_the_way_people_speak():
    assert Decision(False, retry_after=45).retry_after_human == "45 seconds"
    assert Decision(False, retry_after=900).retry_after_human == "15 minutes"
    assert Decision(False, retry_after=7200).retry_after_human == "2 hours"


def test_expired_windows_are_reaped():
    """Every session mints a new key, so without pruning the dict grows one dead
    entry per visitor for the process's life."""
    limiter = RateLimiter()
    # Many short-lived session buckets.
    for i in range(50):
        limiter.check("session", f"visitor-{i}", limit=5, seconds=60)
    assert len(limiter._windows) == 50

    # Age them all past their window, then make one more call.
    for window in limiter._windows.values():
        window.started -= 61
    limiter.check("session", "a-fresh-visitor", limit=5, seconds=60)

    # The 50 expired buckets are gone; only the live one remains.
    assert len(limiter._windows) == 1
    assert "session:a-fresh-visitor" in limiter._windows


def test_reaping_does_not_disturb_a_live_window():
    """A window still inside its period must keep its count across a reap."""
    limiter = RateLimiter()
    for _ in range(3):
        limiter.check("session", "steady", limit=5, seconds=60)
    # A short-lived unrelated bucket that will expire.
    limiter.check("session", "transient", limit=5, seconds=1)
    limiter._windows["session:transient"].started -= 2

    verdict = limiter.check("session", "steady", limit=5, seconds=60)
    assert verdict.used == 4, "the live window kept its running count"
    assert "session:transient" not in limiter._windows, "the expired one was reaped"
