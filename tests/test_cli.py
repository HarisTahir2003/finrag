"""Entry-point behaviour that the library tests cannot see."""

from __future__ import annotations

import pytest

from finrag import cli


def test_main_loads_dotenv_before_dispatch(monkeypatch):
    """The CLI must read .env, and must do it before anything reads a setting.

    Only app.py used to call load_dotenv, so `finrag` ignored the file the
    README tells you to create. The loudest symptom was SEC_CONTACT_EMAIL,
    which raises; the dangerous ones were silent. An unread FINRAG_LLM_BACKEND
    falls back to `anthropic` and an unread FINRAG_RETRIEVAL_K falls back to
    20 -- a paid call carrying a context no free tier will accept, from a
    config file that says groq and 6.
    """
    loaded: list[tuple] = []
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: loaded.append(a))
    monkeypatch.setattr(cli, "find_dotenv", lambda **k: "")

    # No subcommand, so argparse exits immediately -- but only after main() has
    # had its chance to load the environment.
    with pytest.raises(SystemExit):
        cli.main([])

    assert loaded, "main() must load .env before parsing or dispatching"


def test_dotenv_search_is_anchored_to_the_working_directory(monkeypatch):
    """The nearer .env wins.

    The default search walks up from cli.py, which finds the repo's .env under
    an editable install and nothing at all once finrag is installed into
    site-packages. Anchoring to the working directory is what makes `cd
    somewhere && finrag ...` behave the way every other dev CLI does.
    """
    seen: dict = {}
    monkeypatch.setattr(cli, "find_dotenv", lambda **k: seen.update(k) or "")
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)

    with pytest.raises(SystemExit):
        cli.main([])

    assert seen.get("usecwd") is True, "find_dotenv must search from the working directory"
