"""The committed Streamlit config, and why it has to exist.

Streamlit Community Cloud runs `streamlit run app.py` for you and offers no way
to add command-line arguments, so a hosted app is configured by a
.streamlit/config.toml committed to the repository. The Docker images pass the
same settings as flags. Two mechanisms for one intent drift unless something
checks, and the drift is invisible until a deployed log fills with tracebacks.
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 ships no stdlib TOML parser
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".streamlit" / "config.toml"


def _config() -> dict:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def test_the_config_is_committed():
    """Without this file the hosted app has no way to be configured at all."""
    assert CONFIG.exists(), "Community Cloud reads .streamlit/config.toml; nothing else"


def test_the_file_watcher_is_off():
    """Streamlit's watcher walks every module in sys.modules for a source file.

    That touches transformers' lazy __getattr__, which imports image processors
    that require torchvision. This project installs CPU-only torch and no
    torchvision, so every walk raises ModuleNotFoundError and the deployed log
    fills with tracebacks from a feature nothing here uses. Reproduced against
    the installed packages: walking sys.modules with the embedding model loaded
    raises on transformers/models/zoedepth/image_processing_zoedepth.py.

    "none" is the value that matters: streamlit/runtime/app_session.py guards
    the watcher with `if config.get_option("server.fileWatcherType") != "none"`,
    so anything else still constructs it.
    """
    assert _config()["server"]["fileWatcherType"] == "none"


def test_the_containers_pass_the_same_setting():
    """One intent, two mechanisms. They must not drift."""
    for name in ("Dockerfile.space", "compose.yaml"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "--server.fileWatcherType=none" in text, (
            f"{name} no longer disables the file watcher, but "
            ".streamlit/config.toml still does -- one of them is wrong"
        )


def test_usage_stats_are_off_everywhere():
    """A public demo should not report its visitors' usage to a third party."""
    assert _config()["browser"]["gatherUsageStats"] is False
    for name in ("Dockerfile.space", "compose.yaml"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "--browser.gatherUsageStats=false" in text, name


def test_streamlit_actually_honours_the_file():
    """The mechanism, not just the contents.

    Streamlit resolves config from the working directory, so this reads the
    real file through Streamlit's own loader rather than trusting the TOML.
    """
    import streamlit.config as st_config

    st_config.get_config_options()
    assert st_config.get_option("server.fileWatcherType") == "none"
    assert str(CONFIG) in str(st_config.get_where_defined("server.fileWatcherType"))
