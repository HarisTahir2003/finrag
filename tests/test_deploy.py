"""The Space upload allowlist.

A deploy script is the one place in this repository where getting it wrong
publishes something permanently. Git history on the Hub is public and rewriting
it is not a thing a beginner will manage under pressure, so the properties
below are asserted rather than trusted: what ships is an allowlist, the 1GB
corpus and the .env are not on it, and a credential-shaped string stops the
upload instead of warning about it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "space" / "deploy.py"


@pytest.fixture(scope="module")
def deploy():
    spec = importlib.util.spec_from_file_location("finrag_deploy", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_payload_is_an_allowlist_not_an_ignore_list(deploy):
    """An ignore list leaks by omission; an allowlist can only miss something."""
    sources = {source for source, _ in deploy.PAYLOAD}

    assert "data/chroma_local" in sources, "the index must ship or the Space cannot answer"
    assert "data" not in sources, "shipping all of data/ would include 1GB of raw filings"
    assert "data/sec_filings" not in sources
    assert not any(s.startswith(".env") for s in sources)


def test_the_space_readme_is_substituted_for_the_project_readme(deploy):
    """Hugging Face reads its configuration from frontmatter in README.md.

    The project's own README must not grow a frontmatter block, so the Space
    gets a different file uploaded under that name.
    """
    mapping = dict(deploy.PAYLOAD)

    assert mapping["deploy/space/README.md"] == "README.md"
    assert "README.md" not in mapping, "the project README must not be uploaded as-is"

    frontmatter = (ROOT / "deploy" / "space" / "README.md").read_text(encoding="utf-8")
    assert frontmatter.startswith("---\n"), "the Space README needs YAML frontmatter first"
    head = frontmatter.split("---", 2)[1]
    for key in ("title:", "sdk: docker", "app_port: 7860"):
        assert key in head, f"frontmatter is missing {key!r}"


def test_the_dockerfile_is_uploaded_under_the_name_hugging_face_builds(deploy):
    """HF builds ./Dockerfile. A file named Dockerfile.space is simply ignored."""
    assert dict(deploy.PAYLOAD)["Dockerfile.space"] == "Dockerfile"


def test_every_payload_source_actually_exists(deploy):
    """A typo here fails at upload time, on the user's machine, mid-deploy."""
    missing = [source for source, _ in deploy.PAYLOAD if not (ROOT / source).exists()]
    assert not missing, f"payload names things that are not in the repo: {missing}"


def test_a_credential_shaped_string_stops_the_upload(deploy, tmp_path):
    """The audit must refuse, not warn. A key that leaks once has leaked."""
    (tmp_path / "config.py").write_text('KEY = "gsk_AbCdEf0123456789AbCdEf0123456789"\n')
    staged = [tmp_path / "config.py"]

    with pytest.raises(SystemExit) as raised:
        deploy.audit(staged, tmp_path)
    assert raised.value.code == 1


def test_the_audit_does_not_fire_on_a_variable_name(deploy, tmp_path):
    """A refusal that cries wolf gets worked around, which is worse than none."""
    (tmp_path / "llm.py").write_text('KEY_VAR = "GROQ_API_KEY"\nENV = "ANTHROPIC_API_KEY"\n')

    deploy.audit([tmp_path / "llm.py"], tmp_path)  # must not raise


def test_a_dotenv_is_refused_even_if_something_stages_it(deploy, tmp_path):
    """Belt and braces: src/ is copied as a tree and could carry a stray file."""
    secret = tmp_path / ".env"
    secret.write_text("GROQ_API_KEY=whatever\n")

    with pytest.raises(SystemExit):
        deploy.audit([secret], tmp_path)
