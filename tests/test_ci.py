"""The CI workflow itself.

Nothing else checks this file, and it fails silently in the direction that
matters: a job that stops covering something does not go red, it goes quiet.
`deploy/` was a real source directory for several commits while CI's lint step
named only `src tests app.py`, so nothing checked it and nothing said so.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 ships no stdlib TOML parser
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _all_run_commands(workflow: dict) -> str:
    commands = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if isinstance(step.get("run"), str):
                commands.append(step["run"])
            if isinstance(step.get("uses"), str):
                commands.append(step["uses"])
    return "\n".join(commands)


def test_the_workflow_is_valid_yaml(workflow):
    """A syntax error here is only discovered by pushing it."""
    assert workflow["jobs"], "no jobs defined"


def test_every_python_source_directory_is_linted(workflow):
    """A directory CI does not name is a directory CI does not check.

    Finding the omission requires noticing an absence, which is why this is a
    test rather than a habit.
    """
    linted = [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if isinstance(step.get("run"), str) and step["run"].strip().startswith("ruff check")
    ]
    assert linted, "no ruff check step in the workflow"
    covered = " ".join(linted)

    # Top-level directories holding Python that a human wrote.
    expected = {
        path.name
        for path in ROOT.iterdir()
        if path.is_dir()
        and not path.name.startswith((".", "_"))
        and path.name not in {"data", "results", "docs", "mlruns", "build", "htmlcov"}
        and any(path.rglob("*.py"))
    }
    missing = sorted(name for name in expected if name not in covered)
    assert not missing, f"CI never lints {missing}; add them to the ruff check step"


def test_the_deployment_install_path_is_exercised(workflow):
    """requirements.txt decides whether the public demo builds.

    It is not pyproject.toml, so none of the other jobs touch it. Without this
    the one file the deployment actually depends on is the one file CI never
    exercises.
    """
    assert "deploy-install" in workflow["jobs"], "no job installs requirements.txt"
    commands = _all_run_commands(workflow)
    assert "pip install -r requirements.txt" in commands


def test_ci_would_notice_cuda_creeping_back_in(workflow):
    """The failure is silent and expensive.

    A plain `torch` requirement resolves to the CUDA build on Linux -- 527MB
    and fifteen nvidia-* packages -- which installs fine and then does not fit
    in the host's 690MB ceiling. Nothing about the symptom points at the cause.
    """
    commands = _all_run_commands(workflow)
    assert "nvidia-" in commands, "nothing checks that the CPU-only torch pin held"


def test_both_dockerfiles_are_built(workflow):
    """Neither is exercised by the test suite, and both have broken before."""
    commands = _all_run_commands(workflow)
    assert "docker/build-push-action" in commands

    built = str(workflow["jobs"].get("images", {}))
    for dockerfile in ("Dockerfile", "Dockerfile.space"):
        assert dockerfile in built, f"{dockerfile} is never built in CI"


def test_the_test_job_covers_the_python_versions_pyproject_claims(workflow):
    """requires-python is a promise; the matrix is whether it is kept."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    floor = data["project"]["requires-python"].lstrip(">=")

    versions = workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    assert floor in versions, (
        f"pyproject promises Python {floor} but CI never runs it; "
        "either test it or raise requires-python"
    )
