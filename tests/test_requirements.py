"""requirements.txt must not drift from pyproject.toml.

Two files listing the same dependencies is a duplication the deployment target
forces on us: Streamlit Community Cloud installs a requirements file and does
not read pyproject. Duplication that nothing checks is duplication that silently
diverges, and the way it would show up is a deployed app missing a package that
works fine on every developer's machine.
"""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_EXTRAS = ("local", "app", "groq")

# Declared in requirements.txt but not in pyproject, deliberately. torch
# arrives transitively through sentence-transformers either way; naming it here
# is the only way to say WHICH build, and on Linux the default is a 527MB CUDA
# wheel with fifteen nvidia-* dependencies. Anything added to this set is a
# claim that the deployment needs to control a transitive dependency's version,
# which needs a reason written next to it.
PINNED_TRANSITIVES = {"torch"}


def _names(requirements: list[str]) -> set[str]:
    """Distribution names, normalised: `Foo_Bar>=1` and `foo-bar` are one thing."""
    found = set()
    for line in requirements:
        line = line.split("#")[0].strip()
        if not line or line.startswith(("-", "http://", "https://")):
            continue
        name = re.split(r"[<>=!\[;@ ]", line, maxsplit=1)[0]
        if name:
            found.add(name.lower().replace("_", "-"))
    return found


def _pyproject_expected() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    wanted = list(project["dependencies"])
    for extra in SHIPPED_EXTRAS:
        wanted += project["optional-dependencies"][extra]
    return _names(wanted)


def _requirements_declared() -> set[str]:
    return _names((ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines())


def test_requirements_covers_everything_the_app_imports():
    missing = _pyproject_expected() - _requirements_declared()
    assert not missing, (
        f"requirements.txt is missing {sorted(missing)}; the hosted app would "
        "fail on import while every local install keeps working"
    )


def test_requirements_ships_nothing_the_app_does_not_need():
    """Extras cost build time and memory on a tier that has little of either."""
    extra = _requirements_declared() - _pyproject_expected() - PINNED_TRANSITIVES
    assert not extra, (
        f"requirements.txt declares {sorted(extra)}, which pyproject does not. "
        "If this is a transitive dependency whose build must be controlled, add "
        "it to PINNED_TRANSITIVES with the reason."
    )


def test_torch_comes_from_the_cpu_index_on_every_architecture():
    """The line that decides whether the build succeeds at all.

    A plain `torch` requirement resolves to the CUDA wheel on Linux: 527MB and
    fifteen nvidia-* dependencies, on a free tier with no GPU.

    Two halves, and both are needed. The extra index is where a "+cpu" build
    exists at all; the "+cpu" local version is what forces pip to take it from
    there rather than from PyPI. Pinning the wheel URL instead also works and
    is architecture-locked -- the x86_64 URL this file used to carry fails to
    resolve on aarch64 with "not a supported wheel on this platform".
    """
    lines = [
        ln.strip()
        for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]

    assert any(
        "--extra-index-url" in ln and "download.pytorch.org/whl/cpu" in ln for ln in lines
    ), "no CPU index declared; a plain torch requirement pulls CUDA on Linux"

    torch_lines = [ln for ln in lines if ln.split("=")[0].split(";")[0].strip().lower() == "torch"]
    assert torch_lines, "torch must be declared explicitly, not left to a transitive resolve"

    # "+cpu" is only meaningful where a CUDA build exists, which is Linux.
    # PyTorch publishes no "+cpu" wheel for macOS at all -- requiring it there
    # makes `pip install -r requirements.txt` fail outright on a Mac, which is
    # how this test earned its marker-awareness.
    linux = [ln for ln in torch_lines if 'sys_platform != "linux"' not in ln]
    assert linux, "no torch requirement applies on Linux, which is where it is deployed"
    for line in linux:
        assert "+cpu" in line, (
            f"{line!r} applies on Linux without requesting a +cpu build, so pip "
            "may take the 527MB CUDA wheel from PyPI"
        )

    assert not any(".whl" in ln for ln in lines), (
        "a pinned wheel URL is architecture-locked; use the +cpu local version instead"
    )


# Streamlit Community Cloud picks ONE dependency file, by a fixed precedence,
# searching the entrypoint's directory then the repository root:
#
#     uv.lock  >  Pipfile  >  environment.yml  >  requirements.txt  >  pyproject.toml
#
# https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
HIGHER_PRECEDENCE = ("uv.lock", "Pipfile", "environment.yml", "environment.yaml")


def test_nothing_outranks_the_requirements_file():
    """Only requirements.txt may win the dependency-file election.

    The deployed build already logs "WARN: More than one requirements file
    detected" because pyproject.toml is also present. That one is harmless:
    requirements.txt outranks it. Adding any of the files above would not be --
    it would silently win, and the CPU-only torch pin lives in requirements.txt
    alone, so the build would start pulling the 527MB CUDA wheel and fifteen
    nvidia-* packages onto a host with a 690MB memory floor.

    The failure would appear as a deploy that used to work and now does not,
    with nothing in the diff obviously about torch.
    """
    found = [name for name in HIGHER_PRECEDENCE if (ROOT / name).exists()]
    assert not found, (
        f"{found} outranks requirements.txt on Streamlit Community Cloud, so the "
        "CPU-only torch pin would be skipped. Move those dependencies into "
        "requirements.txt or delete the file."
    )


def test_pyproject_is_not_poetry_shaped():
    """The one that would matter if precedence ever changed.

    Community Cloud reads pyproject.toml as a Poetry file. This project's is
    setuptools, so being chosen would not merely install the wrong extras -- it
    would not be understood at all. Nothing here should ever add a
    [tool.poetry] section in the hope of making it work; requirements.txt is
    the supported path and already wins.
    """
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "poetry" not in data.get("tool", {}), (
        "a [tool.poetry] section would make pyproject.toml a viable dependency "
        "file for Community Cloud, creating two live paths instead of one"
    )
    assert data["build-system"]["build-backend"].startswith("setuptools")
