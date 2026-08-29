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

    torch_lines = [ln for ln in lines if ln.split("=")[0].strip().lower() == "torch"]
    assert torch_lines, "torch must be declared explicitly, not left to a transitive resolve"
    for line in torch_lines:
        assert "+cpu" in line, (
            f"{line!r} does not request a +cpu build, so pip may take the CUDA wheel from PyPI"
        )
    assert not any(".whl" in ln for ln in lines), (
        "a pinned wheel URL is architecture-locked; use the +cpu local version instead"
    )
