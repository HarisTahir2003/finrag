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
    extra = _requirements_declared() - _pyproject_expected()
    assert not extra, f"requirements.txt declares {sorted(extra)}, which pyproject does not"


def test_torch_is_pinned_to_a_cpu_only_wheel():
    """The line that decides whether the build succeeds at all.

    A plain `torch` requirement resolves to the CUDA wheel on Linux: 527MB and
    fifteen nvidia-* dependencies, on a free tier with no GPU.
    """
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    torch_lines = [
        ln
        for ln in text.splitlines()
        if "torch" in ln and not ln.strip().startswith("#") and ln.strip()
    ]

    assert torch_lines, "torch must be declared explicitly, not left to a transitive resolve"
    for line in torch_lines:
        assert "download.pytorch.org/whl/cpu" in line, (
            f"{line.strip()!r} does not pin the CPU index; this build pulls CUDA"
        )
