"""Publish the Streamlit UI to a Hugging Face Space.

    python deploy/space/deploy.py --repo-id <your-username>/finrag --dry-run
    python deploy/space/deploy.py --repo-id <your-username>/finrag

Two things this does that a plain `git push` to the Space remote would not.

It stages an explicit allowlist rather than the repository. The Space needs
seven things; the repository contains a 1GB corpus of raw filings, a .env, and
evaluation results. An allowlist cannot leak a file by forgetting to ignore it,
and the failure mode of getting it wrong is "the build is missing something"
rather than "a secret is now public and permanent in git history".

It substitutes README.md. Hugging Face reads the Space's configuration from
YAML frontmatter at the top of README.md, and this project's README is a
filled-in project README that must not grow a frontmatter block. deploy/space/
holds the Space's own README and it is uploaded under that name.

The API key is NOT uploaded and must not be. Set it once in the Space's
Settings -> Variables and secrets, as a *secret* named GROQ_API_KEY, where it
reaches the container as an environment variable at runtime and never enters a
layer or a commit.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (source, destination-in-space). Everything the image needs and nothing else.
PAYLOAD: list[tuple[str, str]] = [
    ("deploy/space/README.md", "README.md"),  # the frontmatter HF reads
    ("Dockerfile.space", "Dockerfile"),  # HF builds ./Dockerfile by name
    ("pyproject.toml", "pyproject.toml"),
    ("LICENSE", "LICENSE"),
    ("app.py", "app.py"),
    ("src", "src"),
    ("data/chroma_local", "data/chroma_local"),
]

# Anything matching these never ships, whatever the allowlist says. Belt and
# braces: src/ is copied as a tree, so a stray file inside it would otherwise
# ride along.
FORBIDDEN = re.compile(r"(^|/)(\.env|\.git|.*\.key|.*\.pem|secrets?\.(ya?ml|json))(/|$)")

# A key that has leaked once has leaked permanently, so this refuses to upload
# rather than warn. Groq keys start gsk_; the generic patterns catch the rest.
SECRET_SHAPES = [
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{30,}"),
]
SCANNABLE = {".py", ".toml", ".md", ".txt", ".yaml", ".yml", ".cfg", ".ini", ""}


def stage(destination: Path) -> list[Path]:
    """Copy the allowlist into a clean directory and return what landed."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for source_name, target_name in PAYLOAD:
        source = ROOT / source_name
        if not source.exists():
            sys.exit(f"missing from the repository: {source_name}")
        target = destination / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "*.egg-info"),
            )
        else:
            shutil.copy2(source, target)

    return sorted(p for p in destination.rglob("*") if p.is_file())


def audit(files: list[Path], destination: Path) -> None:
    """Refuse to continue if anything staged looks like a credential."""
    problems: list[str] = []

    for path in files:
        relative = path.relative_to(destination).as_posix()
        if FORBIDDEN.search(relative):
            problems.append(f"{relative}: matches a never-upload pattern")
            continue
        if path.suffix.lower() not in SCANNABLE or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for shape in SECRET_SHAPES:
            found = shape.search(text)
            if found:
                problems.append(
                    f"{relative}: contains something shaped like a key ({found.group()[:7]}…)"
                )

    if problems:
        print("\nREFUSING TO UPLOAD:\n", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/finrag")
    parser.add_argument("--private", action="store_true", help="create the Space private")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="stage and audit, print the manifest, upload nothing",
    )
    parser.add_argument(
        "--staging",
        default=str(ROOT / "build" / "space"),
        help="where to assemble the upload",
    )
    args = parser.parse_args()

    destination = Path(args.staging)
    files = stage(destination)
    audit(files, destination)

    total = sum(p.stat().st_size for p in files)
    print(f"staged {len(files)} files, {total / 1_048_576:.1f} MB, in {destination}")
    for path in files:
        size = path.stat().st_size
        if size > 1_000_000:
            print(f"  {path.relative_to(destination).as_posix():<40} {size / 1_048_576:>7.1f} MB")

    if args.dry_run:
        print("\ndry run: nothing uploaded.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("needs huggingface_hub: pip install huggingface_hub")

    api = HfApi()
    try:
        whoami = api.whoami()
    except Exception:  # noqa: BLE001 - an unauthenticated run should say so plainly
        sys.exit("not logged in. Run:  hf auth login")
    print(f"\nuploading as {whoami.get('name', '?')} to space {args.repo_id}")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(destination),
        repo_id=args.repo_id,
        repo_type="space",
        commit_message="Deploy finrag",
    )
    print(f"\ndone: https://huggingface.co/spaces/{args.repo_id}")
    print("Set GROQ_API_KEY under Settings -> Variables and secrets, as a SECRET.")


if __name__ == "__main__":
    main()
