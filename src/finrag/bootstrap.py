"""Make a shipped index usable on a host that only gives you a git checkout.

The container images bake the index in at build time. A platform-as-a-service
does not build an image: it clones the repository, installs a dependency file
and runs the entrypoint, so whatever the index needs must survive in git.

134MB of Chroma does not. GitHub blocks a single file over 100MB outright, and
Git LFS is not reliably fetched by these platforms -- the failure mode is a
130-byte pointer file arriving where a database was expected, which surfaces
much later as an empty index rather than as a download error.

Compressed the same index is 45MB, which is an ordinary git object. So the
repository carries an archive and this unpacks it on first boot. Measured on
the real index: 134MB -> 45MB, 2.4 seconds to unpack, done once per container.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

from .config import Settings, get_settings

log = logging.getLogger(__name__)

ARCHIVE_NAME = "chroma_local.tar.xz"


def archive_for(settings: Settings) -> Path:
    """Where the shipped archive lives: beside the index it unpacks into."""
    return settings.index_dir.parent / ARCHIVE_NAME


def ensure_index(settings: Settings | None = None, archive: Path | None = None) -> bool:
    """Unpack the shipped index if it is not already on disk.

    Returns True when it unpacked something, False when there was nothing to
    do -- which is the normal case everywhere except a cold container.

    Deliberately conservative: an existing index is never touched, even if the
    archive is newer. A half-written index is worse than a stale one, and the
    place this runs is a web app's startup path where nobody is watching.
    """
    settings = settings or get_settings()
    index_dir = settings.index_dir

    if index_dir.exists() and any(index_dir.iterdir()):
        return False

    archive = archive or archive_for(settings)
    if not archive.exists():
        log.debug("no index and no archive at %s; nothing to unpack", archive)
        return False

    destination = index_dir.parent
    destination.mkdir(parents=True, exist_ok=True)
    log.info("unpacking %s into %s", archive.name, destination)

    with tarfile.open(archive, "r:xz") as tar:
        # filter="data" refuses absolute paths, parent-directory escapes,
        # symlinks pointing outside the tree, and device nodes. Without it a
        # tarball can write anywhere the process can, which is the whole
        # CVE-2007-4559 family. Python 3.14 makes this the default; naming it
        # keeps the behaviour identical on 3.10 through 3.13.
        tar.extractall(destination, filter="data")

    if not index_dir.exists():
        raise RuntimeError(
            f"{archive.name} did not contain {index_dir.name}/ -- "
            "the archive was built from the wrong directory"
        )
    return True


def pack_index(settings: Settings | None = None, archive: Path | None = None) -> Path:
    """Compress the index into the archive the repository ships.

    Run this after re-indexing, or the deployed app serves the old corpus.

    Written in Python rather than left as a `tar -cJf` in a README for one
    reason: on macOS, tar stores extended attributes as AppleDouble members --
    `._chroma.sqlite3` beside `chroma.sqlite3` -- and those then land on a Linux
    host that has no idea what they are. tarfile does not do that on any
    platform, so the archive is identical wherever it is built.
    """
    settings = settings or get_settings()
    index_dir = settings.index_dir
    if not index_dir.exists() or not any(index_dir.iterdir()):
        raise FileNotFoundError(f"no index at {index_dir}; run `finrag index` first")

    archive = archive or archive_for(settings)
    archive.parent.mkdir(parents=True, exist_ok=True)

    # preset=6 rather than xz's default 9: on the real index 9 saves under 2%
    # for roughly twice the compression time, and this runs on a laptop.
    with tarfile.open(archive, "w:xz", preset=6) as tar:
        tar.add(index_dir, arcname=index_dir.name)

    log.info(
        "packed %s (%.0fMB) into %s (%.0fMB)",
        index_dir.name,
        sum(f.stat().st_size for f in index_dir.rglob("*") if f.is_file()) / 1e6,
        archive.name,
        archive.stat().st_size / 1e6,
    )
    return archive
