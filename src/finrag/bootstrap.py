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
import os
import shutil
import tarfile
import threading
from pathlib import Path

from .config import Settings, get_settings

log = logging.getLogger(__name__)

# Serializes ensure_index across a process's threads -- see the function.
_UNPACK_LOCK = threading.Lock()

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

    # Serialize unpacking within the process. Streamlit runs each browser
    # session's script in its own thread, so on a cold container several reruns
    # reach this at once. Without the lock they raced on a shared staging
    # directory: one renamed the unpacked tree away while another was still
    # extracting into it, and the loser then found chroma_local/ missing and
    # crashed the app with "the archive was built from the wrong directory" --
    # for an archive that is in fact correct. The first thread in unpacks; the
    # rest re-check and return.
    with _UNPACK_LOCK:
        if index_dir.exists() and any(index_dir.iterdir()):
            return False

        destination = index_dir.parent
        destination.mkdir(parents=True, exist_ok=True)
        log.info("unpacking %s into %s", archive.name, destination)

        # A staging directory unique to this attempt -- pid and thread id -- so
        # two runs can never share or delete each other's, even if the lock is
        # ever removed. Unpack into it and rename, rather than extracting in
        # place: rename on one filesystem is atomic, so index_dir is either
        # absent or complete, never the half-written state the guard above would
        # mistake for a real index.
        staging = destination / f".{index_dir.name}.unpacking.{os.getpid()}.{threading.get_ident()}"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)

        try:
            with tarfile.open(archive, "r:xz") as tar:
                # filter="data" refuses absolute paths, parent-directory
                # escapes, symlinks pointing outside the tree, and device nodes.
                # Without it a tarball can write anywhere the process can, which
                # is the whole CVE-2007-4559 family. Python 3.14 makes this the
                # default; naming it keeps the behaviour identical on 3.10-3.13.
                tar.extractall(staging, filter="data")

            unpacked = staging / index_dir.name
            if not unpacked.exists():
                raise RuntimeError(
                    f"{archive.name} did not contain {index_dir.name}/ -- "
                    "the archive was built from the wrong directory"
                )

            try:
                unpacked.rename(index_dir)
            except OSError:
                # The lock serializes threads within this process; a *second
                # process* (the CLI, or multiple Docker workers) has no such
                # guard, and renaming onto a directory another process already
                # populated raises. Its copy is complete, so drop ours.
                if index_dir.exists() and any(index_dir.iterdir()):
                    log.info("another worker unpacked the index first; keeping theirs")
                    return False
                raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

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
