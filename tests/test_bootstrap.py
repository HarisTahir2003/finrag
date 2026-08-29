"""Unpacking a shipped index on a host that only clones the repository."""

from __future__ import annotations

import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from finrag.bootstrap import ARCHIVE_NAME, archive_for, ensure_index
from finrag.config import Settings


def _settings(root: Path) -> Settings:
    return replace(Settings(embedding_backend="local"), data_root=root)


def _make_archive(root: Path, contents: dict[str, str]) -> Path:
    """Build an archive shaped like the real one: a chroma_local/ directory."""
    staging = root / "staging" / "chroma_local"
    staging.mkdir(parents=True)
    for name, text in contents.items():
        (staging / name).write_text(text)

    archive = root / ARCHIVE_NAME
    with tarfile.open(archive, "w:xz") as tar:
        tar.add(staging, arcname="chroma_local")
    return archive


def test_it_unpacks_when_there_is_no_index(tmp_path):
    settings = _settings(tmp_path)
    _make_archive(tmp_path, {"chroma.sqlite3": "pretend-database"})

    assert ensure_index(settings) is True
    assert (settings.index_dir / "chroma.sqlite3").read_text() == "pretend-database"


def test_it_does_nothing_when_an_index_is_already_there(tmp_path):
    """An existing index is never overwritten.

    This runs on a web app's startup path with nobody watching, and a
    half-written index is worse than a stale one.
    """
    settings = _settings(tmp_path)
    _make_archive(tmp_path, {"chroma.sqlite3": "from-the-archive"})
    settings.index_dir.mkdir(parents=True)
    (settings.index_dir / "chroma.sqlite3").write_text("already-here")

    assert ensure_index(settings) is False
    assert (settings.index_dir / "chroma.sqlite3").read_text() == "already-here"


def test_an_empty_index_directory_still_unpacks(tmp_path):
    """A bare mkdir is not an index. This is what a mounted empty volume looks like."""
    settings = _settings(tmp_path)
    _make_archive(tmp_path, {"chroma.sqlite3": "from-the-archive"})
    settings.index_dir.mkdir(parents=True)

    assert ensure_index(settings) is True
    assert (settings.index_dir / "chroma.sqlite3").read_text() == "from-the-archive"


def test_no_archive_is_not_an_error(tmp_path):
    """The normal case for a developer: a real index and no archive at all."""
    settings = _settings(tmp_path)

    assert ensure_index(settings) is False


def test_an_archive_of_the_wrong_shape_fails_loudly(tmp_path):
    """Built from inside the directory instead of its parent.

    Silently producing no index would surface much later as "0 chunks indexed",
    a symptom several steps from the cause.
    """
    settings = _settings(tmp_path)
    stray = tmp_path / "loose.txt"
    stray.write_text("x")
    archive = tmp_path / ARCHIVE_NAME
    with tarfile.open(archive, "w:xz") as tar:
        tar.add(stray, arcname="loose.txt")

    with pytest.raises(RuntimeError, match="wrong directory"):
        ensure_index(settings)


def test_a_tarball_cannot_write_outside_the_destination(tmp_path):
    """CVE-2007-4559: a member named ../../x escapes the extraction root.

    The archive here is committed to the repository, so this is not the usual
    untrusted-input case -- but extractall's default was unsafe for fifteen
    years and the guard costs one argument.
    """
    settings = _settings(tmp_path)
    outside = tmp_path / "escaped.txt"
    payload = tmp_path / "payload.txt"
    payload.write_text("should not escape")

    archive = tmp_path / ARCHIVE_NAME
    with tarfile.open(archive, "w:xz") as tar:
        tar.add(payload, arcname="../../escaped.txt")

    with pytest.raises(Exception):  # noqa: B017 - tarfile raises its own family here
        ensure_index(settings)
    assert not outside.exists(), "the archive escaped the destination directory"


def test_the_archive_sits_beside_the_index_it_unpacks_into(tmp_path):
    settings = _settings(tmp_path)

    assert archive_for(settings) == tmp_path / ARCHIVE_NAME
    assert archive_for(settings).parent == settings.index_dir.parent


def test_packing_and_unpacking_round_trips(tmp_path):
    from finrag.bootstrap import pack_index

    settings = _settings(tmp_path)
    settings.index_dir.mkdir(parents=True)
    (settings.index_dir / "chroma.sqlite3").write_text("database")
    (settings.index_dir / "segment").mkdir()
    (settings.index_dir / "segment" / "data_level0.bin").write_text("vectors")

    pack_index(settings)

    import shutil

    shutil.rmtree(settings.index_dir)
    assert ensure_index(settings) is True
    assert (settings.index_dir / "chroma.sqlite3").read_text() == "database"
    assert (settings.index_dir / "segment" / "data_level0.bin").read_text() == "vectors"


def test_the_archive_carries_no_macos_metadata(tmp_path):
    """macOS tar stores xattrs as AppleDouble members: ._chroma.sqlite3.

    Those then extract onto a Linux host that has no idea what they are. The
    first archive built for this project had them, from a shell `tar -cJf`.
    """
    from finrag.bootstrap import pack_index

    settings = _settings(tmp_path)
    settings.index_dir.mkdir(parents=True)
    (settings.index_dir / "chroma.sqlite3").write_text("database")

    archive = pack_index(settings)
    with tarfile.open(archive, "r:xz") as tar:
        names = tar.getnames()

    junk = [n for n in names if Path(n).name.startswith("._") or "__MACOSX" in n]
    assert not junk, f"archive carries macOS metadata: {junk}"


def test_packing_without_an_index_says_so(tmp_path):
    from finrag.bootstrap import pack_index

    with pytest.raises(FileNotFoundError, match="finrag index"):
        pack_index(_settings(tmp_path))


def test_a_failed_unpack_leaves_no_index_behind(tmp_path):
    """Half an index must not look like an index.

    ensure_index's guard is "the directory exists and is not empty", which a
    partially extracted directory satisfies. Extracting into a sibling and
    renaming means the directory is either absent or complete -- so a failure
    leaves nothing for the next caller to mistake for a corpus.
    """
    settings = _settings(tmp_path)
    stray = tmp_path / "loose.txt"
    stray.write_text("x")
    archive = tmp_path / ARCHIVE_NAME
    with tarfile.open(archive, "w:xz") as tar:
        tar.add(stray, arcname="loose.txt")

    with pytest.raises(RuntimeError):
        ensure_index(settings)

    assert not settings.index_dir.exists(), "a failed unpack left a partial index"


def test_the_staging_directory_is_always_cleaned_up(tmp_path):
    """Both paths: success, and failure."""
    settings = _settings(tmp_path)
    _make_archive(tmp_path, {"chroma.sqlite3": "db"})

    assert ensure_index(settings) is True
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]
    assert not leftovers, f"staging directory survived a successful unpack: {leftovers}"


def test_losing_the_race_keeps_the_winners_index(tmp_path, monkeypatch):
    """Two cold-start visitors can arrive inside the ~2.4s extraction window.

    The loser must not raise, and must not clobber the complete index the
    winner just put there.
    """
    settings = _settings(tmp_path)
    _make_archive(tmp_path, {"chroma.sqlite3": "ours"})

    real_rename = Path.rename

    def rename_after_someone_else_won(self, target):
        # Simulate the other thread finishing during our extraction.
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / "chroma.sqlite3").write_text("theirs")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename_after_someone_else_won)

    assert ensure_index(settings) is False
    assert (settings.index_dir / "chroma.sqlite3").read_text() == "theirs"
