from __future__ import annotations

import pytest

from finrag.chunking import chunk_filing, chunk_recursive


def test_recursive_chunking_respects_size():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 100 for i in range(40))
    chunks = chunk_recursive(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 700 for c in chunks), "allow slack for separator handling"


def test_no_empty_chunks():
    assert all(c.strip() for c in chunk_recursive("\n\n\n  \n\ntext here\n\n\n", 100, 10))


def test_short_text_is_one_chunk():
    assert chunk_recursive("A single short sentence.", 3000, 300) == ["A single short sentence."]


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown chunk strategy"):
        chunk_filing(html="<p>x</p>", text="x", strategy="magic")


def test_semantic_chunking_survives_an_unusable_unstructured(monkeypatch):
    """Installed-but-broken has to degrade exactly like not-installed.

    Current `unstructured` fetches a spaCy sentence model the first time it
    partitions anything, so an offline machine, a proxy, or a missing CA bundle
    raises from inside `partition_html` rather than at import. Catching only
    ImportError left that case fatal, and fatal at the worst moment: chunking
    runs after every filing is downloaded and parsed.
    """
    partition = pytest.importorskip("unstructured.partition.html")

    def unusable(*args, **kwargs):
        raise RuntimeError("Failed to download spaCy model: certificate verify failed")

    monkeypatch.setattr(partition, "partition_html", unusable)

    html = "<p>" + "Revenue rose sharply. " * 400 + "</p>"
    chunks = chunk_filing(html=html, text="", strategy="semantic", chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1, "should have fallen back to recursive chunking, not returned nothing"
    assert all(c.strip() for c in chunks)


def test_semantic_chunking_handles_inline_xbrl():
    """The shape every real 10-K arrives in, which no fixture had.

    Filings are inline XBRL: an `<XBRL>` element and an XML declaration sit in
    front of the `<html>` tag. partition_html reads that preamble, decides the
    document is XML, and returns zero elements -- no error, at any size. The
    committed fixtures are hand-written HTML starting at `<html>`, so semantic
    chunking passed every test while producing an empty index for all four
    real filings on disk.
    """
    pytest.importorskip("unstructured.partition.html")

    body = "".join(f"<p>Total net sales increased {i} percent year over year.</p>" for i in range(80))
    wrapped = (
        "\n<XBRL>\n<?xml version='1.0' encoding='ASCII'?>\n"
        "<!--XBRL Document Created with the Workiva Platform-->\n"
        f'<html xmlns:link="http://www.xbrl.org/2003/linkbase"><body>{body}</body></html>'
    )

    chunks = chunk_filing(
        html=wrapped, text="", strategy="semantic", chunk_size=500, chunk_overlap=50
    )
    assert chunks, "inline-XBRL filings must produce chunks, not an empty index"
    assert any("net sales" in c for c in chunks)


def test_semantic_chunking_falls_back_when_it_produces_nothing(monkeypatch):
    """An empty result is a failure that never raises, so it needs its own guard."""
    partition = pytest.importorskip("unstructured.partition.html")
    monkeypatch.setattr(partition, "partition_html", lambda *a, **k: [])

    html = "<html><body><p>" + "Revenue rose sharply. " * 300 + "</p></body></html>"
    chunks = chunk_filing(html=html, text="", strategy="semantic", chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1, "should have fallen through to recursive rather than returning []"
