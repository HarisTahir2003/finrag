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
