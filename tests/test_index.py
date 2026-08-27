"""Chunk identity and document construction.

These cover the idempotency fix. The original loop called
Chroma.from_documents() once per file with no IDs, so a second ingest run
appended a duplicate copy of every chunk.
"""

from __future__ import annotations

from datetime import date

from finrag.ingest.index import chunk_id, documents_for_filing
from finrag.ingest.metadata import FilingMetadata

META = FilingMetadata(ticker="AMZN", fiscal_year=2022, period_of_report=date(2022, 12, 31))


def test_chunk_id_is_stable_for_identical_content():
    assert chunk_id(META, 0, "revenue was 514,005") == chunk_id(META, 0, "revenue was 514,005")


def test_chunk_id_changes_with_content():
    assert chunk_id(META, 0, "alpha") != chunk_id(META, 0, "beta")


def test_chunk_id_changes_with_position():
    assert chunk_id(META, 0, "same") != chunk_id(META, 1, "same")


def test_chunk_id_separates_companies_and_years():
    other_company = FilingMetadata(ticker="AAPL", fiscal_year=2022)
    other_year = FilingMetadata(ticker="AMZN", fiscal_year=2023)
    assert chunk_id(META, 0, "x") != chunk_id(other_company, 0, "x")
    assert chunk_id(META, 0, "x") != chunk_id(other_year, 0, "x")


def test_chunk_id_is_readable():
    """IDs are inspected by hand when debugging retrieval, so keep them legible."""
    assert chunk_id(META, 7, "content").startswith("AMZN:2022:00007:")


def test_documents_carry_corrected_metadata(amzn_fy2022, recursive_settings):
    docs = documents_for_filing(amzn_fy2022, recursive_settings)
    assert docs, "fixture should produce at least one chunk"
    for doc in docs:
        assert doc.metadata["ticker"] == "AMZN"
        assert doc.metadata["year"] == 2022, "must be the fiscal year, not the 2023 filing year"
        assert doc.metadata["id"].startswith("AMZN:2022:")


def test_documents_are_deterministic_across_runs(amzn_fy2022, recursive_settings):
    """Re-ingesting unchanged filings must upsert the same rows, not new ones."""
    first = documents_for_filing(amzn_fy2022, recursive_settings)
    second = documents_for_filing(amzn_fy2022, recursive_settings)
    assert [d.metadata["id"] for d in first] == [d.metadata["id"] for d in second]


def test_exhibit_text_is_not_indexed(amzn_fy2022, recursive_settings):
    docs = documents_for_filing(amzn_fy2022, recursive_settings)
    assert not any("SUBSIDIARIES OF THE REGISTRANT" in d.page_content for d in docs)
