"""The regression tests for the fiscal-year bug.

The old implementation read the year out of the accession number, which encodes
when a filing was submitted rather than the period it covers. These tests pin
the corrected behaviour for both the cases it got wrong and the cases it got
right by coincidence.
"""

from __future__ import annotations

from datetime import date

import pytest

from finrag.ingest.metadata import (
    filing_metadata,
    fiscal_year_from_period,
    parse_submission_header,
    ticker_from_path,
)


def accession_year(filed_yyyymmdd: str) -> int:
    """Reproduce the old, buggy derivation so the difference is explicit."""
    return int("20" + filed_yyyymmdd[2:4])


@pytest.mark.parametrize(
    ("ticker", "period_end", "filed", "expected_fy"),
    [
        # December year ends: the old code was off by one on every one of these.
        ("AMZN", date(2022, 12, 31), "20230203", 2022),
        ("GOOGL", date(2022, 12, 31), "20230203", 2022),
        ("META", date(2023, 12, 31), "20240202", 2023),
        ("TSLA", date(2023, 12, 31), "20240126", 2023),
        ("NFLX", date(2023, 12, 31), "20240126", 2023),
        ("JPM", date(2022, 12, 31), "20230221", 2022),
        # Non-December year ends: filed in the same calendar year, so the old
        # code happened to agree.
        ("AAPL", date(2023, 9, 30), "20231103", 2023),
        ("MSFT", date(2023, 6, 30), "20230727", 2023),
        ("V", date(2023, 9, 30), "20231115", 2023),
        ("NVDA", date(2024, 1, 28), "20240221", 2024),
    ],
)
def test_fiscal_year_matches_period_not_filing_date(ticker, period_end, filed, expected_fy):
    assert fiscal_year_from_period(period_end, ticker) == expected_fy


@pytest.mark.parametrize(
    ("ticker", "period_end", "filed"),
    [
        ("AMZN", date(2022, 12, 31), "20230203"),
        ("GOOGL", date(2022, 12, 31), "20230203"),
        ("META", date(2023, 12, 31), "20240202"),
        ("TSLA", date(2023, 12, 31), "20240126"),
        ("NFLX", date(2023, 12, 31), "20240126"),
        ("JPM", date(2022, 12, 31), "20230221"),
    ],
)
def test_december_year_ends_were_previously_wrong(ticker, period_end, filed):
    """Guards the specific defect: these six must not equal the accession year."""
    assert fiscal_year_from_period(period_end, ticker) != accession_year(filed)


def test_parses_header_fields(amzn_fy2022):
    header = parse_submission_header(amzn_fy2022.read_text())
    assert header["period_of_report"] == "20221231"
    assert header["filed_as_of_date"] == "20230203"
    assert header["form_type"] == "10-K"
    assert header["cik"] == "0001018724"


def test_filing_metadata_end_to_end(amzn_fy2022):
    meta = filing_metadata(amzn_fy2022)
    assert meta.ticker == "AMZN"
    assert meta.fiscal_year == 2022, "filed in 2023, but covers fiscal 2022"
    assert meta.period_of_report == date(2022, 12, 31)
    assert meta.filed_as_of_date == date(2023, 2, 3)


def test_january_year_end_labels_by_ending_year(nvda_fy2024):
    assert filing_metadata(nvda_fy2024).fiscal_year == 2024


def test_chroma_metadata_is_flat(aapl_fy2023):
    meta = filing_metadata(aapl_fy2023).as_chroma_metadata()
    assert meta["ticker"] == "AAPL"
    assert meta["year"] == 2023
    assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())


def test_ticker_comes_from_the_path(amzn_fy2022):
    assert ticker_from_path(amzn_fy2022) == "AMZN"
    assert ticker_from_path("/tmp/somewhere/else.txt") == "UNKNOWN"


def test_missing_period_refuses_to_guess(tmp_path):
    """A wrong year is invisible at query time, so failing is safer than guessing."""
    bad = tmp_path / "sec-edgar-filings" / "XXX" / "10-K" / "acc" / "full-submission.txt"
    bad.parent.mkdir(parents=True)
    bad.write_text("<SEC-HEADER>\nCONFORMED SUBMISSION TYPE:\t10-K\n</SEC-HEADER>")
    with pytest.raises(ValueError, match="CONFORMED PERIOD OF REPORT"):
        filing_metadata(bad)
