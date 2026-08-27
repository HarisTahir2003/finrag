"""Deriving trustworthy metadata from a filing.

This module exists because of a bug. The original pipeline read the fiscal year
out of the accession number::

    accession_dir = parts[root_idx + 3]      # 0000320193-23-000106
    year = int("20" + accession_dir.split("-")[1])

The middle segment of an accession number is the year the filing was *submitted*,
which is not the year it covers. A company whose fiscal year ends in December
files its FY2022 annual report in early 2023, so every chunk of that report was
labelled ``year: 2023``. Any query filtered to 2023 then retrieved the FY2022
report, silently.

Of the ten tickers in the default set, six were wrong (AMZN, GOOGL, META, TSLA,
NFLX, JPM) and four were right only because their fiscal years happen to end
before December.

The fix is to read ``CONFORMED PERIOD OF REPORT`` from the SEC submission header,
which is the authoritative period-end date for the filing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# The header sits in the first few KB of full-submission.txt as plain
# "TAG: value" lines, before any document markup.
HEADER_SCAN_BYTES = 16_384

_HEADER_FIELDS = {
    "period_of_report": re.compile(r"^\s*CONFORMED PERIOD OF REPORT:\s*(\d{8})\s*$", re.M),
    "filed_as_of_date": re.compile(r"^\s*FILED AS OF DATE:\s*(\d{8})\s*$", re.M),
    "form_type": re.compile(r"^\s*CONFORMED SUBMISSION TYPE:\s*(\S+)\s*$", re.M),
    "company_name": re.compile(r"^\s*COMPANY CONFORMED NAME:\s*(.+?)\s*$", re.M),
    "cik": re.compile(r"^\s*CENTRAL INDEX KEY:\s*(\d+)\s*$", re.M),
}

# Companies whose own fiscal-year label disagrees with the calendar year their
# fiscal year ends in. Retailers with a late-January or February year end are the
# usual case: Target's "fiscal 2023" ends in February 2024. None of the default
# ten tickers need an entry -- NVIDIA ends in late January and labels it by the
# ending calendar year, which is what the default rule already produces.
# Map ticker -> offset added to the period-end calendar year.
FISCAL_YEAR_OVERRIDES: dict[str, int] = {}


@dataclass(frozen=True)
class FilingMetadata:
    """What we know about one filing, and where it came from."""

    ticker: str
    fiscal_year: int
    period_of_report: date | None = None
    filed_as_of_date: date | None = None
    form_type: str | None = None
    company_name: str | None = None
    cik: str | None = None
    source_path: str | None = None

    def as_chroma_metadata(self) -> dict[str, str | int]:
        """Flatten for the vector store, which only accepts scalars."""
        meta: dict[str, str | int] = {"ticker": self.ticker, "year": self.fiscal_year}
        if self.period_of_report:
            meta["period_of_report"] = self.period_of_report.isoformat()
        if self.filed_as_of_date:
            meta["filed_as_of_date"] = self.filed_as_of_date.isoformat()
        if self.form_type:
            meta["form_type"] = self.form_type
        if self.cik:
            meta["cik"] = self.cik
        return meta


def _parse_yyyymmdd(value: str) -> date | None:
    try:
        return date(int(value[0:4]), int(value[4:6]), int(value[6:8]))
    except (ValueError, IndexError):
        return None


def parse_submission_header(text: str) -> dict[str, str]:
    """Pull the SEC header fields out of a full-submission document."""
    head = text[:HEADER_SCAN_BYTES]
    found: dict[str, str] = {}
    for key, pattern in _HEADER_FIELDS.items():
        match = pattern.search(head)
        if match:
            found[key] = match.group(1).strip()
    return found


def fiscal_year_from_period(period_end: date, ticker: str | None = None) -> int:
    """Fiscal year label for a filing covering a period ending on ``period_end``.

    The convention across almost all US registrants is that the fiscal year is
    named for the calendar year in which it ends: a year ending 2022-12-31 is
    FY2022, and one ending 2024-01-28 is FY2024. Companies that label it
    otherwise are listed in FISCAL_YEAR_OVERRIDES.
    """
    year = period_end.year
    if ticker:
        year += FISCAL_YEAR_OVERRIDES.get(ticker.upper(), 0)
    return year


def ticker_from_path(file_path: str | Path) -> str:
    """Recover the ticker from sec-edgar-downloader's directory layout.

    Layout is ``.../sec-edgar-filings/<TICKER>/<FORM>/<ACCESSION>/full-submission.txt``.
    The path is the only place the ticker appears -- the header carries the
    company's legal name and CIK, not its trading symbol.
    """
    parts = Path(file_path).parts
    if "sec-edgar-filings" in parts:
        idx = parts.index("sec-edgar-filings")
        if idx + 1 < len(parts):
            return parts[idx + 1].upper()
    return "UNKNOWN"


def filing_metadata(file_path: str | Path, text: str | None = None) -> FilingMetadata:
    """Build metadata for one filing, reading the header for the fiscal year.

    Raises ValueError when the period of report is missing, rather than guessing.
    A filing we cannot date correctly is worse than one we skip, because a wrong
    year is invisible at query time.
    """
    path = Path(file_path)
    if text is None:
        text = path.read_text(encoding="utf-8", errors="ignore")[:HEADER_SCAN_BYTES]

    header = parse_submission_header(text)
    ticker = ticker_from_path(path)

    raw_period = header.get("period_of_report")
    if not raw_period:
        raise ValueError(
            f"{path}: no CONFORMED PERIOD OF REPORT in the submission header, so the fiscal "
            "year cannot be determined. Refusing to guess."
        )
    period = _parse_yyyymmdd(raw_period)
    if period is None:
        raise ValueError(f"{path}: unparseable CONFORMED PERIOD OF REPORT {raw_period!r}")

    filed_raw = header.get("filed_as_of_date")
    return FilingMetadata(
        ticker=ticker,
        fiscal_year=fiscal_year_from_period(period, ticker),
        period_of_report=period,
        filed_as_of_date=_parse_yyyymmdd(filed_raw) if filed_raw else None,
        form_type=header.get("form_type"),
        company_name=header.get("company_name"),
        cik=header.get("cik"),
        source_path=str(path),
    )
