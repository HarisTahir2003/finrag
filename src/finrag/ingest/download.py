"""Fetching 10-K filings from SEC EDGAR."""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import DEFAULT_TICKERS, Settings, get_settings

log = logging.getLogger(__name__)


def download_filings(
    tickers: tuple[str, ...] | list[str] = DEFAULT_TICKERS,
    years: int = 5,
    settings: Settings | None = None,
) -> dict[str, int]:
    """Download the last ``years`` 10-K filings for each ticker.

    Returns ticker -> number of filings on disk afterwards. Failures are logged
    and skipped rather than aborting the run, because one delisted or renamed
    ticker should not cost the other nine.
    """
    settings = settings or get_settings()
    settings.require_sec_contact()

    from sec_edgar_downloader import Downloader

    settings.filings_dir.mkdir(parents=True, exist_ok=True)
    downloader = Downloader(
        settings.sec_company_name, settings.sec_contact_email, str(settings.filings_dir)
    )

    counts: dict[str, int] = {}
    for ticker in tickers:
        try:
            downloader.get("10-K", ticker, limit=years)
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the batch
            log.warning("%s: download failed: %s", ticker, exc)
        counts[ticker] = len(list_filings(ticker, settings))
        log.info("%s: %d filings on disk", ticker, counts[ticker])
    return counts


def list_filings(ticker: str | None = None, settings: Settings | None = None) -> list[Path]:
    """Every downloaded full-submission file, optionally for one ticker."""
    settings = settings or get_settings()
    root = settings.filings_dir
    if not root.exists():
        return []
    pattern = (
        f"**/sec-edgar-filings/{ticker.upper()}/**/full-submission.txt"
        if ticker
        else "**/full-submission.txt"
    )
    return sorted(root.glob(pattern))
