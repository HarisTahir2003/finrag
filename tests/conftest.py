from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
FILINGS = FIXTURES / "sec-edgar-filings"


@pytest.fixture
def amzn_fy2022() -> Path:
    """The off-by-one case: fiscal year ends Dec 2022, filed Feb 2023."""
    return FILINGS / "AMZN" / "10-K" / "0000000000-23-000001" / "full-submission.txt"


@pytest.fixture
def aapl_fy2023() -> Path:
    """Fiscal year ends Sep 2023, filed Nov 2023 - the old code got this right by luck."""
    return FILINGS / "AAPL" / "10-K" / "0000000000-23-000001" / "full-submission.txt"


@pytest.fixture
def nvda_fy2024() -> Path:
    """Fiscal year ends late Jan 2024 and is labelled FY2024."""
    return FILINGS / "NVDA" / "10-K" / "0000000000-24-000001" / "full-submission.txt"


@pytest.fixture
def recursive_settings(monkeypatch, tmp_path):
    """Settings that avoid the optional unstructured dependency."""
    monkeypatch.setenv("FINRAG_CHUNK_STRATEGY", "recursive")
    monkeypatch.setenv("FINRAG_DATA_ROOT", str(tmp_path))
    from finrag.config import get_settings

    return get_settings()
