"""Turning a raw EDGAR submission into text worth retrieving.

A full-submission.txt bundles every exhibit into one file. Only the 10-K itself
is wanted; the exhibits are mostly legal boilerplate that dilutes retrieval.
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup

_DOC_SPLIT = "<DOCUMENT>"
_TYPE_RE = re.compile(r"<TYPE>\s*([^\s<]+)", re.I)


def html_to_markdown_tables(soup: BeautifulSoup) -> BeautifulSoup:
    """Replace every <table> with a markdown rendering of the same rows.

    Financial statements are tables, and ``get_text()`` on a table produces a
    column of orphaned numbers with no way to tell which line item they belong
    to. Rendering rows as pipe-delimited text keeps a value attached to its
    label, which is what makes the numbers retrievable at all.
    """
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            values = [c.get_text(strip=True).replace("|", "-") for c in cells]
            if any(values):
                rows.append("| " + " | ".join(values) + " |")
        table.replace_with("\n" + "\n".join(rows) + "\n\n" if rows else "\n")
    return soup


def extract_primary_document(raw: str, form_type: str = "10-K") -> str | None:
    """Return the HTML of the primary document, skipping exhibits.

    Matched on an exact <TYPE> so that a 10-K is not confused with EX-10.1, and
    so amendments (10-K/A) are only picked up when explicitly asked for.
    """
    wanted = form_type.upper()
    for block in raw.split(_DOC_SPLIT)[1:]:
        match = _TYPE_RE.search(block[:2000])
        if not match or match.group(1).upper() != wanted:
            continue
        start = block.find("<TEXT>")
        end = block.find("</TEXT>")
        if start != -1 and end != -1 and end > start:
            return block[start + len("<TEXT>") : end]
    return None


_HTML_START = re.compile(r"<html\b", re.IGNORECASE)


def strip_ixbrl_wrapper(html: str) -> str:
    """Return the document from its opening <html> tag, discarding any preamble.

    Every 10-K filed through a modern agent is inline XBRL, and the primary
    document arrives wrapped:

        <XBRL>
        <?xml version='1.0' encoding='ASCII'?>
        <!--XBRL Document Created with the Workiva Platform-->
        <html xmlns:link="http://www.xbrl.org/2003/linkbase" ...>

    BeautifulSoup shrugs at that, which is why the text path never noticed. An
    HTML partitioner does not: it reads the `<XBRL>` element and the XML
    declaration, concludes the document is XML rather than HTML, and returns
    zero elements -- instantly, without error, at any document size. Two
    hundred and fifty bytes of preamble are the whole difference between 557
    elements and none.

    Documents that already start at <html>, and anything with no <html> tag at
    all, are returned unchanged.
    """
    match = _HTML_START.search(html)
    return html[match.start() :] if match else html


def html_to_text(html: str) -> str:
    """Flatten filing HTML to text, preserving tables as markdown."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    soup = html_to_markdown_tables(soup)
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_filing(file_path: str | Path, form_type: str = "10-K") -> str | None:
    """Read a full-submission file and return the primary document as text."""
    raw = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    html = extract_primary_document(raw, form_type=form_type)
    if html is None:
        return None
    return html_to_text(html)
