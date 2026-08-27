from __future__ import annotations

from bs4 import BeautifulSoup

from finrag.ingest.parse import (
    extract_primary_document,
    html_to_markdown_tables,
    html_to_text,
    parse_filing,
)


def test_extracts_the_10k_and_skips_exhibits(amzn_fy2022):
    html = extract_primary_document(amzn_fy2022.read_text(), form_type="10-K")
    assert html is not None
    assert "Management's Discussion" in html
    assert "SUBSIDIARIES OF THE REGISTRANT" not in html, "EX-21.1 must not be picked up"


def test_exact_type_match_does_not_confuse_ex_10_with_10k():
    raw = "<DOCUMENT>\n<TYPE>EX-10.1\n<TEXT>should not match</TEXT>\n</DOCUMENT>"
    assert extract_primary_document(raw, form_type="10-K") is None


def test_tables_become_markdown_rows():
    soup = BeautifulSoup(
        "<table><tr><td>Total current assets</td><td>143,566</td></tr></table>", "lxml"
    )
    text = html_to_markdown_tables(soup).get_text()
    assert "| Total current assets | 143,566 |" in text


def test_table_values_stay_attached_to_their_labels(amzn_fy2022):
    """The reason tables are converted at all: a bare number is unretrievable."""
    text = parse_filing(amzn_fy2022)
    assert "| Total net sales | 513,983 | 469,822 |" in text
    assert "| Total current assets | 146,791 | 161,580 |" in text


def test_scripts_and_styles_are_dropped():
    text = html_to_text("<html><body><script>var x=1;</script><p>Real text</p></body></html>")
    assert "var x" not in text
    assert "Real text" in text


def test_unknown_form_returns_none(aapl_fy2023):
    assert extract_primary_document(aapl_fy2023.read_text(), form_type="10-Q") is None


def test_strip_ixbrl_wrapper_removes_the_xbrl_preamble():
    """Real 10-Ks do not begin at <html>, and one partitioner cares a great deal."""
    from finrag.ingest.parse import strip_ixbrl_wrapper

    wrapped = (
        "\n<XBRL>\n<?xml version='1.0' encoding='ASCII'?>\n"
        "<!--XBRL Document Created with the Workiva Platform-->\n"
        '<html xmlns:link="http://www.xbrl.org/2003/linkbase"><body><p>Revenue</p></body></html>'
    )
    out = strip_ixbrl_wrapper(wrapped)
    assert out.startswith("<html")
    assert "<XBRL>" not in out
    assert "Revenue" in out, "the document itself must survive intact"


def test_strip_ixbrl_wrapper_leaves_plain_html_alone():
    from finrag.ingest.parse import strip_ixbrl_wrapper

    plain = "<html><body><p>x</p></body></html>"
    assert strip_ixbrl_wrapper(plain) == plain


def test_strip_ixbrl_wrapper_passes_through_documents_with_no_html_tag():
    from finrag.ingest.parse import strip_ixbrl_wrapper

    assert strip_ixbrl_wrapper("no markup at all") == "no markup at all"
