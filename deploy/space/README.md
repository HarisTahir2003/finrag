---
title: finrag
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Ask questions about SEC 10-K filings and get cited answers
---

# finrag

Ask a question about a company's annual report and get an answer with the passage
it came from.

The corpus is 50 SEC 10-K filings — 10 companies across fiscal years 2022–2024 —
parsed, chunked and indexed into 12,376 passages. A question is answered by an
agent that searches the filings, reads what it finds, and does arithmetic with a
calculator rather than in its head.

**Try:** *What was Apple's total net sales in fiscal 2024?*

Every answer shows its sources. If a figure is not in the passages underneath it,
the answer is wrong and you can see that it is wrong — which is the point.

## What it does under the hood

- **Hybrid retrieval** — BM25 and vector search over one filing at a time, fused
  by reciprocal rank fusion, then reordered by a cross-encoder.
- **Filtered by construction** — a question about Apple FY2024 cannot retrieve a
  chunk from Microsoft FY2022; the filter is applied in the store, not hoped for.
- **A calculator tool** — margins and ratios are computed, not generated.

Measured on a labelled evaluation set: retrieval hit rate 1.00, MRR 0.824,
tool-path accuracy 1.00, RAGAS faithfulness 0.975.

## Notes

One company per question works best. The free Groq tier caps a request at 8,000
tokens per minute, and comparing two companies sends several filings' worth of
context at once.

If the daily quota runs out, the sidebar takes your own Groq API key. It is kept
for your browser session only, is never written to disk, and is not shared with
anyone else using this page.

Source: https://github.com/HarisTahir2003/finrag
