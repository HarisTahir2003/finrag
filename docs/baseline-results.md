# Baseline results (v0.1 — historical)

These are the numbers the original coursework notebooks actually produced, recovered from the
committed cell outputs before those outputs were stripped from the repository. They are recorded
here so the evidence survives, and so later work has something to be measured against.

> **These predate every fix in v0.2 and v0.3.** They were produced with incorrect fiscal years for
> six of the ten tickers, fixed-width chunking, and — for Part 1 — evaluation against gold context
> rather than the retriever. Treat them as a historical baseline, not as a description of the
> current system. Reproduce the old Part 1 measurement with `finrag eval ragas --gold-context`, and
> the honest one by dropping the flag.

**Read the caveats.** Two of the three modules measure something narrower than the headline number
suggests.

Only Part 3 is carried into this repository. Parts 1 and 2 are recorded here anyway, because the
chunking, prompting and evaluation choices in Part 3 came out of them, and because their numbers
are the ones most often quoted. The notebooks themselves stay in the
[original repo](https://github.com/HarisTahir2003/NLP_Applications_for_Financial_Reports).

---

## Part 1 — Extractive QnA on FinanceBench

150 rows from the `patronusai/financebench` train split. Gemini 2.5 Flash Lite as the generator,
Gemini 2.5 Flash as the RAGAS judge.

| Metric | Score |
|---|---|
| RAGAS faithfulness | **0.7393** |
| RAGAS answer relevancy | **0.4476** |
| RAGAS context precision | **0.7667** |
| Exact Match (recall-oriented) | **0.1133** |
| BERTScore F1 | 0.8779 |
| BERTScore precision | 0.8752 |
| BERTScore recall | 0.8820 |

> **Caveat — this is not an end-to-end RAG measurement.** The evaluation loop does
> `contexts.append([clean_context])`, where `clean_context` is FinanceBench's own gold evidence
> field. The Chroma retriever and the reciprocal-rank-fusion function defined earlier in the
> notebook are never called during evaluation. So faithfulness of 0.7393 describes how well the
> generator stays grounded when handed **perfect** context — a prompt-engineering result, not a
> retrieval one. A true end-to-end score would almost certainly be lower.
>
> Note also that **answer relevancy is 0.4476**, which is weak, and exact match is 0.1133. Exact
> match is a harsh metric for free-form financial answers, but neither figure should be omitted
> when the faithfulness number is quoted.

An earlier baseline of 0.18 faithfulness is referenced elsewhere but is **not reproducible from
this repository** — no committed output contains it.

## Part 2 — Summarization on ECTSum

Earnings-call transcripts from `mrSoul7766/ECTSum`, summarized with `gemini-2.0-flash`.

| Metric | Score |
|---|---|
| ROUGE-1 | 0.1413 |
| ROUGE-2 | 0.0503 |
| ROUGE-L | 0.0883 |
| BERTScore F1 | 0.8182 |
| G-Eval (LLM-as-judge, 1–5) | **4.39 / 5.0** |

> **Caveat.** The lexical overlap scores are low. ECTSum's reference summaries are terse and
> extractive while the generated summaries are discursive investment memos, so ROUGE penalises a
> stylistic difference rather than a factual one — but the 4.39/5.0 judge score should not be
> quoted on its own, because an LLM judge scoring free-form prose is the most generous of the three
> measurements here.

## Part 3 — Agentic analysis of SEC 10-K filings

50 filings (10 tickers × 5 years), `gemini-2.5-pro` as the reasoning engine, over a 20-question
evaluation set covering pure calculation, pure textual reasoning, and mixed tasks.

| Metric | Score |
|---|---|
| Functional tool call accuracy | **60.00%** (12 / 20 passed) |

> **Dominant failure mode.** On several questions the agent replies *"I need a stock ticker. Which
> company are you interested in?"* — for a ticker the evaluation set had already specified. The
> harness passes the question text alone, so the ticker never reaches the tool call. This is an
> evaluation-harness defect as much as a reasoning failure, and it drags the score down.

---

## Known correctness issue affecting Part 3

The fiscal year attached to every indexed chunk is derived from the **accession number**, which
encodes the year the filing was *submitted*, not the fiscal year it covers. For companies whose
fiscal year ends in December, the FY2022 10-K is filed in early 2023 and is therefore indexed as
`year: 2023`.

| Ticker | Fiscal year end | Correctly labelled? |
|---|---|---|
| AAPL, MSFT, V | Sep / Jun / Sep | Yes — coincidentally |
| NVDA | late Jan | Yes — coincidentally |
| AMZN, GOOGL, META, TSLA, NFLX, JPM | Dec | **No — off by one year** |

Any year-filtered retrieval against those six tickers returns the wrong report, which means the
Part 3 evaluation above was run against partly incorrect data. Scheduled for the next milestone.
