# finrag

An agentic retrieval-augmented generation system over SEC 10-K filings. It downloads annual
reports straight from EDGAR, converts the filing HTML (tables included) into retrievable text,
indexes it into a vector store with per-company and per-year metadata, and puts a tool-calling
agent in front of it that can both look facts up and compute with them.

Ask it *"what was Apple's current ratio in 2023?"* and it retrieves the balance-sheet chunks,
extracts the two figures, and runs the division — rather than guessing a plausible-looking number,
which is the usual failure mode when an LLM is asked to do arithmetic on a document.

> **Status: v0.2 — packaged and correct; evaluation is next.**
> This began as a coursework notebook. The logic now lives in an installable, tested package, and
> the fiscal-year bug that corrupted six of the ten default tickers is fixed. What has *not* been
> redone is the evaluation: the numbers in [`docs/baseline-results.md`](docs/baseline-results.md)
> were produced by the old pipeline and one of them does not measure what it appears to. See
> [Known limitations](#known-limitations) before quoting any figure from this repository.

## What is in here

| Path | What it is |
|---|---|
| `src/finrag/config.py` | Environment-driven settings. Every value has a working default. |
| `src/finrag/ingest/metadata.py` | Fiscal-year resolution from the SEC submission header. |
| `src/finrag/ingest/parse.py` | Extracts the 10-K from a submission, renders tables as markdown. |
| `src/finrag/ingest/index.py` | Chunking and idempotent upsert into Chroma. |
| `src/finrag/chunking.py` | Structure-aware or fixed-width chunking. |
| `src/finrag/embeddings.py` | Local (sentence-transformers) or Google embedding backends. |
| `src/finrag/retrieval.py` | Filtered search, query expansion, reciprocal rank fusion. |
| `src/finrag/calculator.py` | AST-whitelisted arithmetic — the agent's calculator tool. |
| `src/finrag/agent.py` | Tool-calling agent over retrieval + calculator. |
| `src/finrag/cli.py` | `finrag download / index / status / ask`. |
| `Part3.ipynb` | Narrative walkthrough of the pipeline, importing from the package. |
| `app.py` | Streamlit chat interface. |
| `tests/` | 70 tests over parsing, fiscal years, chunk identity and calculator safety. |

The original coursework project had two further modules — extractive QnA over FinanceBench, and
earnings-call summarization on ECTSum. Neither is part of this repository, which is the SEC filing
agent only. Both remain in the
[original repo](https://github.com/HarisTahir2003/NLP_Applications_for_Financial_Reports), and
their results are included in `docs/baseline-results.md` because they are what the retrieval and
prompting choices here were derived from.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/HarisTahir2003/finrag.git
cd finrag

python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[local,semantic,google,app,dev]"

cp .env.example .env       # then fill in SEC_CONTACT_EMAIL, and GOOGLE_API_KEY for the agent
```

The default embedding backend is **local** (`sentence-transformers/all-MiniLM-L6-v2`), so
downloading, indexing, retrieval and the whole test suite run with **no API key and no per-call
cost**. A key is only needed to download from EDGAR (a contact address, not a paid key) and to run
the agent itself. Set `FINRAG_EMBEDDINGS=google` for the higher-quality embedding path.

Two environment variables matter:

- **`GOOGLE_API_KEY`** — for the Gemini models and Google embeddings.
  [Get one here](https://aistudio.google.com/app/apikey).
- **`SEC_CONTACT_EMAIL`** — EDGAR requires a real contact address in the User-Agent header of
  every request and rate-limits anonymous traffic.

`FINRAG_DATA_ROOT` controls where filings and the vector store are written. It defaults to
`./data`, which is gitignored. Nothing is stored outside the repository unless you point it
elsewhere.

## Running it

```bash
finrag download --tickers AAPL,AMZN --years 2   # fetch from EDGAR
finrag index                                    # parse, chunk, embed (idempotent)
finrag status                                   # configuration and index size
finrag ask "What was Apple's current ratio in fiscal 2023?"
```

Indexing is safe to re-run: chunk IDs are derived from ticker, fiscal year, position and content,
so unchanged filings are rewritten in place rather than duplicated.

`Part3.ipynb` walks through the same steps with commentary, and the Streamlit app gives you a chat
interface over the agent:

```bash
streamlit run app.py
```

### Tests

```bash
pytest
```

70 tests, no API key and no network required — they run against committed SEC-format fixtures.

## Known limitations

**1. The evaluation does not measure the retrieval pipeline.** The RAGAS run recorded in
`docs/baseline-results.md` feeds FinanceBench's gold evidence in as context instead of calling the
retriever, so faithfulness of 0.7393 describes how faithfully the generator uses *perfect* context
— a prompt-engineering result, not a retrieval one. Nothing in this repository currently produces
an honest end-to-end number. This is the whole of the next milestone.

**2. The recorded numbers predate the fixes.** They were produced by the old pipeline, including
the incorrect fiscal years, and with fixed-width rather than structure-aware chunking. They are
kept as a baseline to improve on, not as a description of the current system.

**3. Coverage is uneven.** Parsing, fiscal-year resolution, chunk identity and calculator safety
are well covered. The agent, CLI and retrieval modules are not, because they need a live model or
a populated index; they are exercised by the evaluation harness instead, which is v0.3 work.

**4. Fiscal-year labelling assumes the common convention.** The fiscal year is taken to be the
calendar year in which the period ends. This is correct for all ten default tickers, but some
retailers label a year ending in early February as the *previous* fiscal year. Add such companies
to `FISCAL_YEAR_OVERRIDES` in `src/finrag/ingest/metadata.py`.

## Roadmap

- ~~**v0.2 — correctness and packaging.**~~ **Done.** Fiscal year resolved from
  `CONFORMED PERIOD OF REPORT`. Notebook code moved into an installable `src/finrag/` package.
  Ingestion made idempotent with deterministic chunk IDs. Calculator replaced with an
  AST-whitelisted evaluator. 70 unit tests and CI on Python 3.10 and 3.12.
- **v0.3 — honest evaluation.** Run RAGAS end-to-end through the retriever. Track every run with
  MLflow. Add a CI gate that fails the build when retrieval quality regresses.
- **v0.4 — deployment.** Dockerfile, FastAPI service, rewritten Streamlit app, live demo.

## Licence

MIT — see [LICENSE](LICENSE).
