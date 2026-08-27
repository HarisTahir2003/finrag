# finrag

An agentic retrieval-augmented generation system over SEC 10-K filings. It downloads annual
reports straight from EDGAR, converts the filing HTML (tables included) into retrievable text,
indexes it into a vector store with per-company and per-year metadata, and puts a tool-calling
agent in front of it that can both look facts up and compute with them.

Ask it *"what was Apple's current ratio in 2023?"* and it retrieves the balance-sheet chunks,
extracts the two figures, and runs the division — rather than guessing a plausible-looking number,
which is the usual failure mode when an LLM is asked to do arithmetic on a document.

> **Status: v0.1 — research code being turned into a deployable system.**
> This began as a coursework notebook. It works, and its measured results are recorded in
> [`docs/baseline-results.md`](docs/baseline-results.md), but there is a known correctness bug and
> the evaluation does not yet measure what it should. Both are documented below and are the first
> items on the roadmap. Read [Known limitations](#known-limitations) before quoting any number
> from this repository.

## What is in here

| Path | What it is |
|---|---|
| `Part3.ipynb` | The system: EDGAR download, HTML-to-markdown table parsing, semantic indexing into Chroma, and a tool-calling agent with retrieval and calculator tools. |
| `app.py` | Streamlit chat interface over the agent. |
| `docs/baseline-results.md` | Measured results, with caveats. |

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
pip install -r requirements.txt

cp .env.example .env       # then fill in GOOGLE_API_KEY and SEC_CONTACT_EMAIL
```

Two environment variables matter:

- **`GOOGLE_API_KEY`** — for the Gemini models and Google embeddings.
  [Get one here](https://aistudio.google.com/app/apikey).
- **`SEC_CONTACT_EMAIL`** — EDGAR requires a real contact address in the User-Agent header of
  every request and rate-limits anonymous traffic.

`FINRAG_DATA_ROOT` controls where filings and the vector store are written. It defaults to
`./data`, which is gitignored. Nothing is stored outside the repository unless you point it
elsewhere.

## Running it

Work through `Part3.ipynb` in order: download filings, build the index, then query the agent.
The download and indexing steps are slow and cost money in embedding calls, but only need to run
once.

With an index built, the Streamlit app gives you a chat interface over the same agent:

```bash
streamlit run app.py
```

## Known limitations

Being explicit about these because two of them affect numbers that could otherwise be quoted out
of context.

**1. Fiscal year metadata is wrong for calendar-year companies.** The year attached to each chunk
comes from the accession number, which encodes when the filing was *submitted*, not the fiscal
year it covers. A December-year-end company files its FY2022 report in early 2023, so it is
indexed as 2023. This affects AMZN, GOOGL, META, TSLA, NFLX and JPM. AAPL, MSFT, V and NVDA happen
to be correct because their fiscal years end before December. Any year-filtered query against the
first group returns the wrong report.

**2. The RAGAS evaluation in Part 1 is not end-to-end.** It feeds FinanceBench's gold evidence in
as context instead of calling the retriever, so it measures how faithfully the generator uses
*perfect* context, not how the retrieval pipeline performs. The faithfulness figure of 0.7393
should be read that way.

**3. "Semantic element partitioning" is imported but not used.** `partition_html` and
`chunk_by_title` are imported in Part 3 and never called; the active path is BeautifulSoup
followed by fixed 3000-character splitting.

**4. The calculator tool executes model-generated Python.** It combines `eval()` with a Python
REPL tool. Fine in a local notebook, not acceptable for anything publicly reachable.

**5. Ingestion is not idempotent.** `Chroma.from_documents` is called inside the per-file loop, so
re-running the indexing step duplicates the corpus rather than updating it.

**6. There are no tests and no CI.**

## Roadmap

- **v0.2 — correctness and packaging.** Resolve fiscal year from the filing's
  `CONFORMED PERIOD OF REPORT` header. Move the notebook code into an installable `src/finrag/`
  package. Make ingestion idempotent with deterministic chunk IDs. Replace the calculator with an
  AST-whitelisted evaluator. Unit tests and CI.
- **v0.3 — honest evaluation.** Run RAGAS end-to-end through the retriever. Track every run with
  MLflow. Add a CI gate that fails the build when retrieval quality regresses.
- **v0.4 — deployment.** Dockerfile, FastAPI service, rewritten Streamlit app, live demo.

## Licence

MIT — see [LICENSE](LICENSE).
