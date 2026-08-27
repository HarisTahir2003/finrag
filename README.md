# finrag

An agentic retrieval-augmented generation system over SEC 10-K filings. It downloads annual
reports straight from EDGAR, converts the filing HTML (tables included) into retrievable text,
indexes it into a vector store with per-company and per-year metadata, and puts a tool-calling
agent in front of it that can both look facts up and compute with them.

Ask it *"what was Apple's current ratio in 2023?"* and it retrieves the balance-sheet chunks,
extracts the two figures, and runs the division — rather than guessing a plausible-looking number,
which is the usual failure mode when an LLM is asked to do arithmetic on a document.

> **Status: v0.3 — evaluated, gated in CI.**
> This began as a coursework notebook. The logic now lives in an installable, tested package; the
> fiscal-year bug that corrupted six of the ten default tickers is fixed; and evaluation runs
> **through the retriever** rather than around it. A retrieval quality gate runs on every pull
> request with no API key and no cost. The historical numbers in
> [`docs/baseline-results.md`](docs/baseline-results.md) are kept as a baseline and clearly marked
> as measuring something narrower than they appear to.

## What is in here

| Path | What it is |
|---|---|
| `src/finrag/config.py` | Environment-driven settings. Every value has a working default. |
| `src/finrag/ingest/metadata.py` | Fiscal-year resolution from the SEC submission header. |
| `src/finrag/ingest/parse.py` | Extracts the 10-K from a submission, renders tables as markdown. |
| `src/finrag/ingest/index.py` | Chunking and idempotent upsert into Chroma. |
| `src/finrag/chunking.py` | Structure-aware or fixed-width chunking. |
| `src/finrag/embeddings.py` | Local (sentence-transformers) or Google embedding backends. |
| `src/finrag/llm.py` | Chat backends — commercial, hosted open-weight, or local — with per-tier rate limiting and fallback chains. |
| `src/finrag/cache.py` | SQLite LLM response cache: unchanged re-runs cost zero tokens. |
| `src/finrag/eval/checkpoint.py` | Per-case eval checkpointing, so a run survives a daily quota. |
| `src/finrag/retrieval.py` | Filtered search, query expansion, reciprocal rank fusion. |
| `src/finrag/calculator.py` | AST-whitelisted arithmetic — the agent's calculator tool. |
| `src/finrag/agent.py` | Tool-calling agent over retrieval + calculator. |
| `src/finrag/eval/` | Three evaluation tiers, and the CI quality gate. |
| `src/finrag/eval/datasets/` | Evaluation sets as YAML, not buried in a notebook cell. |
| `src/finrag/cli.py` | `finrag download / index / status / ask / eval`. |
| `Part3.ipynb` | Narrative walkthrough of the pipeline, importing from the package. |
| `app.py` | Streamlit chat interface. |
| `tests/` | 92 tests over parsing, fiscal years, chunk identity, calculator safety and the gate. |

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
pip install -e ".[local,semantic,anthropic,app,dev]"   # swap in [groq], [ollama] or [google]

cp .env.example .env       # then fill in SEC_CONTACT_EMAIL and one provider key
```

The default embedding backend is **local** (`sentence-transformers/all-MiniLM-L6-v2`), so
downloading, indexing, retrieval and the whole test suite run with **no API key and no per-call
cost**. A key is only needed to download from EDGAR (a contact address, not a paid key) and to run
the agent itself. Set `FINRAG_EMBEDDINGS=google` for the higher-quality embedding path.

Two environment variables matter:

- **One provider key.** `ANTHROPIC_API_KEY` by default, or `GOOGLE_API_KEY` with
  `FINRAG_LLM_BACKEND=google`. Only the chat model needs a provider; embeddings are separate.
- **`SEC_CONTACT_EMAIL`** — EDGAR requires a real contact address in the User-Agent header of
  every request and rate-limits anonymous traffic.

### Providers and cost

The chat backend and the embedding backend are chosen independently, which matters because
Anthropic publishes no embedding model. Embeddings default to local sentence-transformers, so
**indexing all fifty filings costs nothing** whichever chat provider you use — the only billable
calls are answering questions and running the LLM-scored evaluations.

**Open-weight models, hosted** — someone else's hardware, nothing on yours:

| `FINRAG_LLM_BACKEND` | Default model | Key |
|---|---|---|
| `cerebras` | `gpt-oss-120b` | `CEREBRAS_API_KEY` |
| `groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| `github` | `openai/gpt-4o-mini` | `GITHUB_TOKEN` |
| `openrouter` | `meta-llama/llama-3.3-70b-instruct` | `OPENROUTER_API_KEY` |
| `together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | `TOGETHER_API_KEY` |
| `fireworks` | `llama-v3p3-70b-instruct` | `FIREWORKS_API_KEY` |
| `deepinfra` | `meta-llama/Llama-3.3-70B-Instruct` | `DEEPINFRA_API_KEY` |

All of these speak the OpenAI chat-completions protocol, so they share one client and differ only
by base URL, default model and key variable. Adding another — or pointing at a self-hosted vLLM or
LM Studio server via `FINRAG_OPENAI_BASE_URL` — is one entry in `PROVIDER_PRESETS`, not one more
backend.

**Open-weight models, local** — `ollama`, default `qwen3:4b`. No key, no rate limit, no network,
no cost. Bounded by RAM: a 4B model at 4-bit wants roughly 3GB, an 8B closer to 6GB, and retrieval
context adds more.

**Commercial APIs** — `anthropic` (`claude-haiku-4-5-20251001`) and `google` (`gemini-2.5-flash`).
Most reliable tool calling, paid per token.

`FINRAG_EMBEDDINGS` is separate and defaults to `local`, so **indexing is free on every one of
these**. Only answering questions and the LLM-scored evaluations are billable, and on the two
open-weight backends not even those.

### Running the whole thing at $0

The pipeline is built to run end to end — ingest, agent, both LLM-scored evaluations, CI — on free
tiers alone, and the machinery that makes that *reliable* rather than lucky is part of the
codebase:

| Free-tier failure mode | What handles it |
|---|---|
| Per-minute rate caps (429s) | A client-side token-bucket limiter paces every call under each tier's published RPM — defaults per backend in `DEFAULT_RPM`, override with `FINRAG_RPM`. Never hitting the limit beats recovering from it. |
| Request-size ceilings (Cerebras and GitHub Models enforce ~8K per request as a hard 400) | Retrieved context is trimmed to a token budget before it reaches the model (`FINRAG_MAX_CONTEXT_TOKENS=auto`), dropping the lowest-ranked whole chunks instead of failing the call. |
| Daily quotas dying mid-run | Every completed evaluation case is checkpointed to `results/<suite>-cases.jsonl` as it finishes; rerun with `--resume` and only the unfinished cases spend quota. Failed cases are deliberately not checkpointed, so they retry. |
| Re-running while iterating | Identical LLM calls are served from a SQLite cache (`data/llm_cache.db`, on by default) — an unchanged re-run costs zero tokens. RAGAS judge calls benefit most. |
| One provider's quota too small | `FINRAG_LLM_FALLBACKS=groq,openrouter` chains providers on the plain-chat paths, making the usable budget the union of the tiers. |
| RAGAS's own concurrency | The judge runs with `RunConfig(max_workers=1)` plus bounded retries — the default of 16 concurrent workers is a guaranteed 429 on any free tier. |

The split that works: **Cerebras** for batch evaluation (largest daily token budget), **Groq** for
interactive asking (fastest), **GitHub Models** for CI (the built-in `GITHUB_TOKEN` can call it
once the workflow requests `permissions: models: read` — the `llm-smoke` job runs a real agent
evaluation on every push with zero repository secrets), and **Ollama** offline. None needs a card.

Three things to know when running open-weight models:

- **Free tiers cap tokens per minute**, and a default retrieval of 20 chunks will breach most of
  them. Set `FINRAG_RETRIEVAL_K=6` or lower.
- **Tool calling is the thing to test first**, not general quality. This agent must call a
  retrieval tool and a calculator; a model that silently drops a tool call answers from memory and
  invents a figure, which is worse than refusing. Reliability varies a lot across open models, and
  free variants on aggregators are often the weakest. `finrag eval agent --fixtures --dataset
  smoke` is the three-question check built for exactly this.
- **Ollama truncates prompts** longer than `num_ctx` without reporting it, which presents as the
  model ignoring its context rather than never having received it. `FINRAG_OLLAMA_NUM_CTX` defaults
  to 16384.

Retrieval depth is the main cost and rate-limit lever throughout: `FINRAG_RETRIEVAL_K=8` roughly
halves tokens per call against the default of 20.

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

### Evaluation

Three tiers, separated by what they cost to run.

```bash
finrag eval retrieval --fixtures --gate   # no LLM, no API key, free — this is the CI gate
finrag eval ragas --limit 5               # needs GOOGLE_API_KEY
finrag eval agent                         # needs GOOGLE_API_KEY
```

**Retrieval** is scored by exact substring matching against a labelled set, so it is deterministic
and costs nothing. It reports hit rate, MRR, and `filter_accuracy` — the fraction of retrieved
chunks whose ticker and fiscal year actually match the query. That last one is the direct
regression guard on the metadata bug: under the old indexing it would read 0 for six tickers.

`--fixtures` indexes the committed test filings into a throwaway store, so the gate runs on a fresh
clone with nothing downloaded. Current measured performance on that corpus:

| metric | value |
|---|---|
| hit_rate | 1.000 |
| mrr | 0.926 |
| filter_accuracy | 1.000 |

**RAGAS** scores faithfulness, answer relevancy and context precision with contexts drawn from the
retriever. Pass `--gold-context` to reproduce the original measurement, which fed the dataset's own
gold evidence in as context and so never exercised retrieval at all. The gap between the two runs
is the retrieval contribution, which was previously invisible.

**Agent** runs the full tool-calling loop over the 20-question set and scores tool-path accuracy,
calculator compliance, and how often the agent asks for a ticker it was already given — the
dominant failure mode in the original run, tracked separately because it is a prompt problem rather
than a reasoning one.

Every run is logged to `results/` as JSON, and to MLflow when it is installed, alongside the
configuration that produced it.

### Tests

```bash
pytest
```

92 tests, no API key and no network required — they run against committed SEC-format fixtures.

## Known limitations

**1. The gate runs on a small synthetic corpus.** Three fixture filings, chunked small so ranking is
actually exercised. It will catch a broken filter, a broken parser or a serious retrieval
regression. It will not catch subtle quality loss, because nine probes over ten chunks cannot.
Widening it needs a labelled set over real filings, which is the obvious next step.

**2. The historical numbers predate every fix.** `docs/baseline-results.md` was produced with
incorrect fiscal years, fixed-width chunking and gold-context evaluation. It is kept as a baseline
to beat, not as a description of the current system. The end-to-end RAGAS figure has not yet been
run over the full ten-ticker corpus.

**3. Answer correctness is not scored automatically.** The agent suite scores tool paths and
whether an answer contains figures at all. Judging whether an answer is *right* still needs the
RAGAS tier and, for the narrative questions, a human.

**4. Fiscal-year labelling assumes the common convention.** The fiscal year is taken to be the
calendar year in which the period ends. Correct for all ten default tickers, but some retailers
label a year ending in early February as the *previous* fiscal year. Add them to
`FISCAL_YEAR_OVERRIDES` in `src/finrag/ingest/metadata.py`.

## Roadmap

- ~~**v0.2 — correctness and packaging.**~~ **Done.** Fiscal year resolved from
  `CONFORMED PERIOD OF REPORT`. Notebook code moved into an installable `src/finrag/` package.
  Ingestion made idempotent with deterministic chunk IDs. Calculator replaced with an
  AST-whitelisted evaluator. 70 unit tests and CI on Python 3.10 and 3.12.
- ~~**v0.3 — honest evaluation.**~~ **Done.** RAGAS runs end-to-end through the retriever, with a
  `--gold-context` mode to reproduce the original measurement for comparison. Evaluation sets moved
  to YAML. Runs tracked to JSON and MLflow with their configuration. A retrieval quality gate runs
  in CI with no API key.
- **v0.4 — deployment.** Dockerfile, FastAPI service, rewritten Streamlit app, live demo.

## Licence

MIT — see [LICENSE](LICENSE).
