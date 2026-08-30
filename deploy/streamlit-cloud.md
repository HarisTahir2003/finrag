# Deploying to Streamlit Community Cloud

Free permanently, no credit card, GitHub login only. This is the host this repo
targets for a public demo.

Hugging Face Spaces was the first choice and is ruled out: Gradio and Docker
Spaces now "require a paid plan to create: PRO for personal accounts"
(https://huggingface.co/docs/hub/en/spaces-overview), and the Streamlit SDK was
removed. `Dockerfile.space` still exists and still works — it is the right
artifact for any Docker host, and this guide does not use it, because Community
Cloud does not build images. It clones the repo, installs `requirements.txt`,
and runs `app.py`.

## What makes this work

**The index is committed, compressed.** 134MB of Chroma does not go in git —
GitHub blocks any single file over 100MB. `data/chroma_local.tar.xz` is 45MB,
which is an ordinary git object, and `finrag.bootstrap.ensure_index` unpacks it
on first boot in about two seconds.

Git LFS would be the conventional answer and is the wrong one here: these
platforms do not reliably fetch LFS objects, and the failure mode is a 130-byte
pointer file arriving where a database was expected — which surfaces much later
as an empty index rather than as a download error.

**torch is pinned to a CPU wheel.** See the comment at the top of
`requirements.txt`. This is the line that decides whether the build succeeds.

## Steps

1. **Push.** Everything needed is already in the repo.

2. Go to **https://share.streamlit.io** and sign in with GitHub. Authorise it to
   read your repositories.

3. **Create app** → **Deploy a public app from GitHub**, and fill in:

   | Field | Value |
   |---|---|
   | Repository | `HarisTahir2003/finrag` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | Python version (under *Advanced settings*) | **3.12** |

   3.12 is not optional. The torch wheel pinned in `requirements.txt` is
   `cp312`; on any other version the install fails with "not a supported wheel
   on this platform".

4. Still under **Advanced settings**, paste into **Secrets**:

   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   FINRAG_LLM_BACKEND = "groq"
   FINRAG_RETRIEVAL_K = "20"
   FINRAG_RERANK = "false"

   # Spending limits. The key above is shared by everyone who finds the URL,
   # and one answer costs several Groq requests, so the daily cap is set well
   # below the 1,000-request free tier to leave room for retries: ~250 answers
   # at ~3 requests each is ~750, with headroom. A visitor who wants more can
   # enter their own key -- that spends their quota, not yours.
   FINRAG_QUESTIONS_PER_SESSION = "20"
   FINRAG_QUESTIONS_PER_DAY = "250"

   # Each retry is another request against the daily quota, so one rate-limited
   # question can cost several. The default of 6 is right for the backend
   # comparison and wrong here.
   FINRAG_MAX_RETRIES = "2"

   # Pace client-side requests. The default of 4/min is ~34% over Groq's 8,000
   # tokens-per-minute at real financial-chunk sizes; 3 keeps a two-search
   # question inside one minute's budget.
   FINRAG_RPM = "3"
   ```

   Secrets are not committed and are not visible to visitors. `app.py` copies
   them into the environment at startup, because every module in `finrag` reads
   `os.environ` and none of them should have to know about Streamlit.

5. **Deploy.** The first build takes several minutes, mostly installing torch.

## The "more than one requirements file" warning

The build log says:

```
WARN: More than one requirements file detected in the repository.
Python dependencies were installed from /mount/src/finrag/requirements.txt
```

It is harmless here, and it is worth knowing why rather than ignoring it.
Community Cloud picks exactly one dependency file, by a fixed precedence,
searching the entrypoint's directory and then the repository root:

    uv.lock  >  Pipfile  >  environment.yml  >  requirements.txt  >  pyproject.toml

Both `requirements.txt` and `pyproject.toml` are present, and `requirements.txt`
outranks it, so the right one wins every time -- by documented precedence, not
by luck. That matters more than it looks: Community Cloud reads a
`pyproject.toml` as a **Poetry** file, and this project's is setuptools, so
being chosen would not install the wrong extras, it would fail outright.

Adding a `Pipfile`, `environment.yml` or `uv.lock` later would silently outrank
`requirements.txt`, and the CPU-only torch pin lives there alone -- the build
would start pulling the CUDA wheel. `tests/test_requirements.py` fails if any
of those files appears.

## Why `FINRAG_RERANK = "false"`

Community Cloud guarantees a *minimum* of 690MB per app and allows up to 2.7GB
on a shared node. Measured peak RSS inside a Linux container on the real index:

| configuration | peak |
|---|---|
| rerank off | **690 MB** |
| rerank on, `FINRAG_RERANK_CANDIDATES=25` | 1036 MB |
| rerank on (default 50 candidates) | 1135 MB |

`import torch` alone is 493MB of that, and the cross-encoder accounts for
essentially all the rest.

Reranking costs real quality — MRR 0.824 with it, 0.720 without, because
reranking is what promotes the right chunk to first place. The reason to ship
with it off anyway is that exceeding the ceiling shows the visitor
"😦 Oh no. Error running app." with nothing in the logs, and a recruiter who
sees that has seen the whole project. Start without it; if the app is stable,
delete the line and watch.

Do **not** lower `FINRAG_RETRIEVAL_K` to save memory. k=15 measured RAGAS
faithfulness 0.650 against k=20's 0.975 — see the comment in `config.py`.

## Spending limits

Groq's free tier is 1,000 requests a day across **everyone who finds the URL**,
and one agent answer costs several of them -- a search, a read, sometimes a
calculation. A visitor holding down a starter pill can end the day for
everybody without meaning any harm and without seeing that they did it.

Two limits, because they fail differently. The per-session one stops the
ordinary case: someone curious, clicking repeatedly. The daily one stops what
the session limit cannot see -- many browsers, or one person who reloads to get
a fresh session.

Neither applies to a visitor who has entered their own key: that is their quota,
not yours, and rationing it would be rationing their money. It is also what
makes the limits tolerable -- when the shared budget is gone there is still a
way to use the demo, and every refusal says so.

This is not authentication. It slows an accident down; nothing in a public demo
with a shared key stops someone determined.

Set either variable to `"0"` to disable that scope.

## Reading the logs

Every question logs one line when it starts and one when it finishes, sharing a
request id:

```
INFO finrag.app: q start id=c48b85dd session=d1061013288b backend=groq chars=41
INFO finrag.app: q done  id=c48b85dd elapsed=16.4s steps=1 answer_chars=132 timed_out=False
```

The session id is random, generated in the browser session, and never leaves
the process. It identifies a rate-limit bucket and a log line, not a person.

A question that runs past `FINRAG_ANSWER_TIMEOUT_SECONDS` (default 120) is
abandoned between agent steps and the reader is told, rather than watching a
spinner forever.

## What to expect

- **First question is slow**, 30–60s. There is no `finrag warmup` step here, so
  a fresh container downloads ~170MB of model weights from HuggingFace on first
  use. Every question after that is normal.
- **The app sleeps** after about 12 hours without traffic. Any visitor wakes it
  with one click.
- **The daily Groq quota** is 1,000 requests. When it runs out the app tells
  visitors to enter their own key, which is kept for their session only.

## After re-indexing

The deployed app serves whatever is in the committed archive, so rebuild it:

```python
python -c "from finrag.bootstrap import pack_index; from finrag.config import get_settings; pack_index(get_settings())"
```

then commit `data/chroma_local.tar.xz`. Use that rather than a shell `tar`: on
macOS, `tar` stores extended attributes as AppleDouble members (`._chroma.sqlite3`)
which then extract onto a Linux host that has no idea what they are.

Each re-index adds ~45MB to git history permanently. If the corpus starts
changing often, move the archive to a GitHub Release and download it in
`ensure_index` instead.
