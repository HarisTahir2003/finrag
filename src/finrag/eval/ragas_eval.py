"""End-to-end RAGAS evaluation.

This module exists to correct a specific mistake. The original evaluation built
its context like this::

    clean_context = clean_financial_text_smart(row["evidence"][0]["evidence_text"])
    ...
    contexts.append([clean_context])

``row["evidence"]`` is FinanceBench's own gold evidence. The retriever was never
called. So the reported faithfulness of 0.7393 measured how faithfully the
generator used **perfect** context -- a prompt-engineering result -- while being
described as a retrieval-augmented generation score. The reciprocal rank fusion
function defined earlier in that notebook never ran during evaluation at all.

Here the contexts come from ``search_filing``. The number this produces is
therefore lower and means something: it is what the whole pipeline achieves,
retrieval included.

Set ``use_gold_context=True`` for the oracle comparison. That gap is the point:
it is the retrieval contribution, which was previously invisible.

"Oracle" needs defining, because the first implementation of it here was
worthless. It passed ``case.reference_answer`` as the context -- a terse string
like "0.988 (current assets $143,566M / current liabilities $145,308M)" -- and
scored the generator on its ability to answer from that. Every metric it
produced was meaningless: context_precision was 1.0 by construction, since the
context *was* the reference the metric compares against, and faithfulness came
out at 0.445 because a model asked to answer from sixty characters elaborates
past what they support. It looked like an upper bound and was closer to a
lower one.

Oracle context is now what a perfect retriever would have returned: the indexed
chunks of the real filing that provably contain the answer's figures. Cases
whose reference carries no figures -- the narrative ones -- cannot be scored
this way and are skipped rather than fabricated, so a gold run reports on
fewer cases than a retrieved run. Compare like with like: the ``samples`` count
is in both reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import Settings, get_settings
from ..retrieval import search_filing
from .schema import AgentCase, load_agent_cases, reference_figures

log = logging.getLogger(__name__)

ANSWER_PROMPT = """You are a financial analyst. Answer the question using only the context below.

Rules:
- Quote figures exactly as they appear in the context.
- If the context does not contain the answer, say so plainly. Do not estimate.
- Be concise: the figure, its units, and one sentence of explanation.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RagasSample:
    question: str
    answer: str
    contexts: list[str]
    reference: str
    context_source: str  # "retrieved" or "gold"


@dataclass
class RagasReport:
    samples: list[RagasSample] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    context_source: str = "retrieved"
    # Which model scored these answers. Recorded because a RAGAS number is only
    # comparable against another produced by the same judge.
    judge: str = ""

    def as_metrics(self) -> dict[str, float | str]:
        metrics: dict[str, float | str] = {
            f"ragas_{k}": round(v, 4) for k, v in self.scores.items()
        }
        metrics["samples"] = len(self.samples)
        metrics["judge"] = self.judge
        return metrics


def _generate_answer(llm, question: str, context: str) -> str:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    chain = ChatPromptTemplate.from_template(ANSWER_PROMPT) | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question}).strip()


def build_samples(
    cases: list[AgentCase],
    llm,
    store=None,
    settings: Settings | None = None,
    use_gold_context: bool = False,
    checkpoint_path=None,
) -> list[RagasSample]:
    """Generate an answer per case, with context from the retriever by default.

    Generation is the per-case, quota-hungry half of the evaluation, so with
    ``checkpoint_path`` set each completed sample is persisted and a rerun
    regenerates only what is missing. (The scoring half is protected by the
    LLM response cache instead -- judge calls repeat verbatim across reruns.)
    """
    from .checkpoint import Checkpoint

    settings = settings or get_settings()
    source = "gold" if use_gold_context else "retrieved"
    checkpoint = Checkpoint(checkpoint_path)
    samples: list[RagasSample] = []

    for case in cases:
        done = checkpoint.completed(case.id)
        if done is not None and done.get("context_source") == source:
            samples.append(
                RagasSample(
                    question=done["question"],
                    answer=done["answer"],
                    contexts=list(done["contexts"]),
                    reference=done["reference"],
                    context_source=source,
                )
            )
            continue

        if use_gold_context:
            contexts = oracle_contexts(case, store)
            if not contexts:
                log.warning(
                    "%s: no oracle context derivable (its reference reports no figures); "
                    "skipping rather than scoring against a fabricated context",
                    case.id,
                )
                continue
        else:
            found = search_filing(
                case.question, case.ticker, case.fiscal_year, store=store, settings=settings
            )
            contexts = [d.page_content for d in found.documents]
            if not contexts:
                log.warning("%s: retrieved nothing; scoring against empty context", case.id)

        answer = _generate_answer(llm, case.question, "\n\n".join(contexts))
        sample = RagasSample(
            question=case.question,
            answer=answer,
            contexts=contexts or [""],
            reference=case.reference_answer,
            context_source=source,
        )
        samples.append(sample)
        checkpoint.record(
            case.id,
            {
                "question": sample.question,
                "answer": sample.answer,
                "contexts": sample.contexts,
                "reference": sample.reference,
                "context_source": source,
            },
        )
        log.info("%s: answered from %d context chunks", case.id, len(contexts))

    return samples


def oracle_contexts(case, store) -> list[str]:
    """The chunks a perfect retriever would have returned for this case.

    Selected by the figures the reference answer reports, so membership is
    verifiable rather than judged: a chunk is in the oracle set if it literally
    contains the number the answer depends on. Returns an empty list when the
    reference carries no figures, which is how narrative cases fall out.

    Shares ``reference_figures`` with the dataset validator deliberately. The
    two have to agree on what counts as a figure: this selects the chunks that
    contain one, and the validator asserts such chunks exist. If they drifted
    apart, a case could validate and then have no oracle context.
    """
    figures = reference_figures(case.reference_answer)
    if not figures:
        return []
    where = {"$and": [{"ticker": case.ticker.upper()}, {"year": int(case.fiscal_year)}]}
    documents = store.get(where=where, include=["documents"])["documents"]
    return [d for d in documents if any(f in d for f in figures)]


def _aggregate_scores(result) -> dict[str, float]:
    """Mean score per metric from a RAGAS EvaluationResult.

    `dict(result)` looks like the obvious way to do this and is a trap.
    EvaluationResult defines `__getitem__(key: str)` but no `keys()`, no
    `__iter__` and no `__len__`, so `dict()` falls back to Python's legacy
    sequence protocol and asks for `result[0]` -- which reaches
    `_scores_dict[0]` and raises `KeyError: 0`. It fails *after* every judge
    call has been paid for, which cost a fourteen-minute run.

    `_repr_dict` is what the class itself builds in `__post_init__`: one
    `safe_nanmean` per metric. Private, hence the fallbacks -- and the nanmean
    matters, because a judge response that fails to parse leaves a NaN in the
    column and a plain mean would turn one bad parse into a NaN for the whole
    metric.
    """
    import math

    repr_dict = getattr(result, "_repr_dict", None)
    if isinstance(repr_dict, dict) and repr_dict:
        return {
            k: float(v)
            for k, v in repr_dict.items()
            if isinstance(v, (int, float)) and not math.isnan(float(v))
        }

    scores_dict = getattr(result, "_scores_dict", None)
    if isinstance(scores_dict, dict) and scores_dict:
        out: dict[str, float] = {}
        for name, values in scores_dict.items():
            usable = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
            if usable:
                out[name] = sum(usable) / len(usable)
        return out

    # Last resort: the public frame. Metric columns are the numeric ones.
    frame = result.to_pandas()
    return {
        str(col): float(frame[col].mean())
        for col in frame.columns
        if frame[col].dtype.kind in "fi" and not math.isnan(float(frame[col].mean()))
    }


def evaluate_ragas(
    cases: list[AgentCase] | None = None,
    store=None,
    settings: Settings | None = None,
    use_gold_context: bool = False,
    limit: int | None = None,
    checkpoint_path=None,
) -> RagasReport:
    """Score answer quality with RAGAS.

    Needs whichever provider key settings.llm_backend requires. Free-tier
    survival is layered in here: the chat model carries a client-side rate
    limiter and an optional fallback chain, the SQLite response cache absorbs
    repeated judge calls across reruns, generation checkpoints per case, and
    the scoring pass runs serially (RunConfig) because RAGAS's default of 16
    concurrent workers is a guaranteed 429 on a free tier.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise ImportError(
            "RAGAS evaluation needs the eval extra: pip install 'finrag[eval]'"
        ) from exc

    from ..cache import enable_llm_cache
    from ..embeddings import get_embeddings
    from ..llm import build_with_fallbacks, get_judge_model

    settings = settings or get_settings()
    cases = cases if cases is not None else load_agent_cases()
    if limit:
        cases = cases[:limit]

    enable_llm_cache(settings)
    llm = build_with_fallbacks(settings)
    judge, judge_label = get_judge_model(settings)
    log.info("generator=%s judge=%s", settings.llm_backend, judge_label)

    if store is None:
        from ..ingest.index import open_store

        # Needed in gold mode too: oracle context is selected from the index,
        # not conjured from the reference answer.
        store = open_store(settings)

    samples = build_samples(
        cases, llm, store, settings, use_gold_context, checkpoint_path=checkpoint_path
    )

    dataset = Dataset.from_dict(
        {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.reference for s in samples],
        }
    )
    # Both must be passed explicitly. RAGAS falls back to OpenAI for the judge
    # *and* for the embeddings that answer_relevancy uses, so leaving either
    # unset fails with an OpenAI key error however the rest is configured.
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=LangchainLLMWrapper(judge),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings(settings)),
        run_config=RunConfig(timeout=300, max_retries=10, max_wait=60, max_workers=1),
    )

    scores = _aggregate_scores(result)
    return RagasReport(
        samples=samples,
        scores=scores,
        context_source="gold" if use_gold_context else "retrieved",
        judge=judge_label,
    )
