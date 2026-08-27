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

Set ``use_gold_context=True`` to reproduce the old measurement side by side.
That comparison is the point -- the gap between the two is the retrieval
contribution, which was previously invisible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import Settings, get_settings
from ..retrieval import search_filing
from .schema import AgentCase, load_agent_cases

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

    def as_metrics(self) -> dict[str, float]:
        metrics = {f"ragas_{k}": round(v, 4) for k, v in self.scores.items()}
        metrics["samples"] = len(self.samples)
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
) -> list[RagasSample]:
    """Generate an answer per case, with context from the retriever by default."""
    settings = settings or get_settings()
    source = "gold" if use_gold_context else "retrieved"
    samples: list[RagasSample] = []

    for case in cases:
        if use_gold_context:
            contexts = [case.reference_answer]
        else:
            found = search_filing(
                case.question, case.ticker, case.fiscal_year, store=store, settings=settings
            )
            contexts = [d.page_content for d in found.documents]
            if not contexts:
                log.warning("%s: retrieved nothing; scoring against empty context", case.id)

        answer = _generate_answer(llm, case.question, "\n\n".join(contexts))
        samples.append(
            RagasSample(
                question=case.question,
                answer=answer,
                contexts=contexts or [""],
                reference=case.reference_answer,
                context_source=source,
            )
        )
        log.info("%s: answered from %d context chunks", case.id, len(contexts))

    return samples


def evaluate_ragas(
    cases: list[AgentCase] | None = None,
    store=None,
    settings: Settings | None = None,
    use_gold_context: bool = False,
    limit: int | None = None,
) -> RagasReport:
    """Score answer quality with RAGAS. Requires GOOGLE_API_KEY."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        raise ImportError(
            "RAGAS evaluation needs the eval extra: pip install 'finrag[eval]'"
        ) from exc

    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = settings or get_settings()
    cases = cases if cases is not None else load_agent_cases()
    if limit:
        cases = cases[:limit]

    llm = ChatGoogleGenerativeAI(model=settings.chat_model, temperature=0)

    if store is None and not use_gold_context:
        from ..ingest.index import open_store

        store = open_store(settings)

    samples = build_samples(cases, llm, store, settings, use_gold_context)

    dataset = Dataset.from_dict(
        {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.reference for s in samples],
        }
    )
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])

    scores = {k: float(v) for k, v in dict(result).items() if isinstance(v, (int, float))}
    return RagasReport(
        samples=samples, scores=scores, context_source="gold" if use_gold_context else "retrieved"
    )
