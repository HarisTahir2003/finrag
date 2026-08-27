"""Command line entry point.

finrag download --tickers AAPL,MSFT --years 5
finrag index
finrag ask "What was Apple's current ratio in 2023?"
finrag status
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import DEFAULT_TICKERS, get_settings


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finrag", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download", help="fetch 10-K filings from SEC EDGAR")
    p_download.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    p_download.add_argument("--years", type=int, default=5)

    sub.add_parser("index", help="parse and index downloaded filings (idempotent)")
    sub.add_parser("status", help="show configuration and index size")

    p_ask = sub.add_parser("ask", help="ask the agent a question")
    p_ask.add_argument("question")

    p_eval = sub.add_parser("eval", help="run an evaluation suite")
    p_eval.add_argument("suite", choices=["retrieval", "ragas", "agent"])
    p_eval.add_argument(
        "--fixtures",
        action="store_true",
        help="build a throwaway index from the committed test fixtures (no downloads, no API key)",
    )
    p_eval.add_argument(
        "--gate", action="store_true", help="exit non-zero if thresholds are breached"
    )
    p_eval.add_argument("--limit", type=int, help="evaluate only the first N cases")
    p_eval.add_argument(
        "--gold-context",
        action="store_true",
        help="ragas only: score against gold context instead of the retriever, reproducing the original measurement",
    )
    p_eval.add_argument(
        "--dataset",
        default=None,
        help="named case set from eval/datasets (e.g. 'smoke' for the 3-case CI suite)",
    )
    p_eval.add_argument(
        "--resume",
        action="store_true",
        help="skip cases already completed in results/<suite>-cases.jsonl -- lets a run "
        "that died on a free tier's daily quota finish later without repeating calls",
    )

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    settings = get_settings()

    if args.command == "download":
        from .ingest.download import download_filings

        counts = download_filings(
            tuple(t.strip().upper() for t in args.tickers.split(",") if t.strip()),
            years=args.years,
            settings=settings,
        )
        print(f"\n{sum(counts.values())} filings on disk across {len(counts)} tickers")
        return 0

    if args.command == "index":
        from .ingest.index import index_filings

        result = index_filings(settings=settings)
        print(f"\nindexed {result['filings']} filings -> {result['chunks']} chunks")
        return 0

    if args.command == "status":
        from .ingest.download import list_filings

        print(f"data root          {settings.data_root}")
        print(f"embedding backend  {settings.embedding_backend}")
        print(f"chunk strategy     {settings.chunk_strategy}")
        print(f"index              {settings.index_dir}")
        print(f"filings on disk    {len(list_filings(settings=settings))}")
        try:
            from .ingest.index import collection_size

            print(f"chunks indexed     {collection_size(settings)}")
        except Exception as exc:  # noqa: BLE001 - status must never fail hard
            print(f"chunks indexed     unavailable ({exc})")
        return 0

    if args.command == "eval":
        return _run_eval(args, settings)

    if args.command == "ask":
        from .agent import build_agent

        agent = build_agent(settings=settings, verbose=args.verbose)
        result = agent.invoke({"input": args.question})
        print(f"\n{result['output']}")
        return 0

    return 1


def _fixture_index(settings):
    """Index the committed fixtures into a temporary store.

    Lets the retrieval suite run on a fresh clone with no EDGAR download and no
    API key, which is what makes it usable as a CI gate.
    """
    import tempfile
    from dataclasses import replace
    from pathlib import Path

    from .ingest.index import index_filings, open_store

    fixtures = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "sec-edgar-filings"
    paths = sorted(fixtures.glob("**/full-submission.txt"))
    if not paths:
        raise SystemExit(f"no fixtures found under {fixtures}")

    # The fixtures are only a couple of KB each. At the production chunk size they
    # would each index as a single chunk, so every filtered query would have exactly
    # one candidate and hit rate and MRR would be 1.0 by construction -- a gate that
    # cannot fail is not a gate. A smaller chunk size splits each filing into several
    # passages so ranking is actually exercised.
    tmp = Path(tempfile.mkdtemp(prefix="finrag-eval-"))
    settings = replace(settings, data_root=tmp, chunk_size=400, chunk_overlap=60)
    result = index_filings(paths=paths, settings=settings)
    print(f"indexed {result['filings']} fixtures -> {result['chunks']} chunks\n")
    return open_store(settings), settings


def _run_eval(args, settings) -> int:
    from pathlib import Path

    from .eval.tracking import config_params, track_run

    store = None
    if args.fixtures:
        store, settings = _fixture_index(settings)

    # Named case sets live next to the built-in ones; --dataset smoke selects
    # the 3-case suite sized for free daily quotas (and used by CI).
    cases = None
    if args.dataset:
        from .eval.schema import DATASETS_DIR, load_agent_cases

        cases = load_agent_cases(DATASETS_DIR / f"{args.dataset}.yaml")

    # Checkpoints use a stable path per suite so --resume can find them. A
    # fresh (non-resume) run starts clean rather than inheriting stale cases.
    checkpoint_path = None
    if args.suite in ("ragas", "agent"):
        checkpoint_path = Path("results") / f"{args.suite}-cases.jsonl"
        if not args.resume and checkpoint_path.exists():
            checkpoint_path.unlink()

    if args.suite == "retrieval":
        from .eval.gate import check
        from .eval.retrieval_eval import evaluate_retrieval

        report = evaluate_retrieval(store=store, settings=settings)
        print(report.format_table())
        metrics = report.as_metrics()
        with track_run("retrieval", config_params(settings)) as record:
            record(metrics)
        print("\n" + "\n".join(f"  {k:22} {v}" for k, v in metrics.items()))

        if args.gate:
            result = check(report)
            print("\n" + result.format())
            return 0 if result.passed else 1
        return 0

    if args.suite == "ragas":
        from .eval.ragas_eval import evaluate_ragas

        report = evaluate_ragas(
            cases=cases,
            store=store,
            settings=settings,
            use_gold_context=args.gold_context,
            limit=args.limit,
            checkpoint_path=checkpoint_path,
        )
        metrics = report.as_metrics()
        with track_run(f"ragas-{report.context_source}", config_params(settings)) as record:
            record(metrics)
        print(f"\ncontext source: {report.context_source}")
        print("\n".join(f"  {k:22} {v}" for k, v in metrics.items()))
        return 0

    from .cache import enable_llm_cache
    from .eval.agent_eval import evaluate_agent

    enable_llm_cache(settings)
    report = evaluate_agent(
        cases=cases,
        store=store,
        settings=settings,
        limit=args.limit,
        checkpoint_path=checkpoint_path,
    )
    print(report.format_table())
    metrics = report.as_metrics()
    with track_run("agent", config_params(settings)) as record:
        record(metrics)
    print("\n" + "\n".join(f"  {k:22} {v}" for k, v in metrics.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
