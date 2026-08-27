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

    if args.command == "ask":
        from .agent import build_agent

        agent = build_agent(settings=settings, verbose=args.verbose)
        result = agent.invoke({"input": args.question})
        print(f"\n{result['output']}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
