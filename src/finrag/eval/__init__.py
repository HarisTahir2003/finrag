"""Evaluation harness.

Three tiers, separated by what they cost to run:

``retrieval``
    Needs no LLM. Measures whether the right passages come back, using string
    matching against a labelled set. Deterministic and free, so it runs in CI
    and is what the quality gate checks.

``ragas``
    Needs an LLM judge. Measures faithfulness, answer relevancy and context
    precision **through the retriever** -- which is the thing the original
    notebook did not do.

``agent``
    Needs an LLM. Runs the full agent over the question set and scores both the
    answer and whether the expected tools were called.
"""
