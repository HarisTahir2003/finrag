"""Turning an agent run back into something a person can check.

The retrieval tool hands the model one rendered string, because a tool returns
a string. Showing a reader which passages an answer rests on therefore means
parsing that string back apart. That is a little awkward and entirely worth it:
an answer of "$391,035 million" is worth what the passage behind it is worth,
and an agent that invented the figure looks identical to one that retrieved it.

Lives in the package rather than in app.py because it encodes the *agent's*
output format, not any UI's widgets -- the HTTP layer needs exactly the same
parsing, and a copy in each would drift.
"""

from __future__ import annotations

import re

# Fixed by Retrieved.as_context(): a filing header, then numbered chunk blocks.
_FILING_HEADER = re.compile(r"^---\s*([A-Z.]+)\s+FY(\d{4})\s*---", re.MULTILINE)
_CHUNK_BREAK = re.compile(r"\n\[chunk \d+\]\n")


def parse_passages(observation: str) -> tuple[str, list[str]]:
    """Split a retrieval observation into its filing label and passages.

    Returns ("filing", [whole thing]) when the text does not match -- an
    unparsed observation is still worth showing, and a display helper should
    never be the thing that fails a request.
    """
    if not observation:
        return "filing", []

    header = _FILING_HEADER.search(observation)
    label = f"{header.group(1)} FY{header.group(2)}" if header else "filing"
    body = observation[header.end() :] if header else observation
    return label, [p.strip() for p in _CHUNK_BREAK.split(body) if p.strip()]


def calculator_expression(tool_input) -> str:
    """The expression a calculator call carried, without its envelope.

    The tool is invoked with {"expression": "..."}, and rendering that dict
    verbatim puts `{'expression': '152987/176392'}` in front of the reader.
    """
    if isinstance(tool_input, dict):
        return str(tool_input.get("expression", tool_input))
    return str(tool_input)


def describe_action(tool: str, tool_input) -> str:
    """One line naming what the agent is doing, for a progress log.

    An answer takes tens of seconds across several tool calls, and a bare
    spinner for that long is indistinguishable from a hang.
    """
    if tool == "search_10k_reports":
        args = tool_input if isinstance(tool_input, dict) else {}
        target = f"{args.get('ticker', '?')} FY{args.get('fiscal_year', '?')}"
        return f"Searching {target} for *{args.get('query', '')}*"
    if tool == "calculator":
        return f"Calculating `{calculator_expression(tool_input)}`"
    return f"Running `{tool}`"


def summarise_steps(steps: list[dict]) -> dict:
    """Group an agent's tool calls into what a reader wants to check.

    ``steps`` is a list of {"tool", "input", "observation"}. The UI and the HTTP
    API both need this shape, and a copy in each would drift the moment one of
    them learned about a new tool.
    """
    sources, calculations = [], []
    for step in steps:
        tool = step.get("tool")
        if tool == "search_10k_reports":
            filing, passages = parse_passages(str(step.get("observation", "")))
            tool_input = step.get("input") or {}
            sources.append(
                {
                    "filing": filing,
                    "query": tool_input.get("query", "") if isinstance(tool_input, dict) else "",
                    "passages": passages,
                }
            )
        elif tool == "calculator":
            calculations.append(
                {
                    "expression": calculator_expression(step.get("input")),
                    "result": str(step.get("observation", "")),
                }
            )
    return {"sources": sources, "calculations": calculations}


# A `$` opens LaTeX in most markdown renderers, so a sentence carrying two
# dollar figures -- which is most sentences here -- gets everything between them
# rendered as maths. "$574,785 million in 2023, compared to Apple's $383,285"
# came out as one italic equation.
_CODE_SPAN = re.compile(r"(```.*?```|`[^`]*`)", re.DOTALL)


def escape_dollars(text: str) -> str:
    """Neutralise `$` as a maths delimiter, leaving code spans alone.

    Escaping inside a code span would show a literal backslash, so the text is
    split on fenced blocks and inline code first and only the prose between
    them is escaped.
    """
    if not text:
        return text
    parts = _CODE_SPAN.split(text)
    # split() with a capturing group alternates prose, delimiter, prose...
    return "".join(
        part if index % 2 else part.replace("\\$", "$").replace("$", "\\$")
        for index, part in enumerate(parts)
    )


def failure_message(exc: BaseException, backend: str) -> str:
    """What to show a reader when a question fails.

    A provider error rendered raw -- "Error code: 429 - {'error': {'message':
    'Rate limit reached for model ...'}}" -- tells a reader nothing they can act
    on, and on a public demo it is the most likely thing they will ever see.

    Each branch says what to do next, and the two 429s say different things on
    purpose: a per-minute limit clears by itself, a spent daily quota does not.
    """
    from .llm import classify_provider_error

    kind = classify_provider_error(exc)
    name = backend.title()

    if kind == "quota":
        return (
            f"**The daily {name} quota for this demo is used up.** It resets "
            "every 24 hours. To keep going now, put your own free "
            f"{name} API key in the sidebar — it is held for your session only, "
            "is never written to disk, and is not shared with anyone else using "
            "this page."
        )
    if kind == "too_large":
        return (
            "**That question needed more context than the free tier allows in one "
            f"request.** {name}'s free tier caps a single request, and comparing two "
            "companies sends both filings at once. Ask about one company at a time -- "
            '"What was Amazon\'s net income in 2023?", then the same for Apple -- or '
            "put your own key in the sidebar if yours has a higher limit."
        )
    if kind == "rate_limit":
        return (
            f"**{name} is rate-limiting this request.** This one clears on its "
            "own — wait a few seconds and ask again. Asking about a single "
            "company rather than comparing two also helps, since a comparison "
            "sends several filings' worth of context at once."
        )
    if kind in ("auth", "missing_key"):
        return (
            f"**That {name} API key was rejected.** Check it in the sidebar. "
            "Leaving the box empty falls back to the key configured on the "
            "server, if there is one."
        )
    return f"Something went wrong: {exc}"
