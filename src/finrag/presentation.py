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
