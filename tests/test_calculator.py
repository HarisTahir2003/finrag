"""The calculator replaced eval() plus a Python REPL driven by model output.

Half of these tests are arithmetic; the other half assert that the escapes an
LLM can be talked into emitting are refused.
"""

from __future__ import annotations

import pytest

from finrag.calculator import CalculatorError, calculate, calculate_as_text


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", 4),
        ("143566 / 145308", 0.9880116991),  # Apple FY2023 current ratio
        ("143,566 / 145,308", 0.9880116991),  # models emit thousands separators
        ("(143566 - 6331) / 145308", 0.9444352),  # acid test
        ("(106618 - 82338) / 82338 * 100", 29.4882),  # Tesla YoY asset growth
        ("abs(-5)", 5),
        ("round(3.14159, 2)", 3.14),
        ("sqrt(16)", 4),
        ("max(1, 2, 3)", 3),
        ("2 ** 10", 1024),
    ],
)
def test_arithmetic(expression, expected):
    assert calculate(expression) == pytest.approx(expected, rel=1e-4)


def test_strips_code_fences_and_language_hints():
    assert calculate("```python\n1 + 1\n```") == 2


@pytest.mark.parametrize(
    "expression",
    [
        '__import__("os").system("echo pwned")',
        'open("/etc/passwd").read()',
        "().__class__.__bases__[0].__subclasses__()",
        "eval('1+1')",
        "exec('x=1')",
        "globals()",
        "[].__class__",
        "lambda: 1",
        "x = 1",
        "import os",
        "print(1)",
        "os.getcwd()",
    ],
)
def test_refuses_anything_that_is_not_arithmetic(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


def test_bounds_exponentiation():
    """Unbounded ** is the one way to burn CPU without allocating memory."""
    with pytest.raises(CalculatorError, match="exponent"):
        calculate("2 ** 999999")


def test_rejects_oversized_input():
    with pytest.raises(CalculatorError, match="characters"):
        calculate("1+" * 400 + "1")


def test_division_by_zero_is_a_calculator_error():
    with pytest.raises(CalculatorError, match="zero"):
        calculate("1 / 0")


def test_tool_wrapper_never_raises():
    """A bad expression must not abort the agent run."""
    result = calculate_as_text('__import__("os")')
    assert result.startswith("Error:")
    assert calculate_as_text("6 / 3") == "2.0"
