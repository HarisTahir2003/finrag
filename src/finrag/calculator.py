"""A calculator the agent can call without handing it arbitrary code execution.

The original tool combined ``eval()`` with LangChain's ``PythonREPLTool``, both
driven by model output. That is acceptable in a private notebook and completely
unacceptable behind a public URL: the model can be steered into running anything
the process can run.

This evaluates arithmetic by walking the AST and refusing every node type that
is not arithmetic. There is no import machinery, no attribute access, no name
lookup beyond a small whitelist of constants and functions, and no way to reach
a builtin. It handles every ratio in the evaluation set.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

MAX_EXPRESSION_LENGTH = 500

# 143,566 -> 143566, while leaving argument separators like round(x, 2) alone.
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


class CalculatorError(ValueError):
    """Raised when an expression is rejected or cannot be evaluated."""


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

# Only pure numeric helpers. Nothing here can touch the filesystem, the network
# or the interpreter.
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}

# Exponentiation is the one operator that can burn CPU without allocating, so
# the exponent is bounded.
MAX_EXPONENT = 128


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        # bool is a subclass of int, so it is listed only to make the intent explicit.
        if isinstance(node.value, (bool, int, float)):
            return node.value
        raise CalculatorError(f"only numeric literals are allowed, got {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"operator {type(node.op).__name__} is not allowed")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"exponent {right} exceeds the limit of {MAX_EXPONENT}")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"unary {type(node.op).__name__} is not allowed")
        return op(_evaluate(node.operand))

    if isinstance(node, ast.Compare):
        result = _evaluate(node.left)
        for op_node, comparator in zip(node.ops, node.comparators, strict=True):
            op = _COMPARE_OPS.get(type(op_node))
            if op is None:
                raise CalculatorError(f"comparison {type(op_node).__name__} is not allowed")
            right = _evaluate(comparator)
            if not op(result, right):
                return False
            result = right
        return True

    if isinstance(node, ast.Call):
        # Only bare names may be called, so `os.system(...)` and
        # `().__class__.__bases__` style escapes cannot even be parsed into a call.
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("only direct calls to whitelisted functions are allowed")
        fn = _FUNCTIONS.get(node.func.id)
        if fn is None:
            raise CalculatorError(f"function '{node.func.id}' is not allowed")
        if node.keywords:
            raise CalculatorError("keyword arguments are not allowed")
        return fn(*[_evaluate(a) for a in node.args])

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise CalculatorError(f"unknown name '{node.id}'")

    if isinstance(node, (ast.Tuple, ast.List)):
        return [_evaluate(e) for e in node.elts]

    raise CalculatorError(f"{type(node).__name__} is not allowed in an expression")


def calculate(expression: str) -> float:
    """Evaluate a single arithmetic expression, or raise CalculatorError."""
    if not isinstance(expression, str) or not expression.strip():
        raise CalculatorError("expression is empty")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(f"expression exceeds {MAX_EXPRESSION_LENGTH} characters")

    # Models like to wrap answers in code fences or prefix them with `print(...)`.
    cleaned = expression.strip().strip("`").strip()
    if cleaned.lower().startswith("python"):
        cleaned = cleaned[len("python") :].strip()
    # Strip thousands separators only: a comma between a digit and exactly three
    # more digits. Removing every comma would turn max(1, 2, 3) into max(1 2 3).
    cleaned = _THOUSANDS_SEPARATOR.sub("", cleaned)

    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"could not parse expression: {exc.msg}") from exc

    try:
        return _evaluate(tree)
    except CalculatorError:
        raise
    except ZeroDivisionError as exc:
        raise CalculatorError("division by zero") from exc
    except (ValueError, OverflowError, TypeError) as exc:
        raise CalculatorError(str(exc)) from exc


def calculate_as_text(expression: str) -> str:
    """Tool-facing wrapper: never raises, so a bad expression does not kill the agent run."""
    try:
        return str(calculate(expression))
    except CalculatorError as exc:
        return f"Error: {exc}. Provide a single arithmetic expression, for example (143566 - 6331) / 145308"
