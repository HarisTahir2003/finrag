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
# the exponent is bounded -- for the `**` operator and for pow() alike, since
# a whitelisted pow() reaches the interpreter's builtin unchecked otherwise.
MAX_EXPONENT = 128

# A result this large is not a financial figure, it is an attempt to make the
# process do work. 8192 bits is a ~2,466-digit integer -- astronomically past
# any 10-K number, and the ceiling that stops chained exponentiation
# (`(2**128)**128`...), where every individual exponent is under MAX_EXPONENT
# but the running product is not. Floats saturate to inf on overflow (caught
# below), so only integers need this.
MAX_RESULT_BITS = 8192

# Functions that legitimately take an iterable argument. Everywhere else a list
# or tuple is a category error -- and, left evaluable, `[0] * 10**8` is a
# 12-character request that allocates ~800MB and OOM-kills the container.
_ITERABLE_FUNCTIONS = frozenset({"min", "max", "sum"})


def _check_number(value: Any, where: str) -> Any:
    """Every arithmetic operand must be a number, never a list.

    `[0] * 5` is valid Python and returns a list; `list * int` is the shape the
    allocation attack takes. Requiring numeric operands closes it directly,
    independent of how the list was produced.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalculatorError(f"{where} must be a number, got {type(value).__name__}")
    if isinstance(value, int) and value.bit_length() > MAX_RESULT_BITS:
        raise CalculatorError("intermediate value is too large")
    return value


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
        left = _check_number(_evaluate(node.left), "left operand")
        right = _check_number(_evaluate(node.right), "right operand")
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"exponent {right} exceeds the limit of {MAX_EXPONENT}")
        result = op(left, right)
        if isinstance(result, int) and result.bit_length() > MAX_RESULT_BITS:
            raise CalculatorError("result is too large")
        return result

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"unary {type(node.op).__name__} is not allowed")
        return op(_check_number(_evaluate(node.operand), "operand"))

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

        # pow() bypasses the `**` exponent guard otherwise -- pow(2, 4000) is
        # inside the length limit and builds a 1,200-digit integer.
        if node.func.id == "pow" and len(node.args) >= 2:
            exponent = _evaluate(node.args[1])
            if isinstance(exponent, (int, float)) and abs(exponent) > MAX_EXPONENT:
                raise CalculatorError(f"exponent {exponent} exceeds the limit of {MAX_EXPONENT}")

        args = [_evaluate_argument(a, node.func.id) for a in node.args]
        result = fn(*args)
        if isinstance(result, int) and result.bit_length() > MAX_RESULT_BITS:
            raise CalculatorError("result is too large")
        return result

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise CalculatorError(f"unknown name '{node.id}'")

    raise CalculatorError(f"{type(node).__name__} is not allowed in an expression")


def _evaluate_argument(node: ast.AST, function_name: str) -> Any:
    """A call argument, which may be a list/tuple only for min/max/sum.

    A list is permitted here and nowhere else: `sum([1, 2, 3])` is legitimate,
    while `[0] * 5` -- a list as an arithmetic operand -- is the allocation
    attack. Keeping list construction out of `_evaluate` entirely means the only
    way to make one is as a direct argument to a function that consumes it.
    """
    if isinstance(node, (ast.List, ast.Tuple)):
        if function_name not in _ITERABLE_FUNCTIONS:
            raise CalculatorError(f"'{function_name}' does not take a list argument")
        return [_check_number(_evaluate(e), "list element") for e in node.elts]
    return _evaluate(node)


def calculate(expression: str) -> float | int | bool:
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
    except RecursionError as exc:
        # A deeply nested expression, e.g. thousands of parentheses.
        raise CalculatorError("expression is too deeply nested") from exc
    except MemoryError as exc:
        # Belt and braces behind the magnitude guards: a computation that
        # allocates must surface as a rejected expression, not an unhandled
        # error that the tool wrapper does not catch and that kills the run.
        raise CalculatorError("expression would allocate too much memory") from exc
    except (ValueError, OverflowError, TypeError) as exc:
        raise CalculatorError(str(exc)) from exc


def calculate_as_text(expression: str) -> str:
    """Tool-facing wrapper: never raises, so a bad expression does not kill the agent run."""
    try:
        return str(calculate(expression))
    except CalculatorError as exc:
        return f"Error: {exc}. Provide a single arithmetic expression, for example (143566 - 6331) / 145308"
