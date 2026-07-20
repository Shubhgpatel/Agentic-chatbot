# chatbot_backend/tools.py

import ast
import operator

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


# Only these operators are allowed — no names, calls, or attribute access,
# so arbitrary code (e.g. __import__('os').system(...)) can never run.
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")

    if isinstance(node, ast.BinOp):
        op = _ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))

    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))

    raise ValueError(f"Unsupported expression: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Example:
        calculator("25 * 9")
    """

    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)

    except Exception as e:
        return f"Error: {e}"

# Even better: use Tavily search tool for more accurate results, but DuckDuckGo is free and works well for most cases.
search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent information.
    Use this tool whenever the user asks about
    current events, stock prices, news, weather,
    sports, or other real-time information.
    """
    return search.run(query)


tools = [
    calculator,
    web_search
]