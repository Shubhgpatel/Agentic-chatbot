# chatbot_backend/tools.py

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Example:
        calculator("25 * 9")
    """

    try:
        result = eval(expression)
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