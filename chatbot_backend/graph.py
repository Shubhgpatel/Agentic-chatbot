from langgraph.graph import StateGraph, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
import sqlite3
from pathlib import Path

from chatbot_backend.state import ChatState
from chatbot_backend.nodes import chatbot
from chatbot_backend.tools import tools

builder = StateGraph(ChatState)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")

builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

builder.add_edge("tools", "chatbot")

# Anchor the DB to the project root so it's the same file regardless of cwd.
DB_PATH = Path(__file__).resolve().parent.parent / "chatbot.db"

conn = sqlite3.connect(
    str(DB_PATH),
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)

graph = builder.compile(checkpointer=checkpointer)