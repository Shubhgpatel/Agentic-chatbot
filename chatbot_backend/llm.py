# chatbot_backend/llm.py
from chatbot_backend.tools import tools

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# NOTE: llama-3.3-70b-versatile intermittently emits a malformed Llama
# function-call token (e.g. `<function=web_search {...}` missing the closing `>`),
# which Groq rejects with `tool_use_failed`, so avoid going back to it.
# meta-llama/llama-4-scout-17b-16e-instruct was the fix, but Groq has since
# decommissioned it (404 model_not_found). gpt-oss-120b has native, reliable
# tool calling. Other options on the account: "qwen/qwen3.6-27b" or
# "openai/gpt-oss-20b" (lighter). Check availability with client.models.list().
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    streaming=True
)

llm_with_tools = llm.bind_tools(tools)