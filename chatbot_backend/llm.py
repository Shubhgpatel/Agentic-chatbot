# chatbot_backend/llm.py
from chatbot_backend.tools import tools

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    streaming=True
)

llm_with_tools = llm.bind_tools(tools)