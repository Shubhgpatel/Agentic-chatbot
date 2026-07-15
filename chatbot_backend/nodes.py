# chatbot_backend/nodes.py

from chatbot_backend.llm import llm
from chatbot_backend.state import ChatState

def chatbot(state: ChatState):

    ai_response = llm.invoke(state["messages"])

    return {
        "messages": [ai_response]
    }