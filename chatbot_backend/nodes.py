# chatbot_backend/nodes.py

from chatbot_backend.llm import llm_with_tools
from chatbot_backend.state import ChatState

def chatbot(state: ChatState):
    
    ai_response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [ai_response]
    }