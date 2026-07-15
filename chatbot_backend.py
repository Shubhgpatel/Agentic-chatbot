
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


class ChatState(TypedDict):
    message: str
    response: str


def chatbot(state: ChatState):

    user_message = state["message"]

    ai_response = llm.invoke(user_message)

    return {
        "response": ai_response.content
    }


graph_builder = StateGraph(ChatState)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# while True:

#     user_input = input("You: ")

#     if user_input.lower() == "exit":
#         break

#     result = graph.invoke({
#         "message": user_input
#     })

#     print("AI:", result["response"])