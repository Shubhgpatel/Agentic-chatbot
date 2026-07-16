# entry point app.py

import json
import uuid
from pathlib import Path

import streamlit as st
from chatbot_backend.graph import graph
from langchain_core.messages import HumanMessage

THREADS_FILE = Path("threads.json")

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖"
)

# --------------------------
# Thread Helper Functions
# --------------------------

def load_threads():

    if not THREADS_FILE.exists():
        THREADS_FILE.write_text("[]")
        return []

    try:

        with open(THREADS_FILE, "r") as f:
            return json.load(f)

    except json.JSONDecodeError:

        THREADS_FILE.write_text("[]")

        return []

def save_threads(threads):
    with open(THREADS_FILE, "w") as f:
        json.dump(threads, f, indent=4)

threads = load_threads()

# --------------------------
# Session Initialization
# --------------------------

if "thread_id" not in st.session_state:

    threads = threads

    if len(threads) == 0:
        new_thread = str(uuid.uuid4())
        threads.append(new_thread)
        save_threads(threads)
        st.session_state.thread_id = new_thread
    else:
        st.session_state.thread_id = threads[0]


# --------------------------
# Sidebar
# --------------------------

with st.sidebar:

    st.title("Chats")

    if st.button("➕ New Chat", use_container_width=True):

        new_thread = str(uuid.uuid4())

        threads = threads
        threads.insert(0, new_thread)

        save_threads(threads)

        st.session_state.thread_id = new_thread

        st.rerun()

    st.divider()

    for thread in threads:

        if st.button(thread, use_container_width=True):

            st.session_state.thread_id = thread
            st.rerun()


# --------------------------
# Main UI
# --------------------------

st.title("🤖 LangGraph Chatbot")

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

# --------------------------
# Load Conversation
# --------------------------

snapshot = graph.get_state(config)

if snapshot.values:

    for msg in snapshot.values["messages"]:

        # Show user messages
        if msg.type == "human":

            with st.chat_message("user"):
                st.markdown(msg.content)

        # Show only the final AI response
        elif msg.type == "ai":

            # Skip intermediate AI messages that contain tool calls
            if getattr(msg, "tool_calls", None):
                continue

            with st.chat_message("assistant"):
                st.markdown(msg.content)

        # Skip ToolMessage completely
        elif msg.type == "tool":
            continue

# --------------------------
# User Input
# --------------------------

user_input = st.chat_input("Type your message...")

if user_input:

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        def token_generator():

            for chunk, metadata in graph.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=config,
                stream_mode="messages"
            ):

                if metadata.get("langgraph_node") != "chatbot":
                    continue

                if not chunk.content:
                    continue

                yield chunk.content

        st.write_stream(token_generator())

    st.rerun()