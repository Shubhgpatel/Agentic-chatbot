# entry point app.py

import json
import uuid
from collections import Counter
from pathlib import Path

import streamlit as st
from chatbot_backend.graph import graph
from chatbot_backend.tools import ingest_pdfs, indexed_pdfs
from langchain_core.messages import HumanMessage

# Anchor to the project root so the thread list is the same file regardless of cwd.
THREADS_FILE = Path(__file__).resolve().parent / "threads.json"

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖"
)

# --------------------------
# Tool display metadata
# --------------------------

TOOL_ICONS = {
    "calculator": "🧮",
    "web_search": "🌐",
    "duckduckgo_search": "🌐",
}

def tool_icon(name):
    return TOOL_ICONS.get(name, "🔧")

# --------------------------
# Thread Helper Functions
# --------------------------
# Schema: threads.json is a list of {"id": <uuid>, "name": <display name>}.
# Old files stored bare uuid strings; load_threads() migrates them.

def _make_thread(name=None):
    return {
        "id": str(uuid.uuid4()),
        "name": name or "New Chat",
    }

def load_threads():

    if not THREADS_FILE.exists():
        THREADS_FILE.write_text("[]")
        return []

    try:
        with open(THREADS_FILE, "r") as f:
            raw = json.load(f)
    except json.JSONDecodeError:
        THREADS_FILE.write_text("[]")
        return []

    # Migrate any legacy string entries to the new dict shape.
    migrated = []
    changed = False
    for i, entry in enumerate(raw):
        if isinstance(entry, str):
            migrated.append({"id": entry, "name": f"Chat {entry[:8]}"})
            changed = True
        elif isinstance(entry, dict) and "id" in entry:
            migrated.append({"id": entry["id"], "name": entry.get("name", f"Chat {i + 1}")})
        # anything else is ignored as corrupt

    if changed:
        save_threads(migrated)

    return migrated

def save_threads(threads):
    with open(THREADS_FILE, "w") as f:
        json.dump(threads, f, indent=4)

def get_thread(threads, tid):
    for t in threads:
        if t["id"] == tid:
            return t
    return None

threads = load_threads()

# --------------------------
# Session Initialization
# --------------------------

if "thread_id" not in st.session_state:

    if len(threads) == 0:
        new_thread = _make_thread()
        threads.append(new_thread)
        save_threads(threads)
        st.session_state.thread_id = new_thread["id"]
    else:
        st.session_state.thread_id = threads[0]["id"]

# Transient UI state
st.session_state.setdefault("renaming", None)        # id being renamed
st.session_state.setdefault("confirm_delete", None)  # id pending delete confirm

# --------------------------
# Config + current conversation state
# (computed early so both the sidebar activity panel and the main view can use it)
# --------------------------

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

snapshot = graph.get_state(config)
history_messages = snapshot.values.get("messages", []) if snapshot.values else []

def tool_usage(messages):
    """Count how many times each tool was called across the conversation."""
    counts = Counter()
    for msg in messages:
        if getattr(msg, "type", None) == "ai":
            for tc in getattr(msg, "tool_calls", None) or []:
                counts[tc["name"]] += 1
    return counts

def render_activity(box, counts, running=None):
    """Render the Agent Activity panel into a placeholder."""
    lines = []
    if running:
        lines.append(f"⏳ {tool_icon(running)} **{running}** running…")
    if counts:
        for name, n in counts.items():
            lines.append(f"{tool_icon(name)} {name} &nbsp; `×{n}`")
    if not lines:
        lines.append("_No tools used yet._")
    box.markdown("\n\n".join(lines), unsafe_allow_html=True)

# --------------------------
# Sidebar
# --------------------------

with st.sidebar:

    st.title("Chats")

    if st.button("➕ New Chat", use_container_width=True):
        new_thread = _make_thread()
        threads.insert(0, new_thread)
        save_threads(threads)
        st.session_state.thread_id = new_thread["id"]
        st.session_state.renaming = None
        st.session_state.confirm_delete = None
        st.rerun()

    st.divider()

    for t in list(threads):
        tid = t["id"]
        name = t["name"]

        # --- Rename mode: inline text input ---
        if st.session_state.renaming == tid:
            new_name = st.text_input(
                "Rename chat",
                value=name,
                key=f"input_{tid}",
                label_visibility="collapsed",
            )
            c1, c2 = st.columns(2)
            if c1.button("Save", key=f"save_{tid}", use_container_width=True):
                clean = new_name.strip()
                if clean:
                    get_thread(threads, tid)["name"] = clean
                    save_threads(threads)
                st.session_state.renaming = None
                st.rerun()
            if c2.button("Cancel", key=f"cancel_{tid}", use_container_width=True):
                st.session_state.renaming = None
                st.rerun()
            continue

        # --- Delete confirm mode ---
        if st.session_state.confirm_delete == tid:
            c1, c2 = st.columns([0.6, 0.4])
            if c1.button("🗑️ Confirm?", key=f"confirmdel_{tid}", use_container_width=True):
                # Purge the conversation history from chatbot.db, then drop the entry.
                graph.checkpointer.delete_thread(tid)
                threads[:] = [x for x in threads if x["id"] != tid]
                save_threads(threads)

                if st.session_state.thread_id == tid:
                    if threads:
                        st.session_state.thread_id = threads[0]["id"]
                    else:
                        fresh = _make_thread()
                        threads.append(fresh)
                        save_threads(threads)
                        st.session_state.thread_id = fresh["id"]

                st.session_state.confirm_delete = None
                st.rerun()
            if c2.button("✖", key=f"canceldel_{tid}", use_container_width=True):
                st.session_state.confirm_delete = None
                st.rerun()
            continue

        # --- Normal row: [select] [rename] [delete] ---
        c1, c2, c3 = st.columns([0.7, 0.15, 0.15])

        if c1.button(
            f"💬 {name}",
            key=f"select_{tid}",
            use_container_width=True,
            type="primary" if tid == st.session_state.thread_id else "secondary",
        ):
            st.session_state.thread_id = tid
            st.rerun()

        if c2.button("✏️", key=f"rename_{tid}", use_container_width=True):
            st.session_state.renaming = tid
            st.session_state.confirm_delete = None
            st.rerun()

        if c3.button("🗑️", key=f"delete_{tid}", use_container_width=True):
            st.session_state.confirm_delete = tid
            st.session_state.renaming = None
            st.rerun()

    # --- PDF knowledge (per-thread) ---
    st.divider()
    st.subheader("📄 PDFs for this chat")

    # A thread-scoped key gives each chat its own uploader widget, so switching
    # threads shows that thread's uploads rather than leaking across chats.
    uploaded = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.thread_id}",
        label_visibility="collapsed",
    )

    if uploaded:
        # Idempotent: ingest_pdfs skips files already indexed for this thread,
        # so re-runs are cheap and re-uploads never duplicate the index.
        with st.spinner("Indexing PDFs…"):
            added = ingest_pdfs(st.session_state.thread_id, uploaded)
        if added:
            st.success("Indexed: " + ", ".join(added))

    already_indexed = indexed_pdfs(st.session_state.thread_id)
    if already_indexed:
        st.caption("Searchable in this chat:")
        for name in already_indexed:
            st.caption(f"• {name}")
    else:
        st.caption("_No PDFs indexed yet._")

    # --- Agent Activity panel ---
    st.divider()
    st.subheader("🛠️ Agent Activity")
    activity_box = st.empty()

# Render the activity panel from persisted history (updated live during streaming).
history_counts = tool_usage(history_messages)
render_activity(activity_box, history_counts)

# --------------------------
# Main UI
# --------------------------

st.title("🤖 LangGraph Chatbot")

# --------------------------
# Load Conversation
# --------------------------

for msg in history_messages:

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

        # Live tool counts start from history and grow as tools fire this turn.
        live_counts = Counter(history_counts)

        def token_generator():

            running = None

            for chunk, metadata in graph.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=config,
                stream_mode="messages"
            ):

                # Detect a tool call starting (name appears once, on its first delta).
                for tcc in getattr(chunk, "tool_call_chunks", None) or []:
                    name = tcc.get("name")
                    if name:
                        live_counts[name] += 1
                        running = name
                        render_activity(activity_box, live_counts, running=name)

                # Only stream assistant text from the chatbot node.
                if metadata.get("langgraph_node") != "chatbot":
                    continue

                if not chunk.content:
                    continue

                # First text token after a tool means the tool phase is done.
                if running is not None:
                    running = None
                    render_activity(activity_box, live_counts, running=None)

                yield chunk.content

        st.write_stream(token_generator())

    st.rerun()
