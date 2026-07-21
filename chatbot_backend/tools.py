# chatbot_backend/tools.py

import ast
import json
import operator
import os
import tempfile
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    Use this tool to solve mathematical calculations.

    Use it whenever the user asks for:
    - arithmetic
    - percentages
    - averages
    - ratios
    - exponents
    - division
    - multiplication
    - subtraction
    - addition

    Input should be a valid mathematical expression.

    Examples:
    calculator("25 * 9")
    calculator("(45 + 12) / 3")
    calculator("20 * 0.18")
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
    Use this tool whenever the question requires
    up-to-date information that cannot be reliably
    answered from the model's knowledge.

    Examples:
    - latest news
    - sports results
    - stock prices
    - weather
    - recent events
    - current CEO
    - today's date-dependent information
    """
    return search.run(query)


# ─────────────────────────────────────────────────────────────
# PDF RAG
#
# Each chat thread gets its own FAISS index on disk under
# faiss_indexes/<thread_id>/. A manifest.json records which PDFs
# have already been ingested so we never embed the same file twice
# and can append new PDFs to an existing index instead of rebuilding.
# ─────────────────────────────────────────────────────────────

# Anchor to the project root so the indexes live in the same place regardless of cwd.
INDEX_ROOT = Path(__file__).resolve().parent.parent / "faiss_indexes"

# Retrieved chunks closer than this L2 distance count as relevant. Embeddings are
# normalized, so distance runs 0 (identical) → 2 (opposite). Anything farther is
# treated as "not in the PDFs" so the model says so instead of guessing. Tunable.
_RELEVANCE_CUTOFF = 1.1

# The embedding model is heavy to load, so build it once and only on first use —
# the app shouldn't pay this cost unless RAG is actually exercised.
_embeddings = None

# Loaded FAISS stores, keyed by thread_id. Deserializing the index from disk on
# every query is pure repeated work, so we cache the store in memory. Streamlit
# reruns the script per interaction but the process persists, so this survives
# reruns (same reason the _embeddings singleton does). ingest_pdfs refreshes the
# entry on upload so searches always see the latest data.
_store_cache = {}


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _thread_dir(thread_id):
    return INDEX_ROOT / thread_id


def _get_store(thread_id):
    """Return this thread's FAISS store (cached), or None if nothing is indexed."""
    store = _store_cache.get(thread_id)
    if store is not None:
        return store

    if not (_thread_dir(thread_id) / "index.faiss").exists():
        return None

    store = FAISS.load_local(
        str(_thread_dir(thread_id)), _get_embeddings(), allow_dangerous_deserialization=True
    )
    _store_cache[thread_id] = store
    return store


def _load_manifest(thread_id):
    path = _thread_dir(thread_id) / "manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


def _save_manifest(thread_id, names):
    _thread_dir(thread_id).mkdir(parents=True, exist_ok=True)
    (_thread_dir(thread_id) / "manifest.json").write_text(json.dumps(names, indent=2))


def indexed_pdfs(thread_id):
    """Names of PDFs already indexed for this thread (for the sidebar to display)."""
    return _load_manifest(thread_id)


def ingest_pdfs(thread_id, uploaded_files):
    """
    Index any not-yet-seen PDFs for this thread and return the names newly added.

    Called from app.py on every rerun; it's a cheap no-op when there's nothing
    new, because the manifest tells us which files were already embedded.
    """
    already = set(_load_manifest(thread_id))
    new_files = [f for f in uploaded_files if f.name not in already]
    if not new_files:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = []
    for f in new_files:
        # PyPDFLoader needs a real path, so spool the upload to a temp file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        try:
            pages = PyPDFLoader(tmp_path).load()
        finally:
            os.unlink(tmp_path)

        # Stamp the original filename so search results can cite their source.
        for page in pages:
            page.metadata["source"] = f.name
        docs.extend(splitter.split_documents(pages))

    if not docs:
        return []

    embeddings = _get_embeddings()
    thread_dir = _thread_dir(thread_id)

    if (thread_dir / "index.faiss").exists():
        # Append to the existing index rather than rebuilding it.
        store = FAISS.load_local(
            str(thread_dir), embeddings, allow_dangerous_deserialization=True
        )
        store.add_documents(docs)
    else:
        store = FAISS.from_documents(docs, embeddings)

    thread_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(thread_dir))

    # Refresh the cache with the just-appended store so the next search sees the
    # new PDFs immediately, with no reload from disk.
    _store_cache[thread_id] = store

    _save_manifest(thread_id, list(_load_manifest(thread_id)) + [f.name for f in new_files])
    return [f.name for f in new_files]


@tool
def pdf_search(query: str, config: RunnableConfig) -> str:
    """
    Search the PDF documents the user uploaded to THIS chat.

    Use this whenever the user asks about the content of their uploaded
    files — their PDF, document, report, paper, resume, contract, etc.
    Do NOT use it for general knowledge or real-time information.

    Answer only from what this tool returns. If it reports that nothing
    relevant was found, tell the user the answer isn't in their PDFs
    instead of guessing.
    """
    # thread_id is injected by LangGraph from the run config — never seen by the LLM.
    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    if not thread_id:
        return "No chat context is available to locate uploaded PDFs."

    store = _get_store(thread_id)
    if store is None:
        return "No PDFs have been uploaded to this chat yet."

    hits = store.similarity_search_with_score(query, k=4)
    relevant = [(doc, score) for doc, score in hits if score <= _RELEVANCE_CUTOFF]
    if not relevant:
        return "No relevant information was found in the uploaded PDFs."

    blocks = []
    for doc, _score in relevant:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        where = source + (f", page {page + 1}" if isinstance(page, int) else "")
        blocks.append(f"[{where}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


tools = [
    calculator,
    web_search,
    pdf_search,
]

