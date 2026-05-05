"""
RAG Chatbot — AI Workshop Day 2
Streamlit + Ollama version (with model selector & auto-pull)

Usage:
    streamlit run app.py

Requirements:
    pip install streamlit chromadb sentence-transformers requests

Ollama setup:
    1. Install from https://ollama.ai
    2. ollama serve
"""

import requests
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL  = "http://localhost:11434"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE       = 300
CHUNK_OVERLAP    = 40
MAX_NEW_TOKENS   = 512

# Popular models users can pull directly from the UI
AVAILABLE_MODELS = [
    "mistral",
    "llama3.2",
    "llama3.2:1b",
    "gemma:2b",
    "gemma2",
    "phi3",
    "phi3:mini",
    "qwen2:1.5b",
    "deepseek-r1:1.5b",
    "tinyllama",
]

# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def get_chroma_client():
    return chromadb.Client()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "messages":      [],
        "collection":    None,
        "doc_name":      "",
        "doc_indexed":   False,
        "active_model":  None,   # currently selected & confirmed model
        "pull_log":      "",     # streaming pull progress text
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_pulled_models() -> list[str]:
    """Return list of model names already pulled locally."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def pull_model(model_name: str, status_placeholder):
    """
    Pull a model from Ollama registry with streaming progress.
    Updates status_placeholder in real time.
    Returns (success: bool, message: str).
    """
    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=600,
        ) as r:
            r.raise_for_status()
            last_status = ""
            for line in r.iter_lines():
                if not line:
                    continue
                import json
                data = json.loads(line)
                status = data.get("status", "")

                # Show download progress if available
                total     = data.get("total", 0)
                completed = data.get("completed", 0)
                if total and completed:
                    pct = int(completed / total * 100)
                    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                    status_placeholder.info(f"⬇️ `{model_name}` — {status} `{bar}` {pct}%")
                else:
                    if status != last_status:
                        status_placeholder.info(f"⬇️ `{model_name}` — {status}")
                last_status = status

        return True, f"✅ `{model_name}` pulled successfully."
    except Exception as e:
        return False, f"❌ Pull failed: {e}"


def call_ollama(prompt: str, system_prompt: str, temperature: float, model: str) -> str:
    payload = {
        "model":    model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        "stream":  False,
        "options": {"temperature": temperature, "num_predict": MAX_NEW_TOKENS},
    }
    r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

# ─────────────────────────────────────────────────────────────────────────────
# TEXT PROCESSING & RAG
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + CHUNK_SIZE]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def index_document(text: str, doc_name: str, embed_model) -> tuple[int, str]:
    client = get_chroma_client()
    try:
        client.delete_collection("rag_docs")
    except Exception:
        pass
    collection = client.create_collection("rag_docs")

    chunks = chunk_text(text)
    if not chunks:
        return 0, "⚠️ Document appears empty."

    embeddings = embed_model.encode(chunks, show_progress_bar=False)
    collection.add(
        documents  = chunks,
        embeddings = embeddings.tolist(),
        ids        = [f"chunk_{i}" for i in range(len(chunks))],
    )

    st.session_state.collection  = collection
    st.session_state.doc_name    = doc_name
    st.session_state.doc_indexed = True
    return len(chunks), f"✅ Indexed **{doc_name}** → {len(chunks)} chunks"


def retrieve_chunks(question: str, k: int, embed_model) -> tuple[list[str], list[float]]:
    q_vec   = embed_model.encode(question).tolist()
    results = st.session_state.collection.query(query_embeddings=[q_vec], n_results=k)
    chunks  = results["documents"][0]
    scores  = [1 - d for d in results["distances"][0]]
    return chunks, scores


def build_rag_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    return (
        f"Answer the question using ONLY the context provided below.\n"
        f"If the answer is not in the context, say: "
        f"\"I don't have that information in the provided document.\"\n"
        f"Do not make up any facts. Be specific and concise.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title = "RAG Chatbot — AI Workshop Day 2",
        page_icon  = "🧠",
        layout     = "wide",
    )

    init_state()
    embed_model = load_embed_model()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🧠 RAG Chatbot")
        st.caption("AI Workshop · Day 2")
        st.divider()

        # ── Ollama status ─────────────────────────────────────────────────────
        st.subheader("⚙️ Ollama")

        if not ollama_running():
            st.error("🔴 Ollama not running")
            with st.expander("How to fix"):
                st.code("ollama serve", language="bash")
            st.stop()

        st.success("🟢 Ollama is running")

        # ── Model selector ────────────────────────────────────────────────────
        st.subheader("🤖 Model")

        pulled_models = get_pulled_models()

        # Build dropdown: pulled models first, then the rest
        pulled_set    = {m.split(":")[0] for m in pulled_models}
        dropdown_opts = []

        for m in pulled_models:
            dropdown_opts.append(f"✅ {m}")                    # already local

        for m in AVAILABLE_MODELS:
            base = m.split(":")[0]
            if base not in pulled_set and m not in pulled_models:
                dropdown_opts.append(f"⬇️ {m}")               # needs pulling

        # Let user also type a custom model name
        dropdown_opts.append("✏️  Custom model name…")

        selected = st.selectbox(
            "Choose a model",
            dropdown_opts,
            help="✅ = already downloaded  ·  ⬇️ = will be pulled on use",
        )

        # Resolve model name
        if selected.startswith("✏️"):
            custom_model = st.text_input(
                "Enter model name",
                placeholder="e.g. llama3:8b",
            )
            chosen_model = custom_model.strip()
        elif selected.startswith("✅ "):
            chosen_model = selected[len("✅ "):].strip()
        elif selected.startswith("⬇️ "):
            chosen_model = selected[len("⬇️ "):].strip()
        else:
            chosen_model = selected.strip()

        needs_pull = (
            chosen_model
            and chosen_model not in pulled_models
            and chosen_model.split(":")[0] not in pulled_set
        )

        # Pull button (only shown when model isn't local)
        if chosen_model and needs_pull:
            st.warning(f"`{chosen_model}` is not pulled yet.")
            if st.button(f"⬇️ Pull `{chosen_model}`", use_container_width=True, type="primary"):
                pull_placeholder = st.empty()
                ok, pull_msg = pull_model(chosen_model, pull_placeholder)
                if ok:
                    pull_placeholder.success(pull_msg)
                    st.session_state.active_model = chosen_model
                    st.rerun()
                else:
                    pull_placeholder.error(pull_msg)
        elif chosen_model:
            st.session_state.active_model = chosen_model
            st.success(f"Using `{chosen_model}`")

        active_model = st.session_state.active_model

        st.divider()

        # ── Document upload ───────────────────────────────────────────────────
        st.subheader("📄 Document")
        uploaded_file = st.file_uploader(
            "Upload a .txt or .md file",
            type=["txt", "md"],
        )

        if uploaded_file:
            if st.button("📥 Index Document", use_container_width=True, type="primary"):
                text = uploaded_file.read().decode("utf-8", errors="replace")
                if len(text.strip()) < 50:
                    st.warning("File seems too short or empty.")
                else:
                    with st.spinner("Chunking & embedding…"):
                        n, status = index_document(text, uploaded_file.name, embed_model)
                    if n:
                        st.success(status)
                        with st.expander("Document preview"):
                            st.text(text[:500] + "…")
                    else:
                        st.error(status)

        if st.session_state.doc_indexed:
            st.info(f"📖 **{st.session_state.doc_name}**")

        st.divider()

        # ── Settings ──────────────────────────────────────────────────────────
        st.subheader("🎛️ Settings")
        temperature = st.slider(
            "Temperature", 0.1, 1.5, 0.3, 0.1,
            help="0.1 = focused  ·  1.5 = creative",
        )
        top_k = st.slider(
            "Chunks to retrieve (k)", 1, 6, 3, 1,
            help="More chunks = more context, slower response",
        )
        show_sources = st.checkbox("Show retrieved chunks", value=True)

        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Main chat area ────────────────────────────────────────────────────────
    st.header("💬 Chat with your document")

    if not active_model:
        st.warning("👈 Select a model in the sidebar to get started.")
        st.stop()

    if not st.session_state.doc_indexed:
        st.info("👈 Upload and index a document in the sidebar to get started.")

    # Example question buttons
    st.caption("Try an example:")
    cols = st.columns(4)
    examples = [
        "What is the main topic?",
        "List key statistics.",
        "What does it say about AI?",
        "Any education content?",
    ]
    for col, q in zip(cols, examples):
        if col.button(q, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

    st.divider()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Retrieved chunks"):
                    st.markdown(msg["sources"])

    # Chat input
    if prompt := st.chat_input(
        "Ask a question about your document…",
        disabled=not st.session_state.doc_indexed,
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with `{active_model}`…"):
                try:
                    chunks, scores = retrieve_chunks(prompt, top_k, embed_model)
                    rag_prompt = build_rag_prompt(prompt, chunks)
                    sys_prompt = (
                        "You are a precise document assistant. "
                        "Only answer from the provided context. "
                        "Never fabricate information. If unsure, say so."
                    )
                    answer = call_ollama(rag_prompt, sys_prompt, temperature, active_model)
                except Exception as e:
                    answer = f"❌ Error: {e}"
                    chunks = []
                    scores = []

            st.markdown(answer)

            sources_md = ""
            if show_sources and chunks:
                for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
                    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                    sources_md += (
                        f"**Chunk {i}** — Relevance: `{score:.3f}` `{bar}`\n\n"
                        f"> {chunk[:300]}{'…' if len(chunk) > 300 else ''}\n\n---\n\n"
                    )
                with st.expander("📚 Retrieved chunks"):
                    st.markdown(sources_md)

        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources_md,
        })


if __name__ == "__main__":
    main()