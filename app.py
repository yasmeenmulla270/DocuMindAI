import os
from pathlib import Path
import streamlit as st

from rag_engine import (
    ingest_pdf,
    ask_question,
    has_documents,
    clear_database,
)

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
/* ── Global background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}

/* ── Hide Streamlit top header bar ── */
[data-testid="stHeader"],
[data-testid="stToolbar"],
header[data-testid="stHeader"] {
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
}

/* ── Remove top white gradient decoration ── */
[data-testid="stDecoration"],
[data-testid="stDecorationColorBackground"] {
    display: none !important;
}

/* ── Fix main block container top padding ── */
.block-container {
    padding-top: 1rem !important;
}

/* ── Fix white bottom bar behind chat input ── */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
.stBottom,
.stBottom > div {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    background-color: #0f0c29 !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── Main app bottom section ── */
.main > div:last-child,
section.main > div {
    background: transparent !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0;
}

/* ── Stat badges ── */
.badge-row {
    display: flex;
    justify-content: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1.2rem 0 2rem;
}
.badge {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 999px;
    padding: 0.35rem 1rem;
    font-size: 0.8rem;
    color: #a5b4fc;
    backdrop-filter: blur(6px);
}

/* ── Empty state card ── */
.empty-card {
    background: rgba(255,255,255,0.04);
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 2rem auto;
    max-width: 520px;
}
.empty-card h3 { color: #c4b5fd; margin-bottom: 0.5rem; }
.empty-card p  { color: #64748b; }

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px !important;
    margin-bottom: 0.8rem;
    backdrop-filter: blur(6px);
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: #f1f5f9 !important;
    font-size: 0.97rem;
    line-height: 1.75;
}
[data-testid="stChatMessage"] strong {
    color: #c4b5fd !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #f1f5f9 !important;
    line-height: 1.75;
}

/* ── Chat input ── */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInputContainer"],
[data-testid="stChatInputContainer"] > div {
    background: rgba(20, 15, 50, 0.9) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea {
    background: rgba(20, 15, 50, 0.9) !important;
    border: 1px solid rgba(167,139,250,0.5) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    caret-color: #a78bfa !important;
    font-size: 0.97rem !important;
}
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInputContainer"] textarea::placeholder {
    color: #64748b !important;
}
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInputContainer"] textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 2px rgba(167,139,250,0.2) !important;
    outline: none !important;
}

/* ── Expander (sources) ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}

/* ── File uploader — target every layer ── */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] > div > div,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileDropzone"],
section[data-testid="stFileUploader"] div {
    background: rgba(30, 20, 60, 0.6) !important;
    border-color: rgba(167,139,250,0.4) !important;
    border-radius: 12px !important;
    color: #94a3b8 !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {
    color: #94a3b8 !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(124, 58, 237, 0.35) !important;
    border: 1px solid rgba(167,139,250,0.5) !important;
    border-radius: 8px !important;
    color: #c4b5fd !important;
}
[data-testid="stFileUploader"] button:hover {
    background: rgba(124, 58, 237, 0.55) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Success / error ── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Indexed file pill ── */
.file-pill {
    display: inline-block;
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.78rem;
    color: #c4b5fd;
    margin: 0.15rem 0;
}

/* ── Sidebar caption ── */
.sidebar-footer {
    font-size: 0.72rem;
    color: #475569;
    text-align: center;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingested_files" not in st.session_state:
        if has_documents():
            st.session_state.ingested_files = ["Previously indexed document(s)"]
        else:
            st.session_state.ingested_files = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 DocuMind AI")
        st.markdown("<span style='color:#64748b;font-size:0.85rem'>Powered by Llama 3.3 · FAISS · LangChain</span>", unsafe_allow_html=True)
        st.divider()

        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not found. Add it to your `.env` file.")
            st.stop()

        st.markdown("**📂 Upload Document**")
        uploaded = st.file_uploader(
            "Drop a PDF here",
            type=["pdf"],
            help="Your PDF will be split into chunks, embedded, and stored locally.",
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}",
        )

        if uploaded is not None and uploaded.name not in st.session_state.ingested_files:
            save_path = DATA_DIR / uploaded.name
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            with st.spinner(f"Indexing {uploaded.name}..."):
                num_chunks = ingest_pdf(str(save_path))
            st.session_state.ingested_files.append(uploaded.name)
            st.success(f"✅ {num_chunks} chunks indexed!")

        if st.session_state.ingested_files:
            st.markdown("**📑 Indexed files**")
            for f in st.session_state.ingested_files:
                st.markdown(f'<div class="file-pill">📄 {f}</div>', unsafe_allow_html=True)

        st.divider()
        if st.button("🗑️ Clear all documents", use_container_width=True):
            clear_database()
            st.session_state.ingested_files = []
            st.session_state.messages = []
            st.session_state.uploader_key += 1
            st.rerun()

        st.markdown('<div class="sidebar-footer">Built with ❤️ using LangChain · FAISS · Groq</div>', unsafe_allow_html=True)


def render_chat():
    st.markdown("""
    <div class="hero">
        <h1>🧠 DocuMind AI</h1>
        <p>Upload any PDF and get instant, accurate, page-cited answers — powered by AI.</p>
    </div>
    <div class="badge-row">
        <span class="badge">⚡ Llama 3.3 70B</span>
        <span class="badge">🔍 Semantic Search</span>
        <span class="badge">📄 Page Citations</span>
        <span class="badge">🔒 Local Embeddings</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.ingested_files:
        st.markdown("""
        <div class="empty-card">
            <h3>👈 No document loaded yet</h3>
            <p>Upload a PDF from the sidebar to start chatting with your document.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 View Sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{i}] {src['source']} — Page {src['page']}**")
                        st.markdown(f"> {src['content'][:300]}...")

    if question := st.chat_input("Ask anything about your document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("🧠 DocuMind is thinking..."):
                result = ask_question(question)
            st.markdown(result["answer"])
            with st.expander("📄 View Sources"):
                for i, src in enumerate(result["sources"], 1):
                    st.markdown(f"**[{i}] {src['source']} — Page {src['page']}**")
                    st.markdown(f"> {src['content'][:300]}...")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })


def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
