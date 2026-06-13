# 🧠 DocuMind AI

> Upload any PDF. Ask anything. Get instant, accurate, **page-cited answers** — powered by Llama 3.3 and semantic search.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-purple?style=flat-square)

---

## What is DocuMind AI?

DocuMind AI is a **Retrieval-Augmented Generation (RAG)** application that lets you chat with any PDF document. Upload a research paper, policy handbook, contract, or manual — and ask questions in plain English. Every answer is grounded in the document and cites the exact page it came from.

---

## Features

- **Semantic Search** — finds relevant passages even when wording doesn't match exactly
- **Page Citations** — every answer references the exact page(s) from the source document
- **Local Embeddings** — no API cost for embedding; runs fully on CPU via `all-MiniLM-L6-v2`
- **Persistent Index** — FAISS index saved to disk; survives page refreshes
- **Dark UI** — polished glassmorphism interface built with Streamlit + custom CSS
- **Detailed Answers** — structured, multi-paragraph responses (not one-liners)

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit + custom CSS |
| Orchestration | LangChain (LCEL) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector Store | FAISS (persisted to disk) |
| LLM | Groq — Llama 3.3 70B Versatile |
| PDF Parsing | PyPDF |

---

## Architecture

```
┌─────────────┐
│  PDF Upload │
└──────┬──────┘
       │
       ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  PyPDF      │───▶│  Recursive   │───▶│ HuggingFace  │
│  Loader     │    │  Splitter    │    │  Embeddings  │
└─────────────┘    │ (1000/200)   │    └──────┬───────┘
                   └──────────────┘           │
                                              ▼
                                       ┌─────────────┐
                                       │    FAISS    │
                                       │  (on disk)  │
                                       └──────┬──────┘
                                              │
   ┌──────────┐    ┌─────────────┐    ┌──────▼───────┐    ┌──────────────┐
   │ Question │───▶│ Embed query │───▶│  Top-K       │───▶│ Llama 3.3   │
   └──────────┘    └─────────────┘    │  Retrieval   │    │ + Prompt    │
                                       └──────────────┘    └──────┬──────┘
                                                                  │
                                                                  ▼
                                                          ┌──────────────┐
                                                          │ Answer +     │
                                                          │ Page Cites   │
                                                          └──────────────┘
```

---

## Setup

### 1. Clone & create virtual environment

```bash
git clone https://github.com/your-username/DocuMindAI.git
cd DocuMindAI
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com/) — it's free.

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload a PDF, and start asking questions.

---

## Project Structure

```
DocuMindAI/
├── app.py              # Streamlit UI + custom CSS
├── rag_engine.py       # RAG pipeline (ingest, embed, retrieve, generate)
├── requirements.txt
├── .env.example
├── .gitignore
├── data/               # Uploaded PDFs (gitignored)
└── faiss_index/        # Vector store (gitignored, auto-created)
```

---

## Key Design Decisions

**Chunking** — `RecursiveCharacterTextSplitter` with 1000-char chunks and 200-char overlap preserves context across page boundaries.

**Embeddings** — `all-MiniLM-L6-v2` produces 384-dim vectors, runs on CPU, and is completely free. No OpenAI embedding costs.

**Retrieval** — Top-4 semantic search via cosine similarity. Retrieved chunks are injected into the prompt as context.

**Hallucination control** — The prompt explicitly instructs the LLM to answer only from retrieved context and respond with "I could not find this information" otherwise.

**LCEL chain** — Uses LangChain Expression Language instead of the deprecated `RetrievalQA`, compatible with the latest `langchain-core`.

---

## Possible Extensions

- Multi-document support with per-document filtering
- Conversational memory (follow-up questions)
- Reranking with a cross-encoder
- Evaluation with RAGAS (faithfulness + answer relevance metrics)
- Deploy to Streamlit Cloud or Hugging Face Spaces
