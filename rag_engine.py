"""
RAG Engine: Handles document ingestion and question answering.

Pipeline:
  1. Load PDF -> extract text per page
  2. Split text into overlapping chunks (preserves context)
  3. Convert chunks into vector embeddings (semantic representation)
  4. Store vectors in FAISS (local vector index, persisted to disk)
  5. On question: embed it, find similar chunks, send chunks + question to LLM
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4


def get_embeddings():
    """Local embedding model — free, no API calls, runs on CPU."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_index():
    """Load existing FAISS index from disk, or return None if not present."""
    if not os.path.exists(INDEX_DIR):
        return None
    return FAISS.load_local(
        INDEX_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def ingest_pdf(pdf_path: str) -> int:
    """Extract, chunk, embed, and store a PDF. Returns number of chunks created."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # each page becomes a Document with page metadata

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    # Tag each chunk with its source filename for citations
    filename = Path(pdf_path).name
    for chunk in chunks:
        chunk.metadata["source"] = filename

    existing = _load_index()
    if existing is None:
        vectorstore = FAISS.from_documents(chunks, get_embeddings())
    else:
        existing.add_documents(chunks)
        vectorstore = existing

    vectorstore.save_local(INDEX_DIR)
    return len(chunks)


def ask_question(question: str) -> dict:
    """Run a question through the RAG pipeline. Returns answer + sources."""
    prompt = PromptTemplate.from_template(
        """You are DocuMind AI — an expert document analyst. Your job is to read the provided context carefully and give thorough, well-structured answers.

RULES:
- Answer in detail. Never give one-liners unless the question itself is trivial.
- Structure your answer with clear paragraphs. Use bullet points or numbered lists when listing multiple items or steps.
- If the question asks "how", explain the process step by step.
- If the question asks "what", define it clearly and expand with relevant details from the context.
- Always end your answer with a "📄 Sources" line citing the page number(s) you referenced, e.g. "📄 Sources: Page 3, Page 7".
- If the answer is not found in the context, say: "I could not find this information in the uploaded document. Try uploading a more relevant file or rephrasing your question."
- Never make up information. Only use what is in the context below.

Context:
{context}

Question: {question}

Answer:"""
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    vectorstore = _load_index()
    if vectorstore is None:
        raise RuntimeError("No documents indexed yet. Upload a PDF first.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    retrieved_docs = []

    def retrieve_and_store(question):
        docs = retriever.invoke(question)
        retrieved_docs.extend(docs)
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retrieve_and_store, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    sources = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page")
        page_display = page + 1 if isinstance(page, int) else "?"
        sources.append({
            "page": page_display,
            "source": doc.metadata.get("source", "unknown"),
            "content": doc.page_content,
        })

    return {
        "answer": answer,
        "sources": sources,
    }


def has_documents() -> bool:
    """Check if any documents have been ingested."""
    return os.path.exists(INDEX_DIR) and os.path.isfile(os.path.join(INDEX_DIR, "index.faiss"))


def clear_database():
    """Wipe the vector store — useful when uploading a new handbook."""
    import shutil
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
