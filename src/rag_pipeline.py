"""Pipeline principal d'indexation et de question-réponse."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import CHUNK_OVERLAP, CHUNK_SIZE, TOP_K
from .document_loader import load_documents
from .generator import build_prompt, generate_answer
from .retriever import extract_sources, format_context, retrieve
from .text_splitter import split_documents
from .vector_store import add_documents, count_documents


def index_documents(file_paths: list[str | Path]) -> dict[str, Any]:
    """Extrait, découpe et indexe une liste de PDF ou d'images scannées."""
    pages = load_documents(file_paths)
    chunks = split_documents(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    indexed_count = add_documents(chunks)

    return {
        "files": [str(path) for path in file_paths],
        "pages_extracted": len(pages),
        "chunks_created": len(chunks),
        "chunks_indexed": indexed_count,
        "total_chunks_in_vectorstore": count_documents(),
    }


def answer_question(question: str, top_k: int = TOP_K) -> dict[str, Any]:
    """Répond à une question avec le pipeline RAG."""
    question = question.strip()
    if not question:
        return {
            "answer": "Veuillez saisir une question.",
            "sources": [],
            "retrieved_contexts": [],
        }

    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        return {
            "answer": "Aucun document n'est indexé ou aucun passage pertinent n'a été trouvé.",
            "sources": [],
            "retrieved_contexts": [],
        }

    context = format_context(chunks)
    prompt = build_prompt(context=context, question=question)
    answer = generate_answer(prompt=prompt, context=context)
    sources = extract_sources(chunks)

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_contexts": chunks,
    }