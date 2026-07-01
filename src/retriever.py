"""Recherche sémantique dans la base vectorielle."""
from __future__ import annotations

from typing import Any

from .config import TOP_K
from .vector_store import search_similar


def distance_to_score(distance: float | None) -> float:
    """Convertit une distance Chroma en score indicatif entre 0 et 1."""
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + max(distance, 0.0))


def retrieve(question: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Retourne les passages les plus pertinents pour une question depuis PostgreSQL/pgvector."""
    return search_similar(question, top_k=top_k)


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Construit le contexte textuel envoyé au LLM."""
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "document inconnu")
        page = metadata.get("page", "?")
        text = chunk.get("text", "")
        parts.append(f"[Source {index} — {source}, page {page}]\n{text}")
    return "\n\n".join(parts)


def extract_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retourne une liste dédupliquée des sources."""
    seen = set()
    sources: list[dict[str, Any]] = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        key = (metadata.get("source"), metadata.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "source": metadata.get("source", "document inconnu"),
                    "page": metadata.get("page", "?"),
                    "score": round(float(chunk.get("score", 0.0)), 3),
                }
            )
    return sources
