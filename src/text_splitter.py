"""Découpage des textes en chunks avec chevauchement."""
from __future__ import annotations

from typing import Any

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Découpe un texte en morceaux de taille fixe avec chevauchement."""
    text = text.strip()
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size doit être supérieur à overlap")

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        # Essaie de ne pas couper brutalement au milieu d'une phrase si possible.
        if end < text_length:
            last_period = max(chunk.rfind(". "), chunk.rfind("\n"), chunk.rfind("; "))
            if last_period > int(chunk_size * 0.55):
                end = start + last_period + 1
                chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = max(end - overlap, end) if overlap == 0 else end - overlap
        if start <= 0:
            start = end

    return chunks


def split_documents(
    pages: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Découpe les pages extraites en chunks avec métadonnées."""
    documents: list[dict[str, Any]] = []

    for page in pages:
        page_text = page["text"]
        metadata = page.get("metadata", {})
        chunks = split_text(page_text, chunk_size=chunk_size, overlap=overlap)

        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_metadata = dict(metadata)
            chunk_metadata["chunk_id"] = chunk_index
            chunk_metadata["chunk_size"] = len(chunk)
            documents.append({"text": chunk, "metadata": chunk_metadata})

    return documents
