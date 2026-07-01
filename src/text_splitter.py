"""Découpage des textes en chunks avec chevauchement.

Correction importante : lorsque la fin du texte est atteinte, la boucle doit
s'arrêter. Sinon, avec un chevauchement non nul, le dernier morceau peut être
répété indéfiniment et provoquer une explosion de la RAM.
"""
from __future__ import annotations

from typing import Any

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Découpe un texte en morceaux de taille fixe avec chevauchement.

    Args:
        text: texte à découper.
        chunk_size: taille maximale d'un chunk en caractères.
        overlap: nombre de caractères repris entre deux chunks.

    Returns:
        Liste des chunks.
    """
    text = text.strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size doit être positif")
    if overlap < 0:
        raise ValueError("overlap ne peut pas être négatif")
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
            last_separator = max(chunk.rfind(". "), chunk.rfind("\n"), chunk.rfind("; "))
            if last_separator > int(chunk_size * 0.55):
                end = start + last_separator + 1
                chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # IMPORTANT : si la fin est atteinte, on arrête.
        # Sans ce break, start=end-overlap peut rester inférieur à text_length
        # et répéter le dernier chunk indéfiniment.
        if end >= text_length:
            break

        start = max(end - overlap, 0) if overlap > 0 else end

        # Sécurité supplémentaire contre les boucles bloquées.
        if chunks and start >= end:
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
