"""Création des embeddings avec SentenceTransformers."""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Charge le modèle d'embeddings une seule fois."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Transforme une liste de textes en vecteurs."""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(question: str) -> list[float]:
    """Transforme une question en vecteur."""
    return embed_texts([question])[0]


def vector_to_pgvector(vector: list[float]) -> str:
    """Convertit une liste Python en représentation acceptée par pgvector."""
    return "[" + ",".join(str(float(x)) for x in vector) + "]"
