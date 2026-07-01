"""Gestion de la base PostgreSQL sans pgvector.

Cette version stocke les embeddings dans une colonne DOUBLE PRECISION[]
et calcule la similarité cosinus en Python. Elle est adaptée à un prototype local
et à un corpus de taille petite ou moyenne.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

from .config import DATABASE_URL
from .embeddings import embed_query, embed_texts


def _import_psycopg():
    """Import tardif pour afficher une erreur claire si la dépendance manque."""
    try:
        import psycopg
        from psycopg.types.json import Jsonb
    except ImportError as exc:
        raise ImportError(
            "Le paquet psycopg n'est pas installé. Lancez : pip install -r requirements.txt"
        ) from exc
    return psycopg, Jsonb


def get_connection():
    """Ouvre une connexion PostgreSQL."""
    psycopg, _ = _import_psycopg()
    return psycopg.connect(DATABASE_URL)


def init_db() -> None:
    """Crée les tables nécessaires dans PostgreSQL, sans extension pgvector."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL UNIQUE,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    source TEXT NOT NULL,
                    page INTEGER,
                    chunk_id INTEGER,
                    content TEXT NOT NULL,
                    embedding DOUBLE PRECISION[] NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
                CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON chunks USING GIN(metadata);
                """
            )
        conn.commit()


def make_chunk_id(text: str, metadata: dict[str, Any]) -> str:
    """Crée un identifiant stable pour un chunk."""
    base = f"{metadata.get('source','unknown')}|{metadata.get('page','?')}|{metadata.get('chunk_id','?')}|{text[:120]}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _get_or_create_document(cur, source: str, file_path: str | None) -> int:
    cur.execute(
        """
        INSERT INTO documents(filename, file_path)
        VALUES (%s, %s)
        ON CONFLICT(filename) DO UPDATE SET file_path = EXCLUDED.file_path
        RETURNING id;
        """,
        (source, file_path),
    )
    return cur.fetchone()[0]


def add_documents(chunks: list[dict[str, Any]]) -> int:
    """Ajoute des chunks et leurs embeddings dans PostgreSQL."""
    if not chunks:
        return 0

    init_db()
    _, Jsonb = _import_psycopg()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    with get_connection() as conn:
        with conn.cursor() as cur:
            for chunk, embedding in zip(chunks, embeddings):
                text = chunk["text"]
                metadata = dict(chunk.get("metadata", {}))
                source = str(metadata.get("source", "document inconnu"))
                file_path = metadata.get("file_path")
                page = metadata.get("page")
                chunk_number = metadata.get("chunk_id")
                chunk_db_id = make_chunk_id(text, metadata)
                document_id = _get_or_create_document(cur, source, file_path)

                cur.execute(
                    """
                    INSERT INTO chunks(id, document_id, source, page, chunk_id, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata;
                    """,
                    (
                        chunk_db_id,
                        document_id,
                        source,
                        page,
                        chunk_number,
                        text,
                        [float(x) for x in embedding],
                        Jsonb(metadata),
                    ),
                )
        conn.commit()
    return len(chunks)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calcule la similarité cosinus entre deux vecteurs."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def search_similar(question: str, top_k: int) -> list[dict[str, Any]]:
    """Recherche les chunks les plus proches de la question en Python."""
    init_db()
    query_embedding = embed_query(question)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, source, page, chunk_id, metadata, embedding
                FROM chunks;
                """
            )
            rows = cur.fetchall()

    scored_results: list[dict[str, Any]] = []
    for row in rows:
        chunk_id, content, source, page, chunk_number, metadata, embedding = row
        embedding_list = [float(x) for x in embedding]
        score = cosine_similarity(query_embedding, embedding_list)
        scored_results.append(
            {
                "id": chunk_id,
                "text": content,
                "metadata": {
                    **(metadata or {}),
                    "source": source,
                    "page": page,
                    "chunk_id": chunk_number,
                },
                "distance": 1.0 - score,
                "score": score,
            }
        )

    scored_results.sort(key=lambda item: item["score"], reverse=True)
    return scored_results[:top_k]


def reset_vectorstore() -> None:
    """Vide les tables de documents et chunks."""
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE chunks, documents RESTART IDENTITY CASCADE;")
        conn.commit()


def count_documents() -> int:
    """Nombre de chunks indexés."""
    init_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks;")
            return int(cur.fetchone()[0])
