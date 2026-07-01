"""Export des données extraites vers un fichier Excel.

Ce module exporte les données stockées dans PostgreSQL :
- documents importés ;
- chunks extraits ;
- statistiques par document.

Les embeddings complets ne sont pas exportés par défaut afin de garder un fichier Excel lisible.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .vector_store import get_connection, init_db


EXPORT_COLUMNS_CHUNKS = [
    "chunk_db_id",
    "document_id",
    "source",
    "page",
    "chunk_id",
    "content",
    "content_length",
    "embedding_dimension",
    "metadata",
    "created_at",
]


def _json_to_string(value: Any) -> str:
    """Convertit une valeur JSON/objet en texte lisible pour Excel."""
    if value is None:
        return ""
    return str(value)


def load_documents_dataframe() -> pd.DataFrame:
    """Charge la table documents depuis PostgreSQL."""
    init_db()
    with get_connection() as conn:
        query = """
        SELECT
            id,
            filename,
            file_path,
            created_at
        FROM documents
        ORDER BY id;
        """
        return pd.read_sql_query(query, conn)


def load_chunks_dataframe(include_embeddings: bool = False) -> pd.DataFrame:
    """Charge les chunks depuis PostgreSQL.

    Args:
        include_embeddings: si True, ajoute une colonne avec l'embedding complet.
            Par défaut False, car cela rend le fichier Excel très volumineux et peu lisible.
    """
    init_db()
    embedding_column = ", embedding" if include_embeddings else ""

    with get_connection() as conn:
        query = f"""
        SELECT
            id AS chunk_db_id,
            document_id,
            source,
            page,
            chunk_id,
            content,
            LENGTH(content) AS content_length,
            cardinality(embedding) AS embedding_dimension,
            metadata,
            created_at
            {embedding_column}
        FROM chunks
        ORDER BY source, page, chunk_id;
        """
        df = pd.read_sql_query(query, conn)

    if "metadata" in df.columns:
        df["metadata"] = df["metadata"].apply(_json_to_string)

    if include_embeddings and "embedding" in df.columns:
        df["embedding"] = df["embedding"].apply(lambda x: ";".join(map(str, x)) if x is not None else "")

    return df


def build_statistics_dataframe(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """Construit des statistiques simples par document."""
    if chunks_df.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "nombre_chunks",
                "nombre_pages",
                "taille_totale_texte",
                "taille_moyenne_chunk",
            ]
        )

    stats = (
        chunks_df.groupby("source")
        .agg(
            nombre_chunks=("chunk_db_id", "count"),
            nombre_pages=("page", "nunique"),
            taille_totale_texte=("content_length", "sum"),
            taille_moyenne_chunk=("content_length", "mean"),
        )
        .reset_index()
    )
    stats["taille_moyenne_chunk"] = stats["taille_moyenne_chunk"].round(2)
    return stats


def export_extracted_data_to_excel(
    output_path: str | Path | None = None,
    include_embeddings: bool = False,
) -> Path:
    """Exporte les données extraites vers un fichier Excel.

    Le fichier contient plusieurs feuilles :
    - `documents` ;
    - `chunks_extraits` ;
    - `statistiques` ;
    - `description`.
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("data") / "exports" / f"donnees_extraites_{timestamp}.xlsx"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    documents_df = load_documents_dataframe()
    chunks_df = load_chunks_dataframe(include_embeddings=include_embeddings)
    stats_df = build_statistics_dataframe(chunks_df)

    description_df = pd.DataFrame(
        [
            {
                "champ": "documents",
                "description": "Liste des fichiers PDF importés et enregistrés dans PostgreSQL.",
            },
            {
                "champ": "chunks_extraits",
                "description": "Passages textuels extraits des PDF, avec source, page et numéro de chunk.",
            },
            {
                "champ": "statistiques",
                "description": "Statistiques simples par document : nombre de chunks, pages et taille du texte.",
            },
            {
                "champ": "embedding_dimension",
                "description": "Dimension du vecteur d'embedding stocké dans PostgreSQL. L'embedding complet n'est pas exporté par défaut.",
            },
        ]
    )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        documents_df.to_excel(writer, sheet_name="documents", index=False)
        chunks_df.to_excel(writer, sheet_name="chunks_extraits", index=False, columns=[c for c in EXPORT_COLUMNS_CHUNKS if c in chunks_df.columns] + (["embedding"] if include_embeddings and "embedding" in chunks_df.columns else []))
        stats_df.to_excel(writer, sheet_name="statistiques", index=False)
        description_df.to_excel(writer, sheet_name="description", index=False)

        # Ajustement simple de la largeur des colonnes.
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column_cells in worksheet.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter
                for cell in column_cells[:100]:
                    try:
                        max_length = max(max_length, len(str(cell.value or "")))
                    except Exception:
                        pass
                worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 80)

    return output
