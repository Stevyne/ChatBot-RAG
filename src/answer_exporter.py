"""Extraction de réponses par document et export vers Excel.

Cas d'usage : importer plusieurs factures, poser une question comme
"Quel est le nom de la personne à qui la facture est adressée ?", puis exporter
les réponses dans un fichier Excel.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .generator import build_prompt, generate_answer
from .retriever import format_context
from .vector_store import get_chunks_by_source, list_indexed_documents


def limit_chunks_by_context_size(chunks: list[dict[str, Any]], max_chars: int = 8000) -> list[dict[str, Any]]:
    """Limite le nombre de chunks utilisés pour ne pas dépasser un contexte raisonnable."""
    selected: list[dict[str, Any]] = []
    total = 0
    for chunk in chunks:
        text = chunk.get("text", "")
        if total + len(text) > max_chars and selected:
            break
        selected.append(chunk)
        total += len(text)
    return selected


def build_extraction_question(user_question: str) -> str:
    """Renforce la question pour obtenir une réponse courte et exploitable en Excel."""
    return f"""
{user_question}

Consigne de format :
- Donne uniquement l'information demandée, de façon courte.
- Si plusieurs valeurs existent, sépare-les par un point-virgule.
- Si l'information est absente, réponds exactement : Non trouvé.
- N'ajoute pas d'explication inutile.
""".strip()


def extract_answer_for_document(
    source: str,
    question: str,
    max_context_chars: int = 8000,
) -> dict[str, Any]:
    """Pose une question sur un document précis et retourne une ligne exportable."""
    chunks = get_chunks_by_source(source)
    selected_chunks = limit_chunks_by_context_size(chunks, max_chars=max_context_chars)

    if not selected_chunks:
        return {
            "document": source,
            "question": question,
            "reponse": "Non trouvé",
            "pages_utilisees": "",
            "nombre_chunks_utilises": 0,
        }

    extraction_question = build_extraction_question(question)
    context = format_context(selected_chunks)
    prompt = build_prompt(context=context, question=extraction_question)
    answer = generate_answer(prompt=prompt, context=context).strip()

    if "Je ne trouve pas cette information" in answer:
        answer = "Non trouvé"

    pages = sorted({chunk.get("metadata", {}).get("page") for chunk in selected_chunks if chunk.get("metadata", {}).get("page") is not None})

    return {
        "document": source,
        "question": question,
        "reponse": answer,
        "pages_utilisees": "; ".join(str(page) for page in pages),
        "nombre_chunks_utilises": len(selected_chunks),
    }


def extract_answers_for_all_documents(
    question: str,
    selected_sources: list[str] | None = None,
    max_context_chars: int = 8000,
) -> pd.DataFrame:
    """Extrait une réponse pour chaque document indexé."""
    documents = list_indexed_documents()
    if selected_sources:
        selected_set = set(selected_sources)
        documents = [doc for doc in documents if doc["filename"] in selected_set]

    rows = []
    for document in documents:
        rows.append(
            extract_answer_for_document(
                source=document["filename"],
                question=question,
                max_context_chars=max_context_chars,
            )
        )
    return pd.DataFrame(rows)


def export_answers_to_excel(
    question: str,
    selected_sources: list[str] | None = None,
    output_path: str | Path | None = None,
    max_context_chars: int = 8000,
) -> Path:
    """Exporte les réponses extraites vers un fichier Excel."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("data") / "exports" / f"reponses_extraites_{timestamp}.xlsx"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    results_df = extract_answers_for_all_documents(
        question=question,
        selected_sources=selected_sources,
        max_context_chars=max_context_chars,
    )

    metadata_df = pd.DataFrame(
        [
            {"champ": "question", "valeur": question},
            {"champ": "date_export", "valeur": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"champ": "nombre_documents", "valeur": len(results_df)},
            {"champ": "max_context_chars", "valeur": max_context_chars},
        ]
    )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="reponses", index=False)
        metadata_df.to_excel(writer, sheet_name="metadata", index=False)

        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column_cells in worksheet.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter
                for cell in column_cells[:100]:
                    max_length = max(max_length, len(str(cell.value or "")))
                worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 90)

    return output
