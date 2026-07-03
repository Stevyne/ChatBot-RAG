"""Extraction spécialisée de champs de factures vers Excel.

Ce module traite chaque document indexé comme une facture et extrait plusieurs
champs structurés : destinataire, adresse, numéro, date, montants, etc.

Il utilise le contexte OCR/indexé déjà stocké dans PostgreSQL et un modèle local
Ollama/OpenAI selon la configuration du projet.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .generator import generate_answer
from .retriever import format_context
from .vector_store import get_chunks_by_source, list_indexed_documents

DEFAULT_INVOICE_FIELDS = [
    "destinataire_nom",
    "destinataire_adresse",
    "emetteur_nom",
    "numero_facture",
    "date_facture",
    "montant_ht",
    "montant_tva",
    "montant_ttc",
    "devise",
    "commentaire",
]


def limit_chunks_by_context_size(chunks: list[dict[str, Any]], max_chars: int = 12000) -> list[dict[str, Any]]:
    """Limite le contexte utilisé pour un document afin d'éviter une consommation RAM excessive."""
    selected: list[dict[str, Any]] = []
    total = 0
    for chunk in chunks:
        text = chunk.get("text", "")
        if total + len(text) > max_chars and selected:
            break
        selected.append(chunk)
        total += len(text)
    return selected


def build_invoice_extraction_prompt(context: str) -> str:
    """Construit un prompt spécialisé pour extraire des champs de facture en JSON."""
    return f"""
Tu es un système d'extraction d'informations dans des factures scannées.

Objectif : extraire les champs demandés à partir du contexte OCR fourni.

Règles strictes :
1. Réponds uniquement avec un objet JSON valide.
2. N'ajoute aucun texte avant ou après le JSON.
3. Si une information est absente ou incertaine, mets exactement "Non trouvé".
4. N'invente jamais une valeur.
5. Le destinataire est la personne, entreprise ou organisation à qui la facture est adressée, souvent proche de mentions comme "Client", "Facturé à", "À l'attention de", "Bill to", "Destinataire".
6. L'émetteur est l'entreprise ou la personne qui a émis la facture.
7. Conserve les montants avec leur format d'origine si possible.

Champs à extraire :
- destinataire_nom
- destinataire_adresse
- emetteur_nom
- numero_facture
- date_facture
- montant_ht
- montant_tva
- montant_ttc
- devise
- commentaire

Format JSON obligatoire :
{{
  "destinataire_nom": "...",
  "destinataire_adresse": "...",
  "emetteur_nom": "...",
  "numero_facture": "...",
  "date_facture": "...",
  "montant_ht": "...",
  "montant_tva": "...",
  "montant_ttc": "...",
  "devise": "...",
  "commentaire": "..."
}}

Contexte OCR/documentaire :
--------------------------
{context}
--------------------------

JSON :
""".strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extrait et parse le premier objet JSON trouvé dans une réponse de modèle."""
    text = text.strip()

    # Tentative directe.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Retire les blocs Markdown éventuels.
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())

    # Recherche du premier objet JSON.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {}


def normalize_invoice_result(parsed: dict[str, Any], raw_response: str) -> dict[str, str]:
    """Normalise les champs attendus."""
    result: dict[str, str] = {}
    for field in DEFAULT_INVOICE_FIELDS:
        value = parsed.get(field, "Non trouvé") if isinstance(parsed, dict) else "Non trouvé"
        if value is None or str(value).strip() == "":
            value = "Non trouvé"
        result[field] = str(value).strip()

    # Si le JSON n'a pas pu être parsé, on conserve la réponse brute pour diagnostic.
    if not parsed:
        result["commentaire"] = f"Réponse brute non parsée : {raw_response[:500]}"
    return result


def extract_invoice_fields_for_document(source: str, max_context_chars: int = 12000) -> dict[str, Any]:
    """Extrait les champs d'une facture pour un document précis."""
    chunks = get_chunks_by_source(source)
    selected_chunks = limit_chunks_by_context_size(chunks, max_chars=max_context_chars)

    base_row: dict[str, Any] = {
        "document": source,
        "pages_utilisees": "",
        "nombre_chunks_utilises": len(selected_chunks),
    }

    if not selected_chunks:
        return {**base_row, **{field: "Non trouvé" for field in DEFAULT_INVOICE_FIELDS}}

    context = format_context(selected_chunks)
    prompt = build_invoice_extraction_prompt(context)
    raw_answer = generate_answer(prompt=prompt, context=context)
    parsed = extract_json_object(raw_answer)
    normalized = normalize_invoice_result(parsed, raw_answer)

    pages = sorted({
        chunk.get("metadata", {}).get("page")
        for chunk in selected_chunks
        if chunk.get("metadata", {}).get("page") is not None
    })

    return {
        **base_row,
        **normalized,
        "pages_utilisees": "; ".join(str(page) for page in pages),
    }


def extract_invoice_fields_for_all_documents(
    selected_sources: list[str] | None = None,
    max_context_chars: int = 12000,
) -> pd.DataFrame:
    """Extrait les champs de facture pour tous les documents sélectionnés."""
    documents = list_indexed_documents()
    if selected_sources:
        selected = set(selected_sources)
        documents = [doc for doc in documents if doc["filename"] in selected]

    rows = []
    for document in documents:
        rows.append(
            extract_invoice_fields_for_document(
                source=document["filename"],
                max_context_chars=max_context_chars,
            )
        )

    columns = [
        "document",
        "destinataire_nom",
        "destinataire_adresse",
        "emetteur_nom",
        "numero_facture",
        "date_facture",
        "montant_ht",
        "montant_tva",
        "montant_ttc",
        "devise",
        "commentaire",
        "pages_utilisees",
        "nombre_chunks_utilises",
    ]
    return pd.DataFrame(rows, columns=columns)


def export_invoice_fields_to_excel(
    selected_sources: list[str] | None = None,
    output_path: str | Path | None = None,
    max_context_chars: int = 12000,
) -> Path:
    """Exporte les champs de factures dans un fichier Excel."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("data") / "exports" / f"extraction_factures_{timestamp}.xlsx"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    results_df = extract_invoice_fields_for_all_documents(
        selected_sources=selected_sources,
        max_context_chars=max_context_chars,
    )

    metadata_df = pd.DataFrame(
        [
            {"champ": "date_export", "valeur": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"champ": "nombre_documents", "valeur": len(results_df)},
            {"champ": "max_context_chars", "valeur": max_context_chars},
            {"champ": "description", "valeur": "Extraction structurée de champs de factures depuis les documents indexés."},
        ]
    )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="factures", index=False)
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
