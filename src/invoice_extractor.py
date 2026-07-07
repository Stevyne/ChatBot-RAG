"""Extraction structurée et universelle de champs pour tout type de facture ou document.

Ce module permet d'analyser des documents scannés ou textuels de formats très hétérogènes.
Il offre deux modes d'extraction :
1. Mode standard / personnalisé : extraction sur un schéma de colonnes défini par l'utilisateur.
2. Mode universel (Auto-découverte) : le modèle extrait librement toutes les paires clé-valeur
   pertinentes trouvées dans le document, s'adaptant ainsi à n'importe quelle facture (Français & Malagasy).
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


def build_custom_extraction_prompt(fields: list[str], context: str) -> str:
    """Construit un prompt spécialisé pour extraire une liste personnalisée de champs."""
    fields_list = "\n".join(f"- {f}" for f in fields)
    json_template = ",\n  ".join(f'"{f}": "..."' for f in fields)

    return f"""
Tu es un système d'extraction d'informations dans des factures et documents scannés en Français et en Malagasy.

Objectif : extraire les champs demandés à partir du contexte OCR fourni.

Règles strictes :
1. Réponds uniquement avec un objet JSON valide.
2. N'ajoute aucun texte avant ou après le JSON.
3. Si une information est absente ou incertaine, mets exactement "Non trouvé".
4. N'invente jamais une valeur absente du texte.
5. Tiens compte du vocabulaire malgache (ex: Anarana = Nom, Daty = Date, Vidiny = Prix, Ariary = Devise MGA).

Champs à extraire :
{fields_list}

Format JSON attendu :
{{
  {json_template}
}}

Contexte documentaire :
--------------------------
{context}
--------------------------

JSON :
""".strip()


def build_universal_extraction_prompt(context: str) -> str:
    """Construit un prompt universel qui découvre automatiquement les champs importants de toute facture."""
    return f"""
Tu es un expert en analyse documentaire et extraction structurée de factures (en Français et en Malagasy / Malgache).

Objectif : analyser le contexte documentaire ci-dessous et extraire TOUTES les informations financières, administratives et d'identification pertinentes sous la forme d'un objet JSON plat (clé-valeur).

Règles strictes :
1. Réponds STRICTEMENT par un objet JSON valide, sans aucun texte autour.
2. N'invente aucune information non présente.
3. Utilise des clés claires en snake_case (ex: emetteur, destinataire, numero_facture, date_facture, montant_ht, montant_ttc, devise, nif, stat...).
4. Comprends les mentions malgaches (ex: Faktiora, Anarana, Mpanjifa, Tompony, Daty, Vidiny, Ariary / Ar / MGA, NIF, STAT, Kaominina, Fokontany...).

Contexte de la facture / document :
--------------------------
{context}
--------------------------

Objet JSON complet :
""".strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extrait et parse le premier objet JSON trouvé dans une réponse du modèle."""
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())

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


def calculate_confidence_score(parsed: dict[str, Any], raw_text: str) -> float:
    """Calcule un indice de fiabilité (%) de l'extraction par vérification et regex."""
    if not parsed or not isinstance(parsed, dict):
        return 0.0

    total_keys = len(parsed)
    if total_keys == 0:
        return 0.0

    found_count = 0
    regex_bonus = 0.0

    for k, v in parsed.items():
        val_str = str(v).strip()
        if val_str and val_str.lower() not in {"non trouvé", "inconnu", "null", "none", ""}:
            found_count += 1
            if re.search(r"\d+([.,]\d{1,2})?\s*(€|EUR|\$|USD|XAF|FCFA|Ar|Ariary|MGA|FMG)?", val_str, re.IGNORECASE):
                regex_bonus += 0.05
            if re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", val_str):
                regex_bonus += 0.05

    base_score = (found_count / total_keys) * 85.0
    final_score = min(round(base_score + (regex_bonus * 100), 1), 100.0)
    return final_score


def extract_invoice_fields_for_document(
    source: str,
    fields: list[str] | None = None,
    mode: str = "custom",
    max_context_chars: int = 12000,
) -> dict[str, Any]:
    """Extrait les champs d'une facture précise selon le mode choisi (custom ou auto-découverte)."""
    chunks = get_chunks_by_source(source)
    selected_chunks = limit_chunks_by_context_size(chunks, max_chars=max_context_chars)

    base_row: dict[str, Any] = {
        "document": source,
        "pages_utilisees": "",
        "nombre_chunks_utilises": len(selected_chunks),
    }

    if not selected_chunks:
        if fields:
            return {**base_row, **{f: "Non trouvé" for f in fields}, "indice_fiabilite_%": 0.0}
        return {**base_row, "commentaire": "Aucun texte indexé", "indice_fiabilite_%": 0.0}

    context = format_context(selected_chunks)

    if mode == "universal" or not fields:
        prompt = build_universal_extraction_prompt(context)
    else:
        prompt = build_custom_extraction_prompt(fields, context)

    raw_answer = generate_answer(prompt=prompt, context=context)
    parsed = extract_json_object(raw_answer)
    confidence = calculate_confidence_score(parsed, raw_answer)

    normalized: dict[str, Any] = {}
    if fields and mode != "universal":
        for field in fields:
            val = parsed.get(field, "Non trouvé") if isinstance(parsed, dict) else "Non trouvé"
            normalized[field] = str(val).strip() if str(val).strip() else "Non trouvé"
    else:
        if isinstance(parsed, dict) and parsed:
            for k, v in parsed.items():
                normalized[str(k).strip()] = str(v).strip()
        else:
            normalized["réponse_brute"] = raw_answer[:300]

    pages = sorted({
        chunk.get("metadata", {}).get("page")
        for chunk in selected_chunks
        if chunk.get("metadata", {}).get("page") is not None
    })

    return {
        **base_row,
        **normalized,
        "indice_fiabilite_%": confidence,
        "pages_utilisees": "; ".join(str(page) for page in pages),
    }


def extract_invoice_fields_for_all_documents(
    selected_sources: list[str] | None = None,
    fields: list[str] | None = None,
    mode: str = "custom",
    max_context_chars: int = 12000,
) -> pd.DataFrame:
    """Extrait les champs de facture pour tous les documents sélectionnés et harmonise les colonnes."""
    documents = list_indexed_documents()
    if selected_sources:
        selected = set(selected_sources)
        documents = [doc for doc in documents if doc["filename"] in selected]

    rows = []
    target_fields = fields if (fields and mode != "universal") else DEFAULT_INVOICE_FIELDS

    for document in documents:
        rows.append(
            extract_invoice_fields_for_document(
                source=document["filename"],
                fields=target_fields,
                mode=mode,
                max_context_chars=max_context_chars,
            )
        )

    df = pd.DataFrame(rows)
    first_cols = ["document", "indice_fiabilite_%"]
    existing_first = [c for c in first_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_first]
    return df[existing_first + other_cols]


def export_invoice_fields_to_excel(
    selected_sources: list[str] | None = None,
    fields: list[str] | None = None,
    mode: str = "custom",
    output_path: str | Path | None = None,
    max_context_chars: int = 12000,
) -> Path:
    """Exporte les champs de factures (standards, custom ou auto-découverts) dans un fichier Excel."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("data") / "exports" / f"extraction_factures_{timestamp}.xlsx"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    results_df = extract_invoice_fields_for_all_documents(
        selected_sources=selected_sources,
        fields=fields,
        mode=mode,
        max_context_chars=max_context_chars,
    )

    metadata_df = pd.DataFrame(
        [
            {"champ": "date_export", "valeur": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"champ": "nombre_documents", "valeur": len(results_df)},
            {"champ": "mode_extraction", "valeur": mode},
            {"champ": "max_context_chars", "valeur": max_context_chars},
            {"champ": "description", "valeur": "Extraction structurée bilingue Français/Malagasy."},
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