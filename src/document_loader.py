"""Chargement et extraction de texte depuis des documents PDF."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Nettoie légèrement le texte extrait d'un PDF."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def load_pdf(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Extrait le texte d'un PDF page par page.

    Retourne une liste de dictionnaires :
    {
        "text": "...",
        "metadata": {"source": "document.pdf", "page": 1}
    }
    """
    path = Path(file_path)
    reader = PdfReader(str(path))
    pages: list[dict[str, Any]] = []

    for index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = clean_text(raw_text)
        if text:
            pages.append(
                {
                    "text": text,
                    "metadata": {
                        "source": path.name,
                        "page": index,
                        "file_path": str(path),
                    },
                }
            )

    return pages


def load_pdfs(file_paths: list[str | Path]) -> list[dict[str, Any]]:
    """Charge plusieurs PDF et concatène les pages extraites."""
    all_pages: list[dict[str, Any]] = []
    for file_path in file_paths:
        all_pages.extend(load_pdf(file_path))
    return all_pages
