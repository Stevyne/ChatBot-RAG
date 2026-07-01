"""Chargement et extraction de texte depuis des documents PDF.

Le module combine deux méthodes :
1. Extraction directe du texte avec pypdf pour les PDF textuels.
2. OCR avec Tesseract pour les PDF scannés ou contenant très peu de texte extractible.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .config import ENABLE_OCR, OCR_DPI, OCR_LANGUAGE, OCR_MIN_TEXT_LENGTH, TESSERACT_CMD


def clean_text(text: str) -> str:
    """Nettoie légèrement le texte extrait d'un PDF ou obtenu par OCR."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def _configure_tesseract() -> None:
    """Configure le chemin de Tesseract si défini dans .env."""
    if not TESSERACT_CMD:
        return
    try:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    except ImportError:
        # L'erreur détaillée sera levée au moment de l'OCR.
        pass


def ocr_pdf_page(file_path: str | Path, page_index_zero_based: int) -> str:
    """Effectue l'OCR d'une page PDF avec PyMuPDF + Tesseract.

    Args:
        file_path: chemin du PDF.
        page_index_zero_based: index de page PyMuPDF, commençant à 0.
    """
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Les dépendances OCR ne sont pas installées. Lancez : pip install -r requirements.txt"
        ) from exc

    _configure_tesseract()

    pdf_document = fitz.open(str(file_path))
    try:
        page = pdf_document.load_page(page_index_zero_based)
        zoom = OCR_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        text = pytesseract.image_to_string(image, lang=OCR_LANGUAGE)
        return clean_text(text)
    finally:
        pdf_document.close()


def extract_text_with_pypdf(page: Any) -> str:
    """Extrait le texte d'une page avec pypdf."""
    try:
        return clean_text(page.extract_text() or "")
    except Exception:
        return ""


def should_use_ocr(extracted_text: str) -> bool:
    """Détermine si l'OCR doit être utilisé pour une page."""
    return ENABLE_OCR and len(extracted_text.strip()) < OCR_MIN_TEXT_LENGTH


def load_pdf(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Extrait le texte d'un PDF page par page.

    Pour chaque page, le système tente d'abord l'extraction directe avec pypdf.
    Si le texte extrait est trop court, il utilise l'OCR si `ENABLE_OCR=true`.

    Retourne une liste de dictionnaires :
    {
        "text": "...",
        "metadata": {
            "source": "document.pdf",
            "page": 1,
            "extraction_method": "pypdf" ou "ocr"
        }
    }
    """
    path = Path(file_path)
    reader = PdfReader(str(path))
    pages: list[dict[str, Any]] = []

    for index, page in enumerate(reader.pages, start=1):
        text = extract_text_with_pypdf(page)
        extraction_method = "pypdf"
        ocr_error = ""

        if should_use_ocr(text):
            try:
                ocr_text = ocr_pdf_page(path, index - 1)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    extraction_method = "ocr"
            except Exception as exc:
                ocr_error = str(exc)
                # On garde le texte pypdf s'il existe, même faible.

        text = clean_text(text)
        if text:
            metadata = {
                "source": path.name,
                "page": index,
                "file_path": str(path),
                "extraction_method": extraction_method,
            }
            if ocr_error:
                metadata["ocr_error"] = ocr_error[:500]
            pages.append({"text": text, "metadata": metadata})
        else:
            # On conserve une trace minimale des pages vides uniquement si l'OCR a échoué.
            if ocr_error:
                pages.append(
                    {
                        "text": f"[Page {index} : aucun texte extrait. Erreur OCR : {ocr_error[:300]}]",
                        "metadata": {
                            "source": path.name,
                            "page": index,
                            "file_path": str(path),
                            "extraction_method": "failed_ocr",
                            "ocr_error": ocr_error[:500],
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
