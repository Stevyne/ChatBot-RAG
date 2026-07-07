"""Chargement et extraction de texte depuis des documents PDF et images scannées.

Le module combine trois capacités :
1. Extraction directe du texte avec pypdf pour les PDF textuels.
2. OCR avec Tesseract pour les PDF scannés ou contenant très peu de texte extractible.
3. OCR direct pour les fichiers images (factures ou documents scannés en PNG, JPG, TIFF).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .config import ENABLE_OCR, OCR_DPI, OCR_LANGUAGE, OCR_MIN_TEXT_LENGTH, TESSERACT_CMD

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


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


def _configure_tesseract() -> str | None:
    """Configure Tesseract et la variable d'environnement TESSDATA_PREFIX sous Windows."""
    try:
        import pytesseract
    except ImportError:
        return None

    tesseract_exe = None
    if TESSERACT_CMD and os.path.exists(TESSERACT_CMD):
        tesseract_exe = TESSERACT_CMD
    else:
        # Détection automatique sur Windows si TESSERACT_CMD n'est pas dans .env
        windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"D:\Program Files\Tesseract-OCR\tesseract.exe",
        ]
        for path in windows_paths:
            if os.path.exists(path):
                tesseract_exe = path
                break

    if tesseract_exe:
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe
        # Configuration automatique indispensable sous Windows de TESSDATA_PREFIX
        tessdata_dir = os.path.join(os.path.dirname(tesseract_exe), "tessdata")
        if os.path.exists(tessdata_dir):
            os.environ["TESSDATA_PREFIX"] = tessdata_dir
        return tesseract_exe

    return getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")


def _run_tesseract_safe(image: Any) -> str:
    """Lance l'OCR de manière sécurisée avec repli automatique (fallback) de langue."""
    import pytesseract

    _configure_tesseract()

    # Tentative 1 : Langue configurée dans le projet (ex: fra)
    try:
        if OCR_LANGUAGE:
            return pytesseract.image_to_string(image, lang=OCR_LANGUAGE)
    except Exception as exc:
        # Si l'erreur vient du fichier de langue manquant (fra.traineddata absent), on tente en secours
        if "Failed loading language" in str(exc) or "Error opening data file" in str(exc):
            pass
        else:
            raise exc

    # Tentative 2 : Repli sur l'anglais 'eng' (toujours installé par défaut sous Windows)
    try:
        return pytesseract.image_to_string(image, lang="eng")
    except Exception:
        pass

    # Tentative 3 : Repli ultime sans spécification de langue
    return pytesseract.image_to_string(image)


def check_tesseract_availability() -> dict[str, Any]:
    """Vérifie si Tesseract et les langues OCR sont disponibles sur le système."""
    tesseract_path = _configure_tesseract()
    try:
        import pytesseract
        version = str(pytesseract.get_tesseract_version())
        languages = pytesseract.get_languages()
        return {
            "available": True,
            "version": version,
            "languages": languages,
            "path": tesseract_path,
            "has_french": "fra" in languages,
        }
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "path": tesseract_path,
        }


def ocr_image(image_path: str | Path) -> str:
    """Effectue l'OCR direct d'un fichier image (facture scannée en JPG/PNG...)."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "La dépendance Pillow n'est pas installée. Lancez : pip install -r requirements.txt"
        ) from exc

    image = Image.open(str(image_path))
    text = _run_tesseract_safe(image)
    return clean_text(text)


def ocr_pdf_page(file_path: str | Path, page_index_zero_based: int) -> str:
    """Effectue l'OCR d'une page PDF avec PyMuPDF + Tesseract."""
    try:
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Les dépendances OCR (PyMuPDF, Pillow) ne sont pas installées. Lancez : pip install -r requirements.txt"
        ) from exc

    pdf_document = fitz.open(str(file_path))
    try:
        page = pdf_document.load_page(page_index_zero_based)
        zoom = OCR_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        text = _run_tesseract_safe(image)
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


def load_image_document(file_path: str | Path) -> list[dict[str, Any]]:
    """Charge et applique l'OCR sur un fichier image scanné (PNG, JPG, TIFF)."""
    path = Path(file_path)
    ocr_error = ""
    try:
        text = ocr_image(path)
        extraction_method = "ocr_image"
    except Exception as exc:
        text = ""
        ocr_error = str(exc)
        extraction_method = "failed_ocr_image"

    if text:
        return [
            {
                "text": text,
                "metadata": {
                    "source": path.name,
                    "page": 1,
                    "file_path": str(path),
                    "extraction_method": extraction_method,
                },
            }
        ]
    else:
        return [
            {
                "text": f"[Image scannée : aucun texte extrait. Erreur OCR : {ocr_error[:300]}]",
                "metadata": {
                    "source": path.name,
                    "page": 1,
                    "file_path": str(path),
                    "extraction_method": extraction_method,
                    "ocr_error": ocr_error[:500],
                },
            }
        ]


def load_pdf(file_path: str | Path) -> list[dict[str, Any]]:
    """Extrait le texte d'un PDF page par page avec secours OCR si le texte est absent ou trop court."""
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


def load_document(file_path: str | Path) -> list[dict[str, Any]]:
    """Charge un document selon son extension (PDF ou Image scannée)."""
    path = Path(file_path)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return load_image_document(path)
    return load_pdf(path)


def load_pdfs(file_paths: list[str | Path]) -> list[dict[str, Any]]:
    """Alias pour load_documents (conservé pour compatibilité)."""
    return load_documents(file_paths)


def load_documents(file_paths: list[str | Path]) -> list[dict[str, Any]]:
    """Charge plusieurs documents (PDF ou Images scannées) et concatène les pages."""
    all_pages: list[dict[str, Any]] = []
    for file_path in file_paths:
        all_pages.extend(load_document(file_path))
    return all_pages