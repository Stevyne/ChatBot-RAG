"""Test rapide de l'installation OCR/Tesseract.

Utilisation :
    python scripts/test_ocr.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.config import OCR_LANGUAGE, TESSERACT_CMD  # noqa: E402
from src.document_loader import _configure_tesseract  # noqa: E402


def main() -> None:
    try:
        import pytesseract

        _configure_tesseract()
        version = pytesseract.get_tesseract_version()
        languages = pytesseract.get_languages(config="")
        print("OCR/Tesseract détecté correctement.")
        print(f"Version Tesseract : {version}")
        print(f"TESSERACT_CMD     : {TESSERACT_CMD or '[chemin système par défaut]'}")
        print(f"Langue configurée : {OCR_LANGUAGE}")
        print(f"Langues installées: {', '.join(languages)}")
        if OCR_LANGUAGE not in languages:
            print("\nAttention : la langue configurée n'est pas installée dans Tesseract.")
            print("Pour le français, il faut que la langue 'fra' apparaisse dans la liste.")
    except Exception as exc:
        print("Erreur OCR/Tesseract.")
        print(exc)
        print("\nVérifie que Tesseract est installé et que TESSERACT_CMD est correct dans .env.")


if __name__ == "__main__":
    main()
