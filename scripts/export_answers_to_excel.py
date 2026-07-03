"""Script CLI pour extraire des réponses par document et les exporter vers Excel.

Exemple :
    python scripts/export_answers_to_excel.py --question "Quel est le nom de la personne à qui la facture est adressée ?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.answer_exporter import export_answers_to_excel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporter des réponses extraites par document vers Excel.")
    parser.add_argument("--question", required=True, help="Question à poser à chaque document.")
    parser.add_argument("--output", default=None, help="Chemin du fichier Excel de sortie.")
    parser.add_argument("--max-context-chars", type=int, default=8000, help="Taille maximale du contexte par document.")
    args = parser.parse_args()

    output = export_answers_to_excel(
        question=args.question,
        output_path=args.output,
        max_context_chars=args.max_context_chars,
    )
    print(f"Export des réponses créé : {output}")


if __name__ == "__main__":
    main()
