"""Script CLI pour exporter les données extraites vers Excel.

Exemples :
    python scripts/export_to_excel.py
    python scripts/export_to_excel.py --output data/exports/export.xlsx
    python scripts/export_to_excel.py --include-embeddings
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.excel_exporter import export_extracted_data_to_excel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporter les données extraites vers un fichier Excel.")
    parser.add_argument("--output", default=None, help="Chemin du fichier Excel de sortie.")
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help="Inclure les embeddings complets dans Excel. Non recommandé sauf petit corpus.",
    )
    args = parser.parse_args()

    output_path = export_extracted_data_to_excel(
        output_path=args.output,
        include_embeddings=args.include_embeddings,
    )
    print(f"Export Excel créé : {output_path}")


if __name__ == "__main__":
    main()
