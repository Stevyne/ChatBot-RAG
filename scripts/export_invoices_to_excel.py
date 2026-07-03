"""Script CLI pour extraire des champs de factures et les exporter vers Excel.

Exemples :
    python scripts/export_invoices_to_excel.py
    python scripts/export_invoices_to_excel.py --output data/exports/factures.xlsx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.invoice_extractor import export_invoice_fields_to_excel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraire des champs de factures vers Excel.")
    parser.add_argument("--output", default=None, help="Chemin du fichier Excel de sortie.")
    parser.add_argument("--max-context-chars", type=int, default=12000, help="Taille maximale du contexte par facture.")
    args = parser.parse_args()

    output = export_invoice_fields_to_excel(
        output_path=args.output,
        max_context_chars=args.max_context_chars,
    )
    print(f"Export factures créé : {output}")


if __name__ == "__main__":
    main()
