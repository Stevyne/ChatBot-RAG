"""Test rapide du découpage en chunks."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.text_splitter import split_text  # noqa: E402


def main() -> None:
    text = "A" * 2500
    chunks = split_text(text, chunk_size=1000, overlap=200)
    print(f"Nombre de chunks : {len(chunks)}")
    print([len(c) for c in chunks])
    assert len(chunks) == 3, "Le découpage devrait produire 3 chunks pour 2500 caractères."
    print("Test chunking réussi.")


if __name__ == "__main__":
    main()
