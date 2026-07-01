"""Configuration centrale du projet."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"  # conservé pour compatibilité, mais PostgreSQL est utilisé

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "extractive").lower()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents_juridiques_administratifs")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/rag_juridique",
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# Seuil indicatif. Avec Chroma, la distance dépend de la métrique utilisée.
# Dans la V1, on l'utilise surtout comme information, pas comme blocage strict.
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.0"))

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTORSTORE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
