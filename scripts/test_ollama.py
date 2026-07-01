"""Test rapide de connexion à Ollama.

Utilisation :
    python scripts/test_ollama.py

Le script lit OLLAMA_URL et OLLAMA_MODEL depuis le fichier .env.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


def main() -> None:
    prompt = "Réponds en français en une phrase : quel est le rôle d'un chatbot RAG ?"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 120,
        },
    }

    print(f"Test Ollama")
    print(f"URL    : {OLLAMA_URL}")
    print(f"Modèle : {OLLAMA_MODEL}")
    print("\nEnvoi de la requête...")

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        print("\nRéponse du modèle :")
        print(data.get("response", "[Réponse vide]").strip())
        print("\nTest réussi.")
    except requests.exceptions.ConnectionError:
        print("\nErreur : impossible de se connecter à Ollama.")
        print("Vérifie qu'Ollama est installé et lancé avec : ollama serve")
    except requests.exceptions.HTTPError as exc:
        print(f"\nErreur HTTP : {exc}")
        print("Si le modèle est introuvable, installe-le avec :")
        print(f"ollama pull {OLLAMA_MODEL}")
    except Exception as exc:
        print(f"\nErreur inattendue : {exc}")


if __name__ == "__main__":
    main()
