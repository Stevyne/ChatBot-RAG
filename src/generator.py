"""Génération de réponses à partir du contexte récupéré.

Ce module contient le prompt principal du chatbot. Il est volontairement strict
et bilingue (Français & Malagasy) pour limiter les hallucinations dans un contexte juridique et administratif.
"""
from __future__ import annotations

import json
import re
from typing import Any

import requests

from .config import LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_URL, OPENAI_API_KEY, OPENAI_MODEL

SYSTEM_INSTRUCTIONS = """Tu es un assistant documentaire spécialisé dans l'analyse de documents juridiques et administratifs en Français et en Malagasy (Malgache).

Ton rôle : aider l'utilisateur à comprendre les documents fournis, sans inventer d'information.

Règles obligatoires :
1. Réponds uniquement à partir du contexte documentaire fourni.
2. N'utilise pas tes connaissances générales si elles ne sont pas confirmées par le contexte.
3. Si le contexte ne contient pas l'information demandée, réponds exactement : "Je ne trouve pas cette information dans les documents fournis." (ou en malagasy : "Tsy hita ao amin'ny tahirin-kevitra io fanazavana io.").
4. Adaptabilité linguistique : Si la question est posée en malagasy, réponds en malagasy. Si la question est posée en français, réponds en français.
5. Comprends parfaitement le vocabulaire administratif, juridique et financier de Madagascar (ex: Ariary / MGA / Ar, Faktiora, Daty, Anarana, Mpanjifa, Tompony, Fokontany, Kaominina, Faritra, NIF, STAT, CIF, CSB...).
6. Ne donne jamais de conseil juridique personnalisé à la place d'un professionnel.
7. Structure la réponse avec des puces lorsque c'est utile.
8. Cite les sources sous la forme : document, page.
9. Ne mentionne pas des articles, lois ou délais absents du contexte.
"""


def build_prompt(context: str, question: str) -> str:
    """Construit le prompt final envoyé au modèle de langage."""
    return f"""{SYSTEM_INSTRUCTIONS}

Contexte documentaire disponible :
------------------------------
{context}
------------------------------

Question de l'utilisateur :
{question}

Format attendu de la réponse :
1. Réponse courte et directe dans la langue de la question (Français ou Malagasy).
2. Détails utiles sous forme de puces si nécessaire.
3. Sources utilisées, sous la forme :
   - Nom du document, page X

Si l'information n'est pas dans le contexte, réponds uniquement :
"Je ne trouve pas cette information dans les documents fournis."

Réponse :
"""


def generate_answer(prompt: str, context: str = "") -> str:
    """Génère une réponse selon le fournisseur configuré."""
    provider = LLM_PROVIDER.lower()
    if provider == "ollama":
        return post_process_answer(generate_with_ollama(prompt))
    if provider == "openai":
        return post_process_answer(generate_with_openai(prompt))
    return generate_extractive_answer(context)


def generate_with_ollama(prompt: str) -> str:
    """Appelle un modèle local via Ollama avec des paramètres légers pour économiser la RAM."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.85,
            "repeat_penalty": 1.1,
            "num_ctx": 2048,
            "num_predict": 350,
        },
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as exc:
        return (
            "Erreur lors de l'appel à Ollama. Vérifiez qu'Ollama est lancé et que le modèle est installé.\n\n"
            f"Détail : {exc}"
        )


def generate_with_openai(prompt: str) -> str:
    """Appelle l'API OpenAI avec requests, sans dépendance SDK."""
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY est manquante. Configurez la clé dans le fichier .env."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": OPENAI_MODEL,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"Erreur lors de l'appel à l'API OpenAI : {exc}"


def post_process_answer(answer: str) -> str:
    """Nettoie légèrement la réponse générée."""
    answer = answer.strip()
    if not answer:
        return "Je ne trouve pas cette information dans les documents fournis."

    unwanted_prefixes = [
        "Réponse :",
        "Voici la réponse :",
        "Valiny :",
        "D'après le contexte documentaire fourni,",
        "Selon le contexte documentaire fourni,",
    ]
    for prefix in unwanted_prefixes:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()

    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer


def generate_extractive_answer(context: str) -> str:
    """Générateur de secours sans LLM."""
    if not context.strip():
        return "Je ne trouve pas cette information dans les documents fournis."

    cleaned = re.sub(r"\[Source[^\]]+\]", "", context)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned.strip())
    selected = [s.strip() for s in sentences if len(s.strip()) > 40][:5]

    if not selected:
        selected = [cleaned[:900].strip()]

    return (
        "Mode extractif activé : aucun modèle génératif n'est configuré. "
        "Voici les passages les plus pertinents trouvés dans les documents (Français/Malagasy) :\n\n- "
        + "\n- ".join(selected)
    )