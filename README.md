# Chatbot RAG juridique et administratif

Prototype de niveau Master permettant d'interroger des documents PDF juridiques et administratifs avec une approche **RAG** : Retrieval-Augmented Generation.

## Fonctionnalités V1

- Import de PDF
- Extraction du texte page par page
- Découpage en chunks avec chevauchement
- Embeddings multilingues avec SentenceTransformers
- Stockage des documents, métadonnées et embeddings dans PostgreSQL sans pgvector
- Recherche sémantique
- Réponse avec sources
- Mode sans API : `extractive`
- Modes optionnels : `ollama` ou `openai`

## Installation

```bash
cd chatbot-rag-juridique
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```


## Base de données PostgreSQL

Le projet utilise maintenant **PostgreSQL sans pgvector** pour stocker :

- les documents importés ;
- les chunks extraits ;
- les métadonnées : source, page, numéro de chunk ;
- les embeddings vectoriels ;
- les données nécessaires à la recherche sémantique.

Dans cette version locale, les embeddings sont stockés dans une colonne `DOUBLE PRECISION[]`, puis la similarité cosinus est calculée en Python. Cette approche est suffisante pour un prototype et un corpus petit ou moyen.

Créer une base :

```sql
CREATE DATABASE rag_juridique;
```

Configurer `.env` :

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_juridique
EMBEDDING_DIMENSION=384
```

Le schéma SQL est disponible dans :

```text
db/schema.sql
```

## Configuration

Copier le fichier d'exemple :

```bash
cp .env.example .env
```

Par défaut :

```env
LLM_PROVIDER=extractive
```

Ce mode ne nécessite ni clé API ni modèle local. Il retourne les passages les plus pertinents au lieu d'une réponse entièrement générée.

### Option Ollama local

Installer Ollama, puis :

```bash
ollama pull mistral
ollama serve
```

Dans `.env` :

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=mistral
```

### Option OpenAI

Dans `.env` :

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=votre_cle_api
OPENAI_MODEL=gpt-4o-mini
```

## Lancement

```bash
streamlit run app.py
```

## Structure

```text
chatbot-rag-juridique/
├── app.py
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/
│   └── processed/
├── db/
│   └── schema.sql
├── src/
│   ├── config.py
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   └── evaluation.py
└── tests/
```

## Utilisation

1. Lancer l'application Streamlit.
2. Importer un ou plusieurs PDF.
3. Cliquer sur **Indexer les documents**.
4. Poser une question.
5. Consulter la réponse et les sources.

## Remarques importantes

Ce chatbot est un assistant documentaire. Il ne remplace pas un avocat, un juriste ou une administration compétente. Les réponses doivent toujours être vérifiées dans les sources officielles.


## Module d’évaluation

Un module d’évaluation est disponible pour mesurer la qualité du système.

### Fichier de test

Un modèle de fichier CSV est fourni :

```text
data/evaluation/questions_test_template.csv
```

Colonnes disponibles :

- `question` : question posée au chatbot ;
- `expected_source` : nom du PDF attendu ;
- `expected_page` : page attendue, optionnelle ;
- `expected_keywords` : mots-clés attendus dans la réponse, séparés par `;` ;
- `expected_answer` : réponse attendue, utilisée pour l'analyse manuelle ;
- `is_out_of_scope` : `true` si la question est hors corpus.

### Lancer l’évaluation en ligne de commande

```bash
python scripts/run_evaluation.py --csv data/evaluation/questions_test_template.csv --top-k 4
```

Les rapports sont générés dans :

```text
reports/
```

### Métriques calculées

- Retrieval Hit Rate ;
- Mean Reciprocal Rank ;
- Answer Source Hit Rate ;
- couverture des mots-clés ;
- taux de refus correct pour les questions hors contexte ;
- score moyen du meilleur passage.


## Modèle local avec Ollama

Le projet peut utiliser un modèle local via Ollama.

Pour une machine avec peu de mémoire, commencer par :

```bash
ollama pull qwen2.5:1.5b
```

Dans `.env` :

```env
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=qwen2.5:1.5b
```

Tester Ollama depuis Python :

```bash
python scripts/test_ollama.py
```

Un guide complet est disponible dans :

```text
configuration_ollama_windows.md
```
