# Chatbot RAG juridique et administratif

Prototype de niveau Master permettant d’interroger des documents PDF juridiques et administratifs avec une approche **RAG** (*Retrieval-Augmented Generation*).

Le système extrait le texte des documents, découpe les contenus en passages, génère des embeddings, stocke les données dans **PostgreSQL**, recherche les passages pertinents, puis génère une réponse avec un modèle local via **Ollama** ou avec un mode extractif sans LLM.

---

## 1. Fonctionnalités

- Importation de documents PDF.
- Extraction du texte page par page.
- Nettoyage simple du texte extrait.
- Découpage en chunks avec chevauchement.
- Génération d’embeddings avec `SentenceTransformers`.
- Stockage des documents, chunks, métadonnées et embeddings dans **PostgreSQL sans pgvector**.
- Recherche sémantique par similarité cosinus calculée côté Python.
- Génération de réponses avec :
  - `extractive` : mode léger sans API ni modèle génératif ;
  - `ollama` : modèle local, recommandé avec `qwen2.5:3b` ;
  - `openai` : option API si disponible.
- Affichage des sources : document et page.
- Module d’évaluation avec métriques.
- Export des données extraites vers Excel.
- Interface web avec Streamlit.

---

## 2. Architecture générale

```text
PDF importés
   ↓
Extraction du texte
   ↓
Découpage en chunks
   ↓
Embeddings SentenceTransformers
   ↓
Stockage PostgreSQL
   ↓
Question utilisateur
   ↓
Recherche sémantique
   ↓
Contexte documentaire
   ↓
Qwen/Ollama ou mode extractif
   ↓
Réponse avec sources
```

---

## 3. Technologies utilisées

| Besoin | Technologie |
|---|---|
| Langage | Python |
| Interface | Streamlit |
| Extraction PDF | pypdf |
| Embeddings | SentenceTransformers |
| Base de données | PostgreSQL |
| Génération locale | Ollama + Qwen 2.5 3B |
| Évaluation | Pandas, Scikit-learn |
| Connexion PostgreSQL | psycopg |

---

## 4. Structure du projet

```text
chatbot-rag-juridique/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   ├── evaluation/
│   │   └── questions_test_template.csv
│   └── exports/
│       └── .gitkeep
│
├── db/
│   └── schema.sql
│
├── scripts/
│   ├── run_evaluation.py
│   ├── export_to_excel.py
│   └── test_ollama.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   ├── evaluation.py
│   └── excel_exporter.py
│
├── reports/
│   └── prompt_juridique_administratif.md
│
└── tests/
```

---

## 5. Installation sur Windows

### 5.1 Créer et activer l’environnement virtuel

Dans PowerShell :

```powershell
cd chemin\vers\chatbot-rag-juridique
python -m venv .venv
.venv\Scripts\activate
```

Si PowerShell bloque l’activation :

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\activate
```

### 5.2 Installer les dépendances

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Configuration PostgreSQL

Le projet utilise **PostgreSQL sans pgvector**.

Les embeddings sont stockés dans une colonne :

```sql
DOUBLE PRECISION[]
```

La similarité cosinus est ensuite calculée dans Python. Cette solution est adaptée à un prototype local et évite l’installation de l’extension `pgvector`.

### 6.1 Créer la base de données

Dans pgAdmin ou `psql` :

```sql
CREATE DATABASE rag_juridique;
```

Le schéma SQL est disponible dans :

```text
db/schema.sql
```

L’application peut aussi créer automatiquement les tables au premier lancement.

---

## 7. Configuration du fichier `.env`

Copier le fichier d’exemple :

```powershell
copy .env.example .env
```

Puis ouvrir :

```powershell
notepad .env
```

Exemple de configuration locale avec PostgreSQL et Ollama :

```env
LLM_PROVIDER=ollama

OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=qwen2.5:3b

DATABASE_URL=postgresql://postgres:ton_mot_de_passe@localhost:5432/rag_juridique
EMBEDDING_DIMENSION=384

TOP_K=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

> Important : le fichier `.env` est ignoré par Git afin de ne pas exposer les mots de passe ou clés API.

---

## 8. Modèle local avec Ollama

Le projet peut utiliser un modèle local via Ollama.

Dans ce projet, le modèle utilisé est :

```text
qwen2.5:3b
```

### 8.1 Vérifier le modèle

```powershell
ollama list
```

Si le modèle n’est pas installé :

```powershell
ollama pull qwen2.5:3b
```

### 8.2 Tester Ollama directement

```powershell
ollama run qwen2.5:3b
```

Puis poser une question de test.

### 8.3 Tester Ollama depuis Python

```powershell
python scripts/test_ollama.py
```

Un guide complet est disponible dans :

```text
configuration_ollama_windows.md
```

---

## 9. Lancement de l’application

```powershell
streamlit run app.py
```

Ensuite, dans l’interface :

1. importer un ou plusieurs PDF ;
2. cliquer sur **Indexer les documents** ;
3. poser une question ;
4. consulter la réponse ;
5. vérifier les sources affichées.

---

## 10. Module d’évaluation

Un module d’évaluation est disponible pour mesurer la qualité du système.

### 10.1 Fichier de test

Un modèle de fichier CSV est fourni :

```text
data/evaluation/questions_test_template.csv
```

Colonnes disponibles :

| Colonne | Description |
|---|---|
| `question` | Question posée au chatbot |
| `expected_source` | Nom du PDF attendu |
| `expected_page` | Page attendue, optionnelle |
| `expected_keywords` | Mots-clés attendus dans la réponse, séparés par `;` |
| `expected_answer` | Réponse attendue pour analyse manuelle |
| `is_out_of_scope` | `true` si la question est hors corpus |

### 10.2 Lancer l’évaluation en ligne de commande

```powershell
python scripts/run_evaluation.py --csv data/evaluation/questions_test_template.csv --top-k 4
```

Les rapports sont générés dans :

```text
reports/
```

### 10.3 Métriques calculées

- **Retrieval Hit Rate** : proportion de questions dont la bonne source est retrouvée.
- **Mean Reciprocal Rank** : rang moyen de la bonne source.
- **Answer Source Hit Rate** : présence de la bonne source dans les sources affichées.
- **Couverture des mots-clés** : proportion des mots-clés attendus présents dans la réponse.
- **Taux de refus hors contexte** : capacité du chatbot à refuser les questions hors corpus.
- **Score moyen du meilleur passage**.

---

## 11. Prompt juridique et administratif

Le prompt du modèle a été renforcé pour limiter les hallucinations.

Le chatbot doit :

- répondre uniquement à partir du contexte documentaire ;
- refuser si l’information est absente ;
- éviter les conseils juridiques personnalisés ;
- citer les sources documentaires ;
- structurer les réponses en français clair.

Le prompt est documenté dans :

```text
reports/prompt_juridique_administratif.md
```


---

## 12. Export des données extraites vers Excel

Le projet permet d’exporter les données extraites et stockées dans PostgreSQL vers un fichier Excel `.xlsx`.

L’export contient plusieurs feuilles :

| Feuille | Contenu |
|---|---|
| `documents` | Liste des PDF importés |
| `chunks_extraits` | Passages extraits avec source, page, numéro de chunk et contenu |
| `statistiques` | Nombre de chunks, nombre de pages et taille du texte par document |
| `description` | Explication des champs exportés |

Par défaut, les embeddings complets ne sont pas exportés pour garder le fichier lisible. Seule la dimension de l’embedding est indiquée.

### 12.1 Export depuis l’interface Streamlit

Dans l’application, utiliser la section :

```text
4. Exporter les données extraites vers Excel
```

Puis cliquer sur :

```text
Exporter vers Excel
```

Un bouton de téléchargement apparaît ensuite.

### 12.2 Export en ligne de commande

```powershell
python scripts/export_to_excel.py
```

Spécifier un chemin de sortie :

```powershell
python scripts/export_to_excel.py --output data/exports/export_donnees.xlsx
```

Inclure les embeddings complets, uniquement pour un petit corpus :

```powershell
python scripts/export_to_excel.py --include-embeddings
```

Les fichiers générés sont placés par défaut dans :

```text
data/exports/
```

---

## 13. Gestion Git et fichiers ignorés

Le fichier `.gitignore` ignore notamment :

- l’environnement virtuel `.venv/` ;
- le fichier `.env` contenant les mots de passe ;
- les PDF importés dans `data/raw/` ;
- les fichiers traités dans `data/processed/` ;
- les exports Excel générés dans `data/exports/` ;
- les rapports d’évaluation générés automatiquement ;
- les caches Python ;
- les fichiers temporaires et fichiers systèmes.

Les dossiers `data/raw/` et `data/processed/` contiennent un fichier `.gitkeep` pour conserver la structure sans versionner les documents sensibles.

---

## 14. Remarques importantes

Ce chatbot est un **assistant documentaire**. Il ne remplace pas :

- un avocat ;
- un juriste ;
- une administration compétente ;
- une consultation professionnelle.

Les réponses doivent toujours être vérifiées dans les documents sources ou auprès d’une autorité compétente.

---

## 15. Limites connues

- Les PDF scannés sous forme d’image ne sont pas encore traités par OCR.
- La recherche vectorielle est calculée côté Python, donc elle peut devenir lente sur un très grand corpus.
- La qualité des réponses dépend du modèle local utilisé.
- Le chatbot peut se tromper si le texte extrait du PDF est incomplet ou mal structuré.

---

## 16. Perspectives

- Ajouter OCR pour les PDF scannés.
- Ajouter `pgvector` lorsque les ressources matérielles le permettront.
- Ajouter un reranker pour améliorer la recherche.
- Ajouter une authentification utilisateur.
- Ajouter un export PDF/TXT des réponses.
- Comparer plusieurs modèles d’embeddings.
- Comparer plusieurs modèles de génération.
- Déployer l’application sur un serveur local ou distant.
