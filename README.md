# Chatbot RAG juridique, administratif et extraction de factures

Prototype de niveau Master permettant d’interroger des documents PDF juridiques, administratifs ou des factures scannées avec une approche **RAG** (*Retrieval-Augmented Generation*).

Le système extrait le texte des documents, applique l’OCR si nécessaire, découpe les contenus en passages, génère des embeddings, stocke les données dans **PostgreSQL**, recherche les passages pertinents, puis génère des réponses avec un modèle local via **Ollama** ou avec un mode extractif sans LLM.

---

## 1. Fonctionnalités principales

- Importation de documents PDF.
- Extraction du texte page par page avec `pypdf`.
- OCR automatique pour les PDF scannés avec Tesseract.
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
- Export des réponses vers Excel.
- Mode spécialisé d’extraction de factures vers Excel.
- Interface web avec Streamlit.

---

## 2. Architecture générale

```text
PDF importés
   ↓
Extraction du texte avec pypdf
   ↓
OCR Tesseract si PDF scanné
   ↓
Nettoyage du texte
   ↓
Découpage en chunks
   ↓
Embeddings SentenceTransformers
   ↓
Stockage PostgreSQL
   ↓
Question utilisateur ou extraction structurée
   ↓
Recherche/contexte documentaire
   ↓
Qwen/Ollama ou mode extractif
   ↓
Réponse avec sources ou export Excel
```

---

## 3. Technologies utilisées

| Besoin | Technologie |
|---|---|
| Langage | Python |
| Interface | Streamlit |
| Extraction PDF | pypdf |
| OCR PDF scannés | Tesseract, pytesseract, PyMuPDF, Pillow |
| Embeddings | SentenceTransformers |
| Base de données | PostgreSQL |
| Génération locale | Ollama + Qwen 2.5 3B |
| Export Excel | Pandas, openpyxl |
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
├── configuration_ocr_windows.md
├── configuration_ollama_windows.md
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
│   ├── export_answers_to_excel.py
│   ├── export_invoices_to_excel.py
│   ├── test_chunking.py
│   ├── test_ocr.py
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
│   ├── excel_exporter.py
│   ├── answer_exporter.py
│   └── invoice_extractor.py
│
├── reports/
│   └── prompt_juridique_administratif.md
│
└── tests/
```

---

## 5. Installation sur Windows

### 5.1 Créer et activer l’environnement virtuel

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

Exemple de configuration locale avec PostgreSQL, OCR et Ollama :

```env
LLM_PROVIDER=ollama

OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=qwen2.5:3b

DATABASE_URL=postgresql://postgres:ton_mot_de_passe@localhost:5432/rag_juridique
EMBEDDING_DIMENSION=384

TOP_K=3
CHUNK_SIZE=700
CHUNK_OVERLAP=100

ENABLE_OCR=true
OCR_LANGUAGE=fra
OCR_DPI=150
OCR_MIN_TEXT_LENGTH=80
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

> Important : le fichier `.env` est ignoré par Git afin de ne pas exposer les mots de passe ou clés API.

---

## 8. OCR pour les documents scannés

Le projet prend en charge les PDF scannés grâce à **Tesseract OCR**.

Fonctionnement :

1. le système tente d’extraire le texte avec `pypdf` ;
2. si le texte extrait est trop court, il considère que la page est probablement scannée ;
3. la page est convertie en image avec `PyMuPDF` ;
4. `Tesseract` applique l’OCR ;
5. le texte OCR est ensuite nettoyé, découpé et indexé.

### 8.1 Installer Tesseract sur Windows

Télécharger Tesseract depuis :

```text
https://github.com/UB-Mannheim/tesseract/wiki
```

Pendant l’installation, installer la langue française :

```text
fra
```

Chemin fréquent :

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 8.2 Tester l’OCR

```powershell
python scripts/test_ocr.py
```

Un guide complet est disponible dans :

```text
configuration_ocr_windows.md
```

---

## 9. Modèle local avec Ollama

Le projet peut utiliser un modèle local via Ollama.

Modèle utilisé dans ce projet :

```text
qwen2.5:3b
```

### 9.1 Vérifier le modèle

```powershell
ollama list
```

Si le modèle n’est pas installé :

```powershell
ollama pull qwen2.5:3b
```

### 9.2 Tester Ollama directement

```powershell
ollama run qwen2.5:3b
```

### 9.3 Tester Ollama depuis Python

```powershell
python scripts/test_ollama.py
```

Un guide complet est disponible dans :

```text
configuration_ollama_windows.md
```

---

## 10. Lancement de l’application

```powershell
streamlit run app.py
```

Ensuite, dans l’interface :

1. importer un ou plusieurs PDF ;
2. cliquer sur **Indexer les documents** ;
3. poser une question ou lancer une extraction ;
4. consulter les réponses, les sources ou télécharger un fichier Excel.

---

## 11. Utilisation principale : question-réponse documentaire

Après indexation, l’utilisateur peut poser une question libre, par exemple :

```text
Quels sont les documents nécessaires pour cette procédure ?
```

Le chatbot récupère les passages pertinents, construit un contexte documentaire, puis génère une réponse avec sources.

---

## 12. Module d’évaluation

Un module d’évaluation est disponible pour mesurer la qualité du système.

### 12.1 Fichier de test

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

### 12.2 Lancer l’évaluation

```powershell
python scripts/run_evaluation.py --csv data/evaluation/questions_test_template.csv --top-k 4
```

### 12.3 Métriques calculées

- **Retrieval Hit Rate** : proportion de questions dont la bonne source est retrouvée.
- **Mean Reciprocal Rank** : rang moyen de la bonne source.
- **Answer Source Hit Rate** : présence de la bonne source dans les sources affichées.
- **Couverture des mots-clés** : proportion des mots-clés attendus présents dans la réponse.
- **Taux de refus hors contexte** : capacité du chatbot à refuser les questions hors corpus.
- **Score moyen du meilleur passage**.

---

## 13. Export des données extraites vers Excel

Cette fonction exporte les données brutes extraites et stockées dans PostgreSQL.

L’export contient plusieurs feuilles :

| Feuille | Contenu |
|---|---|
| `documents` | Liste des PDF importés |
| `chunks_extraits` | Passages extraits avec source, page, numéro de chunk et contenu |
| `statistiques` | Nombre de chunks, nombre de pages et taille du texte par document |
| `description` | Explication des champs exportés |

Depuis Streamlit :

```text
4. Exporter les données extraites vers Excel
```

En ligne de commande :

```powershell
python scripts/export_to_excel.py
```

Avec chemin personnalisé :

```powershell
python scripts/export_to_excel.py --output data/exports/export_donnees.xlsx
```

Inclure les embeddings complets, uniquement pour un petit corpus :

```powershell
python scripts/export_to_excel.py --include-embeddings
```

---

## 14. Export des réponses vers Excel

Cette fonction permet de poser une même question à chaque document indexé et d’exporter les réponses dans Excel.

Exemple avec des factures :

```text
Quel est le nom de la personne à qui la facture est adressée ?
```

Depuis Streamlit :

```text
5. Extraire des réponses et les exporter vers Excel
```

En ligne de commande :

```powershell
python scripts/export_answers_to_excel.py --question "Quel est le nom de la personne à qui la facture est adressée ?"
```

Le fichier généré contient une feuille `reponses` avec les colonnes :

| Colonne | Description |
|---|---|
| `document` | Nom du PDF traité |
| `question` | Question posée |
| `reponse` | Réponse extraite |
| `pages_utilisees` | Pages utilisées comme contexte |
| `nombre_chunks_utilises` | Nombre de chunks utilisés |

---

## 15. Mode factures : extraction structurée vers Excel

Le projet dispose d’un mode spécialisé pour extraire automatiquement plusieurs champs de factures.

Champs exportés :

| Champ | Description |
|---|---|
| `destinataire_nom` | Personne, entreprise ou organisation à qui la facture est adressée |
| `destinataire_adresse` | Adresse du destinataire |
| `emetteur_nom` | Personne ou entreprise qui a émis la facture |
| `numero_facture` | Numéro de facture |
| `date_facture` | Date de la facture |
| `montant_ht` | Montant hors taxes |
| `montant_tva` | Montant TVA |
| `montant_ttc` | Montant total TTC |
| `devise` | Devise utilisée |
| `commentaire` | Remarque éventuelle ou diagnostic |

Depuis Streamlit :

```text
6. Mode factures : extraction structurée vers Excel
```

En ligne de commande :

```powershell
python scripts/export_invoices_to_excel.py
```

Avec un chemin de sortie personnalisé :

```powershell
python scripts/export_invoices_to_excel.py --output data/exports/factures.xlsx
```

Le fichier Excel généré contient une feuille `factures` avec une ligne par document.

---

## 16. Prompt juridique et administratif

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

## 17. Gestion Git et fichiers ignorés

Le fichier `.gitignore` ignore notamment :

- l’environnement virtuel `.venv/` ;
- le fichier `.env` contenant les mots de passe ;
- les PDF importés dans `data/raw/` ;
- les fichiers traités dans `data/processed/` ;
- les exports Excel générés dans `data/exports/` ;
- les rapports d’évaluation générés automatiquement ;
- les caches Python ;
- les fichiers temporaires et fichiers systèmes.

Les dossiers `data/raw/`, `data/processed/` et `data/exports/` contiennent un fichier `.gitkeep` pour conserver la structure sans versionner les documents sensibles ou les exports générés.

---

## 18. Conseils de performance

Si l’application consomme beaucoup de RAM, utiliser une configuration légère :

```env
TOP_K=3
CHUNK_SIZE=700
CHUNK_OVERLAP=100
OCR_DPI=150
```

Pour le mode factures, réduire si nécessaire :

```text
Taille maximale du contexte par facture : 6000 à 8000
```

Éviter aussi d’indexer un grand nombre de PDF scannés en une seule fois.

---

## 19. Remarques importantes

Ce chatbot est un **assistant documentaire**. Il ne remplace pas :

- un avocat ;
- un juriste ;
- une administration compétente ;
- une consultation professionnelle.

Les réponses doivent toujours être vérifiées dans les documents sources ou auprès d’une autorité compétente.

Pour les factures, les résultats doivent aussi être vérifiés, surtout lorsque les documents sont scannés ou de mauvaise qualité.

---

## 20. Limites connues

- L’OCR peut produire des erreurs si les documents scannés sont flous, inclinés ou de mauvaise qualité.
- La recherche vectorielle est calculée côté Python, donc elle peut devenir lente sur un très grand corpus.
- La qualité des réponses dépend du modèle local utilisé.
- Le chatbot peut se tromper si le texte extrait du PDF est incomplet ou mal structuré.
- L’extraction de factures dépend fortement de la qualité OCR et de la mise en page des factures.

---

## 21. Perspectives

- Améliorer l’OCR avec prétraitement d’image : rotation, contraste, débruitage.
- Ajouter `pgvector` lorsque les ressources matérielles le permettront.
- Ajouter un reranker pour améliorer la recherche.
- Ajouter une authentification utilisateur.
- Ajouter un export PDF/TXT des réponses.
- Ajouter un score de confiance pour l’extraction de factures.
- Ajouter une colonne `texte_source` pour vérifier l’origine des informations extraites.
- Comparer plusieurs modèles d’embeddings.
- Comparer plusieurs modèles de génération.
- Déployer l’application sur un serveur local ou distant.
