"""Module d'évaluation du chatbot RAG.

Ce module évalue deux aspects principaux :
1. La recherche documentaire : le bon document ou la bonne page est-il retrouvé dans le top-k ?
2. La réponse générée : contient-elle les mots-clés attendus et les sources attendues ?

Le module fonctionne avec un fichier CSV de questions de test.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .rag_pipeline import answer_question
from .retriever import retrieve


@dataclass
class EvaluationSummary:
    """Résumé chiffré d'une évaluation."""

    total_questions: int
    retrieval_hit_rate: float
    mean_reciprocal_rank: float
    answer_source_hit_rate: float
    keyword_coverage_mean: float
    out_of_scope_success_rate: float | None
    average_top_score: float


def normalize_text(value: Any) -> str:
    """Normalise une valeur texte pour les comparaisons simples."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def parse_bool(value: Any) -> bool:
    """Convertit plusieurs représentations texte en booléen."""
    text = normalize_text(value)
    return text in {"true", "1", "yes", "oui", "vrai", "hors_contexte", "out_of_scope"}


def parse_expected_page(value: Any) -> int | None:
    """Parse une page attendue optionnelle."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def parse_keywords(value: Any) -> list[str]:
    """Parse une liste de mots-clés séparés par des points-virgules."""
    if value is None or pd.isna(value):
        return []
    return [keyword.strip().lower() for keyword in str(value).split(";") if keyword.strip()]


def load_test_set(csv_path: str | Path) -> pd.DataFrame:
    """Charge un jeu de questions depuis un fichier CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)
    required_columns = {"question"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes obligatoires manquantes : {missing}")

    optional_columns = {
        "expected_source": "",
        "expected_page": "",
        "expected_keywords": "",
        "expected_answer": "",
        "is_out_of_scope": False,
    }
    for column, default in optional_columns.items():
        if column not in df.columns:
            df[column] = default
    return df


def evaluate_retrieval_row(question: str, expected_source: str, expected_page: int | None, top_k: int) -> dict[str, Any]:
    """Évalue la recherche pour une seule question."""
    retrieved = retrieve(question, top_k=top_k)

    expected_source_norm = normalize_text(expected_source)
    hit = False
    reciprocal_rank = 0.0

    for rank, item in enumerate(retrieved, start=1):
        metadata = item.get("metadata", {})
        source = normalize_text(metadata.get("source", ""))
        page = metadata.get("page")

        source_ok = expected_source_norm == "" or expected_source_norm in source
        page_ok = expected_page is None or page == expected_page

        if source_ok and page_ok and expected_source_norm != "":
            hit = True
            reciprocal_rank = 1.0 / rank
            break

    top_score = float(retrieved[0].get("score", 0.0)) if retrieved else 0.0
    retrieved_sources = [
        f"{item.get('metadata', {}).get('source', '?')}#p{item.get('metadata', {}).get('page', '?')}"
        for item in retrieved
    ]

    return {
        "retrieval_hit": hit,
        "reciprocal_rank": reciprocal_rank,
        "top_score": top_score,
        "retrieved_sources": " | ".join(retrieved_sources),
    }


def evaluate_answer_keywords(answer: str, expected_keywords: list[str]) -> tuple[float, str]:
    """Mesure la présence des mots-clés attendus dans la réponse."""
    if not expected_keywords:
        return 0.0, ""

    answer_norm = normalize_text(answer)
    found = [keyword for keyword in expected_keywords if keyword in answer_norm]
    coverage = len(found) / len(expected_keywords)
    return coverage, "; ".join(found)


def evaluate_answer_sources(sources: list[dict[str, Any]], expected_source: str, expected_page: int | None) -> bool:
    """Vérifie si les sources de la réponse contiennent la source attendue."""
    expected_source_norm = normalize_text(expected_source)
    if not expected_source_norm:
        return False

    for source_item in sources:
        source = normalize_text(source_item.get("source", ""))
        page = source_item.get("page")
        source_ok = expected_source_norm in source
        page_ok = expected_page is None or page == expected_page
        if source_ok and page_ok:
            return True
    return False


def is_refusal_answer(answer: str) -> bool:
    """Détecte une réponse de refus/hors contexte."""
    answer_norm = normalize_text(answer)
    refusal_patterns = [
        "je ne trouve pas cette information",
        "aucun document",
        "aucun passage pertinent",
        "information n'est pas présente",
        "information absente",
    ]
    return any(pattern in answer_norm for pattern in refusal_patterns)


def run_evaluation(test_csv_path: str | Path, top_k: int = 4) -> pd.DataFrame:
    """Lance l'évaluation complète sur un CSV de questions."""
    df = load_test_set(test_csv_path)
    rows: list[dict[str, Any]] = []

    for _, item in df.iterrows():
        question = str(item["question"]).strip()
        expected_source = str(item.get("expected_source", "")).strip()
        expected_page = parse_expected_page(item.get("expected_page"))
        expected_keywords = parse_keywords(item.get("expected_keywords"))
        is_out_of_scope = parse_bool(item.get("is_out_of_scope"))

        retrieval_metrics = evaluate_retrieval_row(
            question=question,
            expected_source=expected_source,
            expected_page=expected_page,
            top_k=top_k,
        )

        rag_result = answer_question(question, top_k=top_k)
        answer = rag_result.get("answer", "")
        sources = rag_result.get("sources", [])

        keyword_coverage, found_keywords = evaluate_answer_keywords(answer, expected_keywords)
        answer_source_hit = evaluate_answer_sources(sources, expected_source, expected_page)
        refusal_ok = is_refusal_answer(answer) if is_out_of_scope else None

        rows.append(
            {
                "question": question,
                "expected_source": expected_source,
                "expected_page": expected_page,
                "is_out_of_scope": is_out_of_scope,
                "retrieval_hit": retrieval_metrics["retrieval_hit"],
                "reciprocal_rank": retrieval_metrics["reciprocal_rank"],
                "top_score": retrieval_metrics["top_score"],
                "retrieved_sources": retrieval_metrics["retrieved_sources"],
                "answer_source_hit": answer_source_hit,
                "keyword_coverage": keyword_coverage,
                "found_keywords": found_keywords,
                "refusal_ok": refusal_ok,
                "generated_answer": answer,
                "answer_sources": str(sources),
            }
        )

    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> EvaluationSummary:
    """Calcule un résumé global de l'évaluation."""
    total = len(results)
    if total == 0:
        return EvaluationSummary(0, 0.0, 0.0, 0.0, 0.0, None, 0.0)

    in_scope = results[~results["is_out_of_scope"].astype(bool)]
    out_scope = results[results["is_out_of_scope"].astype(bool)]

    retrieval_hit_rate = float(in_scope["retrieval_hit"].mean()) if len(in_scope) else 0.0
    mrr = float(in_scope["reciprocal_rank"].mean()) if len(in_scope) else 0.0
    answer_source_hit_rate = float(in_scope["answer_source_hit"].mean()) if len(in_scope) else 0.0
    keyword_coverage_mean = float(in_scope["keyword_coverage"].mean()) if len(in_scope) else 0.0
    average_top_score = float(results["top_score"].mean()) if len(results) else 0.0

    out_of_scope_success_rate = None
    if len(out_scope):
        out_of_scope_success_rate = float(out_scope["refusal_ok"].fillna(False).mean())

    return EvaluationSummary(
        total_questions=total,
        retrieval_hit_rate=retrieval_hit_rate,
        mean_reciprocal_rank=mrr,
        answer_source_hit_rate=answer_source_hit_rate,
        keyword_coverage_mean=keyword_coverage_mean,
        out_of_scope_success_rate=out_of_scope_success_rate,
        average_top_score=average_top_score,
    )


def save_evaluation_report(results: pd.DataFrame, output_dir: str | Path = "reports") -> dict[str, Path]:
    """Sauvegarde les résultats en CSV et un résumé Markdown."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"evaluation_results_{timestamp}.csv"
    md_path = output_path / f"evaluation_summary_{timestamp}.md"

    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = summarize_results(results)

    out_scope_text = (
        f"{summary.out_of_scope_success_rate:.2%}"
        if summary.out_of_scope_success_rate is not None
        else "Non évalué"
    )

    markdown = f"""# Rapport d'évaluation RAG

Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Résumé global

| Métrique | Valeur |
|---|---:|
| Nombre de questions | {summary.total_questions} |
| Retrieval Hit Rate | {summary.retrieval_hit_rate:.2%} |
| Mean Reciprocal Rank | {summary.mean_reciprocal_rank:.3f} |
| Answer Source Hit Rate | {summary.answer_source_hit_rate:.2%} |
| Couverture moyenne des mots-clés | {summary.keyword_coverage_mean:.2%} |
| Taux de refus correct hors contexte | {out_scope_text} |
| Score moyen du meilleur passage | {summary.average_top_score:.3f} |

## Explication des métriques

- **Retrieval Hit Rate** : proportion de questions pour lesquelles la source attendue est retrouvée dans les top-k passages.
- **Mean Reciprocal Rank** : mesure le rang moyen de la bonne source ; plus la valeur est proche de 1, mieux c'est.
- **Answer Source Hit Rate** : proportion de réponses dont les sources affichées contiennent la source attendue.
- **Couverture des mots-clés** : proportion des mots-clés attendus présents dans la réponse générée.
- **Taux de refus correct hors contexte** : capacité du système à refuser une question dont la réponse n'est pas dans les documents.

## Fichier détaillé

Les résultats détaillés sont disponibles dans :

```text
{csv_path.name}
```
"""

    md_path.write_text(markdown, encoding="utf-8")
    return {"csv": csv_path, "summary": md_path}
