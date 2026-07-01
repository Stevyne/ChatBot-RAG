"""Script CLI pour lancer l'évaluation du chatbot RAG.

Exemple :
    python scripts/run_evaluation.py --csv data/evaluation/questions_test.csv --top-k 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import run_evaluation, save_evaluation_report, summarize_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluer le chatbot RAG sur un fichier CSV de questions.")
    parser.add_argument("--csv", required=True, help="Chemin du fichier CSV de questions de test.")
    parser.add_argument("--top-k", type=int, default=4, help="Nombre de passages récupérés par question.")
    parser.add_argument("--output-dir", default="reports", help="Dossier de sortie des rapports.")
    args = parser.parse_args()

    results = run_evaluation(args.csv, top_k=args.top_k)
    paths = save_evaluation_report(results, output_dir=args.output_dir)
    summary = summarize_results(results)

    print("\nÉvaluation terminée.")
    print(f"Nombre de questions : {summary.total_questions}")
    print(f"Retrieval Hit Rate : {summary.retrieval_hit_rate:.2%}")
    print(f"MRR : {summary.mean_reciprocal_rank:.3f}")
    print(f"Answer Source Hit Rate : {summary.answer_source_hit_rate:.2%}")
    print(f"Couverture mots-clés : {summary.keyword_coverage_mean:.2%}")
    if summary.out_of_scope_success_rate is not None:
        print(f"Refus correct hors contexte : {summary.out_of_scope_success_rate:.2%}")
    print(f"\nRésultats CSV : {paths['csv']}")
    print(f"Résumé Markdown : {paths['summary']}")


if __name__ == "__main__":
    main()
