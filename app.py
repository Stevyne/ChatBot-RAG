from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

from src.config import LLM_PROVIDER, RAW_DATA_DIR, TOP_K
from src.rag_pipeline import answer_question, index_documents
from src.vector_store import count_documents, reset_vectorstore
from src.evaluation import run_evaluation, save_evaluation_report, summarize_results
from src.excel_exporter import export_extracted_data_to_excel

st.set_page_config(
    page_title="Chatbot RAG juridique et administratif",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Chatbot RAG pour documents juridiques et administratifs")
st.caption("Prototype Master — réponses basées sur vos PDF avec affichage des sources.")

with st.sidebar:
    st.header("Configuration")
    st.write(f"**Mode LLM :** `{LLM_PROVIDER}`")
    st.write(f"**Chunks indexés :** `{count_documents()}`")
    st.info(
        "Par défaut, le projet fonctionne en mode `extractive`, sans API. "
        "Pour de vraies réponses génératives, configurez `.env` avec `LLM_PROVIDER=ollama` ou `openai`."
    )

    if st.button("🗑️ Réinitialiser la base vectorielle"):
        reset_vectorstore()
        st.success("Base vectorielle réinitialisée.")
        st.rerun()

st.subheader("1. Importer et indexer des documents PDF")

uploaded_files = st.file_uploader(
    "Ajoutez un ou plusieurs fichiers PDF",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        target_path = RAW_DATA_DIR / uploaded_file.name
        with target_path.open("wb") as f:
            shutil.copyfileobj(uploaded_file, f)
        saved_paths.append(target_path)

    st.success(f"{len(saved_paths)} fichier(s) sauvegardé(s).")
    st.write("Fichiers :")
    for path in saved_paths:
        st.write(f"- {path.name}")

    if st.button("📚 Indexer les documents"):
        with st.spinner("Extraction, découpage et indexation en cours..."):
            try:
                stats = index_documents(saved_paths)
                st.success("Indexation terminée.")
                st.json(stats)
            except Exception as exc:
                st.error(f"Erreur pendant l'indexation : {exc}")

st.divider()

st.subheader("2. Poser une question")

question = st.text_area(
    "Votre question",
    placeholder="Exemple : Quels sont les documents nécessaires pour renouveler une carte d'identité ?",
    height=100,
)

col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.number_input("Nombre de passages à récupérer", min_value=1, max_value=10, value=TOP_K)

if st.button("💬 Obtenir une réponse", type="primary"):
    if count_documents() == 0:
        st.warning("Veuillez d'abord importer et indexer au moins un document PDF.")
    elif not question.strip():
        st.warning("Veuillez saisir une question.")
    else:
        with st.spinner("Recherche des passages pertinents et génération de la réponse..."):
            result = answer_question(question, top_k=int(top_k))

        st.markdown("### Réponse")
        st.write(result["answer"])

        st.markdown("### Sources")
        if result["sources"]:
            for source in result["sources"]:
                st.write(
                    f"- **{source['source']}**, page **{source['page']}** "
                    f"— score indicatif : `{source['score']}`"
                )
        else:
            st.write("Aucune source disponible.")

        with st.expander("Voir les passages récupérés"):
            for i, ctx in enumerate(result["retrieved_contexts"], start=1):
                metadata = ctx.get("metadata", {})
                st.markdown(
                    f"**Passage {i} — {metadata.get('source', 'document inconnu')}, "
                    f"page {metadata.get('page', '?')} — score {ctx.get('score', 0):.3f}**"
                )
                st.write(ctx.get("text", ""))


st.divider()
st.subheader("3. Évaluer le système")

with st.expander("Lancer une évaluation à partir d'un fichier CSV"):
    st.write(
        "Le fichier CSV doit contenir au minimum une colonne `question`. "
        "Colonnes recommandées : `expected_source`, `expected_page`, `expected_keywords`, `is_out_of_scope`."
    )
    evaluation_file = st.file_uploader(
        "Importer un fichier CSV de questions de test",
        type=["csv"],
        accept_multiple_files=False,
        key="evaluation_csv",
    )
    eval_top_k = st.number_input("Top-k pour l'évaluation", min_value=1, max_value=10, value=TOP_K, key="eval_top_k")

    if evaluation_file is not None and st.button("📊 Lancer l'évaluation"):
        temp_path = RAW_DATA_DIR / "evaluation_questions.csv"
        with temp_path.open("wb") as f:
            f.write(evaluation_file.getbuffer())

        with st.spinner("Évaluation en cours..."):
            try:
                results_df = run_evaluation(temp_path, top_k=int(eval_top_k))
                report_paths = save_evaluation_report(results_df, output_dir="reports")
                summary = summarize_results(results_df)

                st.success("Évaluation terminée.")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Questions", summary.total_questions)
                col_b.metric("Retrieval Hit Rate", f"{summary.retrieval_hit_rate:.1%}")
                col_c.metric("MRR", f"{summary.mean_reciprocal_rank:.3f}")

                col_d, col_e, col_f = st.columns(3)
                col_d.metric("Sources correctes", f"{summary.answer_source_hit_rate:.1%}")
                col_e.metric("Mots-clés", f"{summary.keyword_coverage_mean:.1%}")
                if summary.out_of_scope_success_rate is not None:
                    col_f.metric("Refus hors contexte", f"{summary.out_of_scope_success_rate:.1%}")
                else:
                    col_f.metric("Refus hors contexte", "N/A")

                st.dataframe(results_df)
                st.write(f"Rapport CSV : `{report_paths['csv']}`")
                st.write(f"Résumé Markdown : `{report_paths['summary']}`")
            except Exception as exc:
                st.error(f"Erreur pendant l'évaluation : {exc}")

st.divider()
st.subheader("4. Exporter les données extraites vers Excel")

with st.expander("Créer un fichier Excel à partir des données stockées dans PostgreSQL"):
    st.write(
        "Cette fonction exporte les documents importés, les chunks extraits et des statistiques "
        "dans un fichier `.xlsx`. Les embeddings complets ne sont pas inclus par défaut pour garder "
        "le fichier lisible."
    )
    include_embeddings_excel = st.checkbox(
        "Inclure les embeddings complets dans Excel, non recommandé sauf petit corpus",
        value=False,
    )

    if st.button("📄 Exporter vers Excel"):
        if count_documents() == 0:
            st.warning("Aucun chunk n'est actuellement indexé. Importez et indexez d'abord des PDF.")
        else:
            with st.spinner("Création du fichier Excel..."):
                try:
                    excel_path = export_extracted_data_to_excel(include_embeddings=include_embeddings_excel)
                    st.success(f"Fichier Excel créé : {excel_path}")
                    with open(excel_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Télécharger le fichier Excel",
                            data=f,
                            file_name=excel_path.name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                except Exception as exc:
                    st.error(f"Erreur pendant l'export Excel : {exc}")
