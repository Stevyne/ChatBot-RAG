from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

from src.config import LLM_PROVIDER, RAW_DATA_DIR, TOP_K
from src.document_loader import check_tesseract_availability
from src.rag_pipeline import answer_question, index_documents
from src.vector_store import count_documents, list_indexed_documents, reset_vectorstore
from src.evaluation import run_evaluation, save_evaluation_report, summarize_results
from src.excel_exporter import export_extracted_data_to_excel
from src.answer_exporter import export_answers_to_excel
from src.invoice_extractor import (
    DEFAULT_INVOICE_FIELDS,
    export_invoice_fields_to_excel,
    extract_invoice_fields_for_all_documents,
)

st.set_page_config(
    page_title="Chatbot RAG juridique et administratif",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Chatbot RAG pour documents juridiques et administratifs")
st.caption("Prototype Master — lecture OCR, RAG et extraction structurée adaptative pour tout type de facture vers Excel.")

with st.sidebar:
    st.header("⚙️ Configuration & Statut")
    st.write(f"**Mode LLM :** `{LLM_PROVIDER}`")
    st.write(f"**Chunks indexés :** `{count_documents()}`")
    st.success("**🇲🇬 Contexte linguistique :** `Français & Malagasy` (Reconnaissance OCR & IA active)")

    st.markdown("### 👁️ Moteur OCR (Documents scannés)")
    ocr_status = check_tesseract_availability()
    if ocr_status.get("available"):
        langs = ", ".join(ocr_status.get("languages", []))
        st.success(f"**Tesseract OCR : Détecté**\n\nLangues : `{langs}`")
        if not ocr_status.get("has_french"):
            st.warning("⚠️ Langue `fra` absente. Installez le pack français pour un meilleur résultat.")
    else:
        st.warning(
            "**Tesseract OCR : Non détecté**\n\n"
            "Pour lire les PDF scannés ou images JPG/PNG, installez Tesseract Windows (UB Mannheim) "
            "ou configurez `TESSERACT_CMD` dans votre fichier `.env`."
        )

    st.info(
        "Par défaut, le projet fonctionne en mode `extractive`, sans API. "
        "Pour de vraies réponses génératives, configurez `.env` avec `LLM_PROVIDER=ollama` (Qwen 2.5:3b)."
    )

    if st.button("🗑️ Réinitialiser la base vectorielle"):
        reset_vectorstore()
        st.success("Base vectorielle réinitialisée.")
        st.rerun()

st.subheader("1. Importer et indexer des PDF ou Images scannées")
st.caption("Prend en charge les fichiers PDF (textuels et scannés) ainsi que les images de factures (PNG, JPG, TIFF).")

uploaded_files = st.file_uploader(
    "Ajoutez un ou plusieurs fichiers PDF ou Images",
    type=["pdf", "png", "jpg", "jpeg", "tiff"],
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
    st.write("Fichiers prêts à être indexés :")
    for path in saved_paths:
        st.write(f"- `{path.name}`")

    if st.button("📚 Indexer les documents et faire l'OCR si nécessaire", type="primary"):
        with st.spinner("Extraction (pypdf + OCR Tesseract), découpage et indexation en cours..."):
            try:
                stats = index_documents(saved_paths)
                st.success("Indexation terminée avec succès !")
                st.json(stats)
            except Exception as exc:
                st.error(f"Erreur pendant l'indexation : {exc}")

st.divider()

st.subheader("2. Poser une question sur les documents (Français ou Malagasy)")

question = st.text_area(
    "Votre question / Fanontaniana",
    placeholder="Exemple : Iza no anaran'ny mpanjifa amin'ity faktiora ity ? Ou : Quels sont les documents nécessaires pour renouveler une carte d'identité ?",
    height=100,
)

col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.number_input("Nombre de passages à récupérer", min_value=1, max_value=10, value=TOP_K)

if st.button("💬 Obtenir une réponse", type="primary"):
    if count_documents() == 0:
        st.warning("Veuillez d'abord importer et indexer au moins un document ou image.")
    elif not question.strip():
        st.warning("Veuillez saisir une question.")
    else:
        with st.spinner("Recherche des passages pertinents et génération de la réponse..."):
            result = answer_question(question, top_k=int(top_k))

        st.markdown("### Réponse / Valiny")
        st.write(result["answer"])

        st.markdown("### Sources utilisées")
        if result["sources"]:
            for source in result["sources"]:
                st.write(
                    f"- **{source['source']}**, page **{source['page']}** "
                    f"— score indicatif : `{source['score']}`"
                )
        else:
            st.write("Aucune source disponible.")

        with st.expander("Voir le détail des passages récupérés (texte OCR ou brut)"):
            for i, ctx in enumerate(result["retrieved_contexts"], start=1):
                metadata = ctx.get("metadata", {})
                method = metadata.get("extraction_method", "inconnue")
                st.markdown(
                    f"**Passage {i} — `{metadata.get('source', 'inconnu')}` (Page {metadata.get('page', '?')})** "
                    f"| Méthode d'extraction : `{method}` | Score : `{ctx.get('score', 0):.3f}`"
                )
                st.info(ctx.get("text", ""))


st.divider()
st.subheader("3. Évaluer le système (Mémoire Master)")

with st.expander("📊 Lancer une évaluation automatisée à partir d'un fichier CSV"):
    st.write(
        "Permet de générer des métriques expérimentales (Hit Rate, MRR, couverture mots-clés) "
        "pour la partie expérimentation de votre mémoire de Master."
    )
    evaluation_file = st.file_uploader(
        "Importer un fichier CSV de questions de test",
        type=["csv"],
        accept_multiple_files=False,
        key="evaluation_csv",
    )
    eval_top_k = st.number_input("Top-k pour l'évaluation", min_value=1, max_value=10, value=TOP_K, key="eval_top_k")

    if evaluation_file is not None and st.button("📈 Calculer les métriques d'évaluation"):
        temp_path = RAW_DATA_DIR / "evaluation_questions.csv"
        with temp_path.open("wb") as f:
            f.write(evaluation_file.getbuffer())

        with st.spinner("Évaluation en cours..."):
            try:
                results_df = run_evaluation(temp_path, top_k=int(eval_top_k))
                report_paths = save_evaluation_report(results_df, output_dir="reports")
                summary = summarize_results(results_df)

                st.success("Évaluation terminée avec succès !")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Questions testées", summary.total_questions)
                col_b.metric("Retrieval Hit Rate", f"{summary.retrieval_hit_rate:.1%}")
                col_c.metric("MRR", f"{summary.mean_reciprocal_rank:.3f}")

                col_d, col_e, col_f = st.columns(3)
                col_d.metric("Sources exactes", f"{summary.answer_source_hit_rate:.1%}")
                col_e.metric("Couverture Mots-clés", f"{summary.keyword_coverage_mean:.1%}")
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
st.subheader("4. Exporter toutes les données indexées vers Excel")

with st.expander("📄 Créer un fichier Excel complet de la base documentaire"):
    st.write(
        "Exporte les documents indexés, les pages OCR extraites et les statistiques "
        "dans un fichier `.xlsx`."
    )
    include_embeddings_excel = st.checkbox(
        "Inclure les vecteurs d'embeddings dans Excel (attention à la taille du fichier)",
        value=False,
    )

    if st.button("📑 Générer l'export Excel général"):
        if count_documents() == 0:
            st.warning("Aucun document n'est indexé.")
        else:
            with st.spinner("Création du fichier Excel..."):
                try:
                    excel_path = export_extracted_data_to_excel(include_embeddings=include_embeddings_excel)
                    st.success(f"Fichier Excel créé : `{excel_path}`")
                    with open(excel_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Télécharger le fichier Excel",
                            data=f,
                            file_name=excel_path.name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                except Exception as exc:
                    st.error(f"Erreur pendant l'export Excel : {exc}")


st.divider()
st.subheader("5. Extraire une réponse personnalisée sur chaque document vers Excel")

with st.expander("📋 Poser une question identique à chaque document et exporter un tableau Excel"):
    st.write(
        "Exemple : `Quel est le nom de la personne à qui la facture est adressée ?` ou `Quel est l'objet du contrat ?`"
    )

    extraction_question = st.text_input(
        "Question à appliquer sur chaque document",
        value="Quel est le nom de la personne à qui la facture est adressée ?",
    )

    try:
        indexed_docs = list_indexed_documents()
    except Exception:
        indexed_docs = []

    doc_names = [doc["filename"] for doc in indexed_docs]
    selected_docs = st.multiselect(
        "Sélectionnez les documents à traiter (laissez vide pour tous)",
        options=doc_names,
        default=[],
    )

    max_context_chars = st.number_input(
        "Fenêtre maximale de contexte par document (caractères)",
        min_value=1000,
        max_value=20000,
        value=8000,
        step=1000,
        help="Réduire cette valeur pour économiser la RAM de votre PC.",
    )

    if st.button("📊 Lancer l'extraction personnalisée vers Excel"):
        if count_documents() == 0:
            st.warning("Aucun document n'est indexé.")
        elif not extraction_question.strip():
            st.warning("Veuillez saisir une question d'extraction.")
        else:
            with st.spinner("Interrogation du modèle document par document..."):
                try:
                    excel_path = export_answers_to_excel(
                        question=extraction_question.strip(),
                        selected_sources=selected_docs if selected_docs else None,
                        max_context_chars=int(max_context_chars),
                    )
                    st.success(f"Fichier Excel créé : `{excel_path}`")
                    with open(excel_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Télécharger le tableau de réponses Excel",
                            data=f,
                            file_name=excel_path.name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                except Exception as exc:
                    st.error(f"Erreur pendant l'extraction : {exc}")


st.divider()
st.subheader("6. 🧾 Mode Factures Adaptatif : Extraction structurée universelle vers Excel")

with st.expander("⚡ Extraire tout type de facture (même atypique ou avec des champs différents)"):
    st.write(
        "Ce mode s'adapte à **n'importe quelle facture ou document**. "
        "Vous pouvez soit extraire un schéma de colonnes sur-mesure, soit laisser l'IA **découvrir automatiquement** "
        "toutes les informations clés présentes sur chaque facture (Mode Universel)."
    )

    try:
        invoice_docs = list_indexed_documents()
    except Exception:
        invoice_docs = []

    invoice_doc_names = [doc["filename"] for doc in invoice_docs]
    selected_invoice_docs = st.multiselect(
        "Sélectionnez les factures à analyser (laissez vide pour toutes)",
        options=invoice_doc_names,
        default=[],
        key="invoice_docs_multiselect",
    )

    extraction_mode = st.radio(
        "Choix du mode d'extraction :",
        options=["Universel / Auto-découverte (L'IA extrait librement toutes les paires clé-valeur)", "Sur-mesure (Définir la liste des colonnes/champs)"],
        index=0,
    )

    custom_fields_list: list[str] | None = None
    mode_str = "universal"
    if "Sur-mesure" in extraction_mode:
        mode_str = "custom"
        fields_input = st.text_area(
            "Liste des champs à extraire (séparez par des virgules ou retours à la ligne) :",
            value=", ".join(DEFAULT_INVOICE_FIELDS[:6]),
            help="Adaptez cette liste aux besoins exacts de vos factures atypiques.",
        )
        raw_list = [f.strip() for f in fields_input.replace("\n", ",").split(",") if f.strip()]
        custom_fields_list = raw_list if raw_list else DEFAULT_INVOICE_FIELDS

    invoice_max_context_chars = st.number_input(
        "Taille de contexte par facture (caractères)",
        min_value=2000,
        max_value=30000,
        value=12000,
        step=1000,
        key="invoice_max_context_chars",
    )

    if st.button("🧾 Lancer l'extraction structurée adaptative", type="primary"):
        if count_documents() == 0:
            st.warning("Aucun document ou image n'est indexé.")
        else:
            with st.spinner("Analyse OCR et extraction structurée en cours..."):
                try:
                    df_invoices = extract_invoice_fields_for_all_documents(
                        selected_sources=selected_invoice_docs if selected_invoice_docs else None,
                        fields=custom_fields_list,
                        mode=mode_str,
                        max_context_chars=int(invoice_max_context_chars),
                    )
                    st.markdown("### 👁️ Prévisualisation du tableau d'extraction (avec Indice de Fiabilité %)")
                    st.dataframe(df_invoices, use_container_width=True)

                    invoice_excel_path = export_invoice_fields_to_excel(
                        selected_sources=selected_invoice_docs if selected_invoice_docs else None,
                        fields=custom_fields_list,
                        mode=mode_str,
                        max_context_chars=int(invoice_max_context_chars),
                    )
                    st.success(f"Fichier Excel créé et prêt pour le téléchargement : `{invoice_excel_path}`")
                    with open(invoice_excel_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Télécharger le fichier Excel structuré (.xlsx)",
                            data=f,
                            file_name=invoice_excel_path.name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                except Exception as exc:
                    st.error(f"Erreur pendant l'extraction des factures : {exc}")