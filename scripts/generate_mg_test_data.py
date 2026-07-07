"""Générateur automatique de documents et factures de test (Français & Malagasy) pour Madagascar.

Ce script génère 4 images réalistes de factures et documents administratifs (format PNG)
dans le répertoire `data/raw/`. Idéal pour tester l'OCR, le RAG et l'extraction
structurée adaptative lors de vos démonstrations et soutenances de Master.
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def get_font(size: int = 16, bold: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Tente de charger une police propre ou se rabat sur la police par défaut."""
    common_fonts = [
        "arial.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
        "calibri.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for font_path in common_fonts:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def create_invoice_image(
    filename: str,
    title: str,
    subtitle: str,
    sections: list[tuple[str, list[str]]],
    output_dir: Path,
) -> Path:
    """Crée une image de facture structurée avec en-têtes, encadrés et lignes de texte."""
    width, height = 850, 1100
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    font_title = get_font(26, bold=True)
    font_subtitle = get_font(18, bold=True)
    font_header = get_font(16, bold=True)
    font_body = get_font(15, bold=False)

    # En-tête / Bordure décorative
    draw.rectangle([20, 20, width - 20, height - 20], outline="#1A365D", width=3)
    draw.rectangle([20, 20, width - 20, 110], fill="#1A365D")
    draw.text((40, 35), title, fill="white", font=font_title)
    draw.text((40, 70), subtitle, fill="#E2E8F0", font=font_subtitle)

    y = 140
    for header, lines in sections:
        draw.rectangle([40, y, width - 40, y + 32], fill="#EDF2F7")
        draw.text((50, y + 6), header, fill="#2D3748", font=font_header)
        y += 45

        for line in lines:
            if ":" in line:
                key, val = line.split(":", 1)
                draw.text((55, y), key.strip() + " :", fill="#4A5568", font=font_header)
                draw.text((260, y), val.strip(), fill="#1A202C", font=font_body)
            else:
                draw.text((55, y), line, fill="#1A202C", font=font_body)
            y += 28
        y += 20

    # Pied de page
    draw.line([40, height - 70, width - 40, height - 70], fill="#CBD5E0", width=1)
    draw.text(
        (40, height - 55),
        "Document généré automatiquement pour le projet de fin d'études Master RAG / OCR — Antananarivo, Madagascar.",
        fill="#718096",
        font=get_font(12),
    )

    output_path = output_dir / filename
    img.save(output_path, "PNG")
    return output_path


def generate_all_test_documents() -> list[Path]:
    """Génère le jeu complet de documents de démonstration dans data/raw/."""
    raw_dir = Path("data") / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    docs = []

    # 1. Facture JIRAMA
    docs.append(
        create_invoice_image(
            filename="facture_jirama_antananarivo.png",
            title="JIRO SY RANO MALAGASY (JIRAMA)",
            subtitle="FAKTIORA / FACTURE DE CONSOMMATION ÉLECTRIQUE",
            sections=[
                ("INFORMATIONS ÉMETTEUR", [
                    "Émetteur : JIRAMA Direction Générale Antananarivo",
                    "Adresse : Rue Jean Ralaimongo, Ambohijatovo, Antananarivo 101",
                    "NIF : 1000112233 | STAT : 35111112019000111",
                ]),
                ("DESTINATAIRE / MPANJIFA", [
                    "Destinataire Nom : Monsieur RAKOTOMALALA Jean",
                    "Destinataire Adresse : Lot II B 45 Ankadivato, Antananarivo 101",
                    "Numéro Compteur : 9988776655",
                    "Référence Client : JIR-ANT-2026-991",
                ]),
                ("DÉTAILS DE LA FACTURE / LALANA FAKTIORA", [
                    "Numero Facture : JIR-2026-07-8892",
                    "Date Facture : 05/07/2026",
                    "Date Limite de Paiement : 25/07/2026",
                    "Période de Consommation : Juin 2026 (Consommation : 180 kWh)",
                ]),
                ("RÉSUMÉ DES MONTANTS", [
                    "Montant HT : 120 833 Ar",
                    "Montant TVA (20%) : 24 167 Ar",
                    "Montant TTC : 145 000 Ar",
                    "Devise : Ariary (MGA / Ar)",
                    "Commentaire : Facture réglable par MVola, Orange Money ou en agence.",
                ]),
            ],
            output_dir=raw_dir,
        )
    )

    # 2. Facture TELMA Madagascar
    docs.append(
        create_invoice_image(
            filename="faktiora_telma_fibre.png",
            title="TELMA MADAGASCAR — TELECOM NETWORK",
            subtitle="FAKTIORA INTERNET FIBRE PRO & MOBILE",
            sections=[
                ("EMETTEUR", [
                    "Emetteur Nom : TELMA SA Madagascar",
                    "Adresse Emetteur : Zone Galaxy Andraharo, Antananarivo 101",
                    "NIF : 2000445566 | STAT : 61101112015000222",
                ]),
                ("MPANJIFA / DESTINATAIRE", [
                    "Destinataire Nom : Société TSARA INFORMATIQUE SARL",
                    "Destinataire Adresse : Immeuble Fitaratra, Ankorondrano, Antananarivo 101",
                    "Numéro de Ligne : 034 00 123 45",
                ]),
                ("ÉLÉMENTS DE FACTURATION", [
                    "Numero Facture : TEL-FIB-2026-4410",
                    "Date Facture : 01/07/2026",
                    "Abonnement : Forfait Fibre Optique Pro 100 Mbps",
                ]),
                ("MONTANTS À PAYER / TOTALY", [
                    "Montant HT : 241 667 Ar",
                    "Montant TVA : 48 333 Ar",
                    "Montant TTC : 290 000 Ar",
                    "Devise : Ariary (Ar)",
                    "Commentaire : Prélèvement automatique le 15 du mois.",
                ]),
            ],
            output_dir=raw_dir,
        )
    )

    # 3. Facture commerciale bilingue
    docs.append(
        create_invoice_image(
            filename="facture_prestation_bilingue.png",
            title="RAVALO CONSULTING & SERVICES SARL",
            subtitle="FAKTIORA / FACTURE DE PRESTATION D'AUDIT IA",
            sections=[
                ("INFORMATIONS PRESTATAIRE (EMETTEUR)", [
                    "Emetteur Nom : RAVALO CONSULTING SARL",
                    "Adresse : Lot IV K 12 Isoraka, Antananarivo 101",
                    "Contact : contact@ravalo-consulting.mg | Tél : 038 11 222 33",
                ]),
                ("INFORMATIONS CLIENT (DESTINATAIRE)", [
                    "Destinataire Nom : ONG MADAGASCAR DIGITAL INNOVATION",
                    "Destinataire Adresse : Villa Darafify, Ambatobe, Antananarivo 103",
                    "NIF Client : 4000998877",
                ]),
                ("RÉFÉRENCES", [
                    "Numero Facture : FAK-2026-0142",
                    "Date Facture : 07/07/2026",
                    "Prestation : Mise en place et formation sur système RAG documentaire.",
                ]),
                ("TOTAL DU RÈGLEMENT", [
                    "Montant HT : 1 500 000 Ar",
                    "Montant TVA (20%) : 300 000 Ar",
                    "Montant TTC : 1 800 000 Ar",
                    "Devise : Ariary (MGA)",
                    "Commentaire : Règlement par virement bancaire BNI Madagascar sous 30 jours.",
                ]),
            ],
            output_dir=raw_dir,
        )
    )

    # 4. Attestation administrative juridique (Fokontany)
    docs.append(
        create_invoice_image(
            filename="attestation_residence_fokontany.png",
            title="REPOBLIKAN'I MADAGASIKARA",
            subtitle="Fitiavana - Tanindrazana - Fandrosoana | KAOMININA ANTANANARIVO RENIVOHITRA",
            sections=[
                ("FOKONTANY AMBOHIJATOVO ATSIMO", [
                    "Document : FANAMAFISANA FONENANA / ATTESTATION DE RÉSIDENCE",
                    "Réf Administrative : N° 452/2026-FKT/AMB",
                    "Date de Délivrance : 04/07/2026",
                ]),
                ("CONCERNANT LE CITOYEN", [
                    "Anarana sy Fanampiny (Nom) : RANDRIAMANANJARA Mamy",
                    "Daty nahaterahana (Date de naissance) : 12/05/1990 tao Antananarivo",
                    "Asa atao (Profession) : Injeniera amin'ny kajy mirindra (Ingénieur Informatique)",
                    "Adiresy (Adresse exacte) : Lot V B 88 Ambohijatovo Atsimo, Antananarivo 101",
                ]),
                ("DÉCLARATION DU CHEF DE FOKONTANY", [
                    "Attestation : Nous certifions que la personne susnommée réside effectivement dans notre Fokontany.",
                    "Usage : Pour servir et valoir ce que de droit (Dossier de soutenance Master).",
                    "Fanaovan-tsonia : Le Chef de Fokontany (Signé et tamponné).",
                ]),
            ],
            output_dir=raw_dir,
        )
    )

    return docs


if __name__ == "__main__":
    generated = generate_all_test_documents()
    print(f"✅ {len(generated)} documents et factures générés avec succès dans data/raw/ :")
    for p in generated:
        print(f" - {p}")