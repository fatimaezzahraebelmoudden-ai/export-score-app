from io import BytesIO
import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(layout="wide", page_title="ExportScore")

# --- FONCTIONS UTILES ---
@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Rapport')
    processed_data = output.getvalue()
    return processed_data

def train_and_load(train_file, predict_file):
    try:
        st.session_state.model = joblib.load('best_export_model.pkl')
        st.session_state.df_opportunities = pd.read_excel(predict_file)
        if 'Produit' not in st.session_state.df_opportunities.columns or 'CODE SH' not in st.session_state.df_opportunities.columns:
            st.error("Le fichier de prédiction doit contenir les colonnes 'Produit' et 'CODE SH'.")
            return False
        return True
    except FileNotFoundError:
        st.error("Fichier 'best_export_model.pkl' introuvable. Veuillez l'entraîner et le placer dans le dossier.")
        return False
    except Exception as e:
        st.error(f"Erreur lors du traitement des fichiers : {e}")
        return False

# --- INITIALISATION DE L'ÉTAT DE SESSION ---
if 'page' not in st.session_state:
    st.session_state.page = 'config'
if 'model' not in st.session_state:
    st.session_state.model = None
if 'df_opportunities' not in st.session_state:
    st.session_state.df_opportunities = None
if 'selected_product_name' not in st.session_state:
    st.session_state.selected_product_name = None
if 'strategic_alignment' not in st.session_state:
    st.session_state.strategic_alignment = None
if 'report' not in st.session_state:
    st.session_state.report = []
if 'show_manual_form' not in st.session_state:
    st.session_state.show_manual_form = False

# --- CONSTANTES ---
# CORRECTION MAJEURE : Utiliser les noms de colonnes exacts de vos fichiers Excel
FEATURES_NAMES = [
    'Croissance Maroc (% p.a.)', 
    'ACR', 
    'PCI', 
    'Croissance Monde (% p.a.)', 
    'Taille Marche (millier USD)', 
    'Alignement stratégique'
]

# ==============================================================================
# --- PAGE 1 : CONFIGURATION DU MODÈLE ---
# ==============================================================================
if st.session_state.page == 'config':
    st.title("ExportScore")
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        with st.container(border=True):
            st.markdown("""<div style="text-align: center;"><h2 style="color: #D32F2F;">Configuration du Modèle</h2><p>Importez vos données pour initialiser le modèle</p></div>""", unsafe_allow_html=True)
            train_file = st.file_uploader("1. Base d'Apprentissage (.xlsx)", key="train_uploader", type="xlsx")
            predict_file = st.file_uploader("2. Base de Prédiction (.xlsx)", key="predict_uploader", type="xlsx")
            if st.button("Lancer l'Analyse", use_container_width=True, type="primary"):
                if train_file and predict_file:
                    with st.spinner('Initialisation du modèle et chargement des données...'):
                        success = train_and_load(train_file, predict_file)
                        if success:
                            st.session_state.page = 'dashboard'
                            st.success("Initialisation réussie ! Chargement du tableau de bord...")
                            time.sleep(2)
                            st.rerun()
                else:
                    st.warning("Veuillez importer les deux fichiers.")

# ==============================================================================
# --- PAGE 2 : TABLEAU DE BORD PRÉDICTIF ---
# ==============================================================================
elif st.session_state.page == 'dashboard':
    
    # --- BARRE LATÉRALE (SIDEBAR) ---
    with st.sidebar:
        st.image("logo.png", use_container_width=True)
        st.markdown(f"**MODE PRÉDICTIF**\n\n`{len(st.session_state.df_opportunities)} produits chargés`")
        st.markdown("---")

        if st.button("➕ Nouvelle Opportunité Manuelle", use_container_width=True):
            st.session_state.show_manual_form = not st.session_state.show_manual_form

        if st.session_state.show_manual_form:
            with st.form("manual_form"):
                st.subheader("Nouvelle Opportunité Manuelle")
                nom_produit = st.text_input("Nom du Produit")
                code_sh_manual = st.text_input("Code SH")
                croissance_maroc_manual = st.number_input("Croissance Maroc (%)", value=0.0)
                pci_manual = st.number_input("PCI (Indice Complexité)", value=0.0, format="%.4f")
                taille_marche_manual = st.number_input("Taille Marché (kUSD)", value=0)
                acr_manual = st.number_input("ACR (Avantage Comparatif)", value=0.0, format="%.4f")
                croissance_monde_manual = st.number_input("Croissance Monde (%)", value=0.0)
                alignement_manual = st.selectbox("Alignement Stratégique", [0, 1])

                if st.form_submit_button("Ajouter & Analyser", type="primary"):
                    new_product_data = pd.DataFrame([{
                        'Produit': nom_produit, 'CODE SH': code_sh_manual,
                        'Croissance Maroc (% p.a.)': croissance_maroc_manual,
                        'ACR': acr_manual, 'PCI': pci_manual,
                        'Croissance Monde (% p.a.)': croissance_monde_manual,
                        'Taille Marche (millier USD)': taille_marche_manual,
                        'Alignement stratégique': alignement_manual
                    }])
                    st.session_state.df_opportunities = pd.concat([new_product_data, st.session_state.df_opportunities], ignore_index=True).fillna(0)
                    st.session_state.selected_product_name = nom_produit
                    st.session_state.strategic_alignment = alignement_manual
                    st.session_state.show_manual_form = False
                    st.rerun()
        else:
            st.subheader("1. SÉLECTIONNER")
            search_hs = st.text_input("Filtrer par Code SH...")
            filtered_opportunities = st.session_state.df_opportunities
            if search_hs:
                filtered_opportunities = filtered_opportunities[filtered_opportunities['CODE SH'].astype(str).str.contains(search_hs, na=False)]
            
            product_list = filtered_opportunities['Produit'].tolist()
            st.session_state.selected_product_name = st.selectbox(" ", product_list, label_visibility="collapsed")

            st.markdown("---")
            st.subheader("2. ALIGNEMENT STRATÉGIQUE")
            st.warning("Requis pour le calcul :")
            
            cols = st.columns(2)
            if cols[0].button("Non (0)", key="align_no", use_container_width=True):
                st.session_state.strategic_alignment = 0
            if cols[1].button("Oui (1)", key="align_yes", use_container_width=True):
                st.session_state.strategic_alignment = 1

            if st.session_state.strategic_alignment is not None:
                st.success(f"Alignement sélectionné : **{'OUI' if st.session_state.strategic_alignment == 1 else 'NON'}**")

        st.markdown("---")
        if st.button("Recharger de nouveaux fichiers"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    # --- CORPS PRINCIPAL ---
    st.title("Tableau de Bord Prédictif")
    st.write("Évaluation des opportunités d'exportation pour le Maroc.")

    filtered_opportunities_main = st.session_state.df_opportunities
    if st.session_state.selected_product_name and not filtered_opportunities_main[filtered_opportunities_main['Produit'] == st.session_state.selected_product_name].empty:
        selected_row = filtered_opportunities_main[filtered_opportunities_main['Produit'] == st.session_state.selected_product_name].iloc[0]
        code_sh_selected = selected_row['CODE SH']
        st.info(f"● Produit sélectionné : **{st.session_state.selected_product_name}** (Code SH: {code_sh_selected})")

        if st.session_state.strategic_alignment is None:
            with st.container(border=True):
                st.info("Pour obtenir un score prédictif fiable, vous devez obligatoirement valider l'alignement stratégique dans le panneau latéral.")
        else:
            input_dict = selected_row.to_dict()
            input_dict['Alignement stratégique'] = st.session_state.strategic_alignment
            prediction_df = pd.DataFrame([input_dict]).fillna(0)
            prediction_df_final = prediction_df[FEATURES_NAMES]
            
            proba_succes = st.session_state.model.predict_proba(prediction_df_final)[0][1]

            with st.container(border=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("RÉSULTAT DE LA PRÉDICTION")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=int(proba_succes * 100),
                        title={'text': "Probabilité de Succès"},
                        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#28a745" if proba_succes > 0.5 else "#ff4b4b"}}))
                    fig.update_layout(height=200, margin=dict(t=0, b=0, l=20, r=20))
                    st.plotly_chart(fig, use_container_width=True)

                    if proba_succes > 0.5: st.success("✔ POTENTIEL ÉLEVÉ")
                    else: st.error("❌ POTENTIEL FAIBLE")
                    
                    if st.button("➕ Ajouter au Rapport"):
                        report_item = {"PRODUIT": st.session_state.selected_product_name, "CODE SH": code_sh_selected, "ALIGNEMENT": "OUI" if st.session_state.strategic_alignment == 1 else "NON", "SCORE FINAL": f"{proba_succes:.1%}", "VERDICT": "ÉLEVÉ" if proba_succes > 0.5 else "FAIBLE"}
                        st.session_state.report.append(report_item)
                        st.toast(f"'{st.session_state.selected_product_name}' ajouté au rapport !")
                with col2:
                    st.subheader("Détails des Indicateurs")
                    details_df = pd.DataFrame({
                        'CRITERE': ['Code SH', 'ACR', 'PCI', 'Croissance Maroc', 'Croissance Monde', 'Taille Marché'],
                        'VALEUR': [
                            code_sh_selected,
                            f"{prediction_df_final['ACR'].iloc[0]:.2f}",
                            f"{prediction_df_final['PCI'].iloc[0]:.2f}",
                            f"{prediction_df_final[FEATURES_NAMES[0]].iloc[0]:.1f}%",
                            f"{prediction_df_final[FEATURES_NAMES[3]].iloc[0]:.1f}%",
                            f"{prediction_df_final[FEATURES_NAMES[4]].iloc[0]/1000:.1f}M $" if prediction_df_final[FEATURES_NAMES[4]].iloc[0] > 0 else "N/A"
                        ],
                        'DESCRIPTION': ["Classification internationale", "Avantage Comparatif Révélé (> 1 = fort)", "Indice de Complexité du Produit", "Croissance annuelle des exportations", "Croissance annuelle du marché mondial", "Valeur totale du marché mondial"]
                    })
                    st.table(details_df)
    
    if st.session_state.report:
        st.markdown("---")
        st.subheader(f"Rapport d'Opportunités ({len(st.session_state.report)})")
        report_df = pd.DataFrame(st.session_state.report)
        st.dataframe(report_df, use_container_width=True)
        excel_data = convert_df_to_excel(report_df)
        st.download_button(
            label="📥 Télécharger le Rapport Excel",
            data=excel_data,
            file_name="rapport_opportunites.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")