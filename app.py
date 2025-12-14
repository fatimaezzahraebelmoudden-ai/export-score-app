from io import BytesIO
import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(layout="wide", page_title="ExportScore")

# --- FONCTIONS UTILES ---
# On définit toutes les fonctions en haut du script pour une meilleure organisation

@st.cache_data
def convert_df_to_excel(df):
    """Convertit un DataFrame en un fichier Excel en mémoire pour le téléchargement."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Rapport')
    processed_data = output.getvalue()
    return processed_data

def train_and_load(train_file, predict_file):
    """Charge le modèle pré-entraîné et les données de prédiction."""
    try:
        # NOTE : Idéalement, cette fonction prendrait train_file pour entraîner le modèle.
        # Pour cette application, nous supposons que le modèle est déjà entraîné et nous le chargeons.
        st.session_state.model = joblib.load('best_export_model.pkl')
        
        st.session_state.df_opportunities = pd.read_excel(predict_file)
        
        if 'Produit' not in st.session_state.df_opportunities.columns or 'CODE SH' not in st.session_state.df_opportunities.columns:
            st.error("Le fichier de prédiction doit contenir les colonnes 'Produit' et 'CODE SH'.")
            return False
        return True
    except FileNotFoundError:
        st.error("Fichier 'best_export_model.pkl' introuvable. Veuillez l'entraîner (avec train_model.py) et le placer dans le dossier.")
        return False
    except Exception as e:
        st.error(f"Erreur lors du traitement des fichiers : {e}")
        return False

# --- INITIALISATION DE L'ÉTAT DE SESSION ---
# Cela garantit que les variables persistent entre les rechargements de la page
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

# --- CONSTANTES ---
# Noms des caractéristiques EXACTEMENT comme dans les fichiers Excel et pour l'entraînement du modèle
FEATURES_NAMES = [
    'Croissance Maroc (% p.a.)', 
    'ACR', 
    'PCI', 
    'Croissance Monde (% p.a.)', 
    'Taille Marche (millier USD)', 
    'Alignement stratégique'
]

# ==============================================================================
# --- LOGIQUE D'AFFICHAGE DES PAGES ---
# ==============================================================================

# --- PAGE 1 : CONFIGURATION DU MODÈLE ---
if st.session_state.page == 'config':
    st.title("ExportScore")
    
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        with st.container(border=True):
            st.markdown("""
                <div style="text-align: center;">
                    <h2 style="color: #D32F2F;">Configuration du Modèle</h2>
                    <p>Importez vos données pour initialiser le modèle</p>
                </div>
            """, unsafe_allow_html=True)
            
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

# --- PAGE 2 : TABLEAU DE BORD PRÉDICTIF ---
elif st.session_state.page == 'dashboard':
    
    # --- BARRE LATÉRALE (SIDEBAR) ---
    with st.sidebar:
        st.image("logo.png", use_container_width=True)
        
    # ...        
        st.markdown(f"**MODE PRÉDICTIF**\n\n`{len(st.session_state.df_opportunities)} produits chargés`")
        st.markdown("---")
        st.subheader("1. SÉLECTIONNER")

        search_hs = st.text_input("Filtrer par Code SH...")
        filtered_opportunities = st.session_state.df_opportunities
        if search_hs:
            filtered_opportunities = filtered_opportunities[filtered_opportunities['CODE SH'].astype(str).str.contains(search_hs)]
        
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
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # --- CORPS PRINCIPAL ---
    st.title("Tableau de Bord Prédictif")
    st.write("Évaluation des opportunités d'exportation pour le Maroc.")

    if st.session_state.selected_product_name:
        code_sh_selected = filtered_opportunities[filtered_opportunities['Produit'] == st.session_state.selected_product_name]['CODE SH'].iloc[0]
        st.info(f"● Produit sélectionné : **{st.session_state.selected_product_name}** (Code SH: {code_sh_selected})")

    if st.session_state.strategic_alignment is None:
        with st.container(border=True):
            st.info("Pour obtenir un score prédictif fiable, vous devez obligatoirement valider l'alignement stratégique dans le panneau latéral.")
    else:
        selected_product_data = filtered_opportunities[filtered_opportunities['Produit'] == st.session_state.selected_product_name].iloc[0]

        input_dict = selected_product_data.to_dict()
        input_dict['Alignement stratégique'] = st.session_state.strategic_alignment
        prediction_df = pd.DataFrame([input_dict])[FEATURES_NAMES]
        
        proba_succes = st.session_state.model.predict_proba(prediction_df)[0][1]

        with st.container(border=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("RÉSULTAT DE LA PRÉDICTION")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=int(proba_succes * 100),
                    title={'text': "Probabilité de Succès"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#28a745" if proba_succes > 0.5 else "#ff4b4b"},
                    }))
                fig.update_layout(height=200, margin={'t':0, 'b':0, 'l':20, 'r':20})
                st.plotly_chart(fig, use_container_width=True)

                if proba_succes > 0.5:
                    st.success("✔ POTENTIEL ÉLEVÉ")
                else:
                    st.error("❌ POTENTIEL FAIBLE")
                
                if st.button("➕ Ajouter au Rapport"):
                    report_item = {
                        "PRODUIT": st.session_state.selected_product_name,
                        "CODE SH": code_sh_selected,
                        "ALIGNEMENT": "OUI" if st.session_state.strategic_alignment == 1 else "NON",
                        "SCORE FINAL": f"{proba_succes:.1%}",
                        "VERDICT": "ÉLEVÉ" if proba_succes > 0.5 else "FAIBLE",
                    }
                    st.session_state.report.append(report_item)
                    st.toast(f"'{st.session_state.selected_product_name}' ajouté au rapport !")

            with col2:
                st.subheader("Détails des Indicateurs")
                # On utilise les noms de colonnes corrigés pour récupérer les valeurs
                details_df = pd.DataFrame({
                    'CRITERE': ['Code SH', 'ACR', 'PCI', 'Croissance Maroc', 'Croissance Monde', 'Taille Marché'],
                    'VALEUR': [
                        code_sh_selected,
                        f"{prediction_df['ACR'].iloc[0]:.1f}",
                        f"{prediction_df['PCI'].iloc[0]:.2f}",
                        f"{prediction_df[FEATURES_NAMES[0]].iloc[0]:.1f}%",
                        f"{prediction_df[FEATURES_NAMES[3]].iloc[0]:.1f}%",
                        f"{prediction_df[FEATURES_NAMES[4]].iloc[0]/1000:.1f}M $"
                    ],
                    'DESCRIPTION': [
                        "Classification internationale des produits",
                        "Mesure la spécialisation du Maroc (> 1 = fort)",
                        "Niveau technologique (Positif = haute valeur)",
                        "Taux de croissance annuel des exportations",
                        "Taux de croissance annuel du marché mondial",
                        "Valeur totale du marché mondial"
                    ]
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
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )