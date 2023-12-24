import streamlit as st
from streamlit import session_state as ss
from st_pages import add_page_title
from experiment import pages_management
from streamlit_extras.switch_page_button import switch_page
from common import setup_mlflow


setup_mlflow.init_tracking('air-paradis')
add_page_title()

st.subheader("Entrainement en ligne de modèles pour de l'analyse de sentiment\n\
                Auteur: Sylvain Gagnot", divider="rainbow")

st.markdown("""
Cette application permet de réaliser des entrainements en ligne de modèles pour de l'analyse de sentiment.
Les étapes sont les suivantes:
1. Import des données
2. Analyse des données importées
3. Entrainement paramétré d'un modèle
4. Envoi de la demande de publication si le modèle donne des résultats satisfaisants
            """)

if ('mlflow_ready' in ss):
    pages_management.update_pages()
    if (st.button("Importez les données")):
        switch_page("Import données")
