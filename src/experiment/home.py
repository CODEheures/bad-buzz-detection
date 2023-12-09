import os
import streamlit as st
from streamlit import session_state as ss
from st_pages import Page, show_pages, add_page_title
from streamlit_extras.switch_page_button import switch_page

# Emojies short codes: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/


def update_pages():
    # Specify what pages should be shown in the sidebar, and what their titles and icons
    # should be
    current_dir = os.path.dirname(__file__)

    pages = [
        Page(current_dir + "/home.py", "Accueil", ":house:"),
        Page(current_dir + "/pages/upload.py", "Import données"),
    ]

    if ('dataframe' in ss):
        pages.append(Page(current_dir + "/pages/eda.py", "Analyse des données"))

    if ('analyse_ok' in ss):
        pages.append(Page(current_dir + "/pages/train.py", "Entrainement d'un Model"))

    if ('deploy_ok' in ss):
        pages.append(Page(current_dir + "/pages/deploy.py", "Deploiement d'un Model"))

    show_pages(
        pages
    )


def run():
    add_page_title()
    update_pages()

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

    if (st.button("Importez les données")):
        switch_page("Import données")

def test_function():
    st.write('Hello world Sylvain')
    return 5


def main():
    run()


if __name__ == "__main__":
    main()
