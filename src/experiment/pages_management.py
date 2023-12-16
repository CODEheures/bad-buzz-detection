import os
from streamlit import session_state as ss
from st_pages import Page, show_pages


def update_pages():
    # Specify what pages should be shown in the sidebar, and what their titles and icons
    # should be
    current_dir = os.path.dirname(__file__)

    pages = [
        Page(current_dir + "/home.py", "Accueil", ":house:")
    ]

    if ('mlflow_ready' in ss):
        pages.append(Page(current_dir + "/pages/upload.py", "Import données"))

    if ('dataframe' in ss):
        pages.append(Page(current_dir + "/pages/eda.py", "Analyse des données"))

    if ('analyse_ok' in ss):
        pages.append(Page(current_dir + "/pages/train.py", "Entrainement d'un Model"))

    if ('deploy_ok' in ss):
        pages.append(Page(current_dir + "/pages/deploy.py", "Deploiement d'un Model"))

    show_pages(
        pages
    )
