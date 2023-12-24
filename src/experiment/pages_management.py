import os
from streamlit import session_state as ss
from st_pages import Page, show_pages


def update_pages():
    # Specify what pages should be shown in the sidebar, and what their titles and icons
    # should be
    current_dir = os.path.normpath(os.path.dirname(__file__))

    pages = [
        Page(os.path.join(current_dir, "pages", "home.py"), "Accueil", ":house:")
    ]

    if ('mlflow_ready' in ss):
        pages.append(Page(os.path.join(current_dir, "pages", "upload.py"), "Import données"))

    if ('dataframe' in ss):
        pages.append(Page(os.path.join(current_dir, "pages", "eda.py"), "Analyse des données"))

    if ('analyse_ok' in ss):
        pages.append(Page(os.path.join(current_dir, "pages", "split.py"), "Partitionner les données"))

    if ('split_ok' in ss):
        pages.append(Page(os.path.join(current_dir, "pages", "choice.py"), "Choix et paramétrage model"))

    if ('choice_ok' in ss):
        pages.append(Page(os.path.join(current_dir, "pages", "train.py"), "Entrainement model"))

    if ('train_ok' in ss):
        pages.append(Page(os.path.join(current_dir, "pages", "publish.py"), "Publication model"))

    show_pages(
        pages
    )
