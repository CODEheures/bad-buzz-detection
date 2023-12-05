import os
import streamlit as st
from st_pages import Page, show_pages, add_page_title

# Emojies short codes: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/


def run():
    add_page_title()

    # Specify what pages should be shown in the sidebar, and what their titles and icons
    # should be
    current_dir = os.path.dirname(__file__)
    show_pages(
        [
            Page(current_dir + "/home.py", "Accueil", ":house:"),
            Page(current_dir + "/pages/upload.py", "Import données"),
            Page(current_dir + "/pages/eda.py", "Analyse des données"),
            Page(current_dir + "/pages/train.py", "Entrainement d'un Model"),
            Page(current_dir + "/pages/deploy.py", "Deploiement d'un Model"),
        ]
    )


def test_function():
    st.write('Hello world Sylvain')
    return 5


def main():
    run()


if __name__ == "__main__":
    main()
