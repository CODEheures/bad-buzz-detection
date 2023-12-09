import streamlit as st
from st_pages import add_page_title
from streamlit import session_state as ss
from streamlit_extras.switch_page_button import switch_page
import home

add_page_title()

st.header("Cette page permet d'entrainer un model et d'obtenir les scores et graphiques associés")

if (st.button("Je valide ce model et passe à la demande de déploiment")):
    ss['deploy_ok'] = True
    home.update_pages()
    switch_page("Deploiement d'un Model")
