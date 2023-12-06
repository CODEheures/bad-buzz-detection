import streamlit as st
from streamlit import session_state as ss
from st_pages import add_page_title
from streamlit_extras.switch_page_button import switch_page
import pandas as pd


add_page_title()
st.subheader("Uploadez vos données pour lancer l'analyse et acceder à la modélisation", divider="rainbow")

uploaded_file = st.file_uploader("Choisir un fichier csv", type=['csv'], key='uploaded_file',)


if uploaded_file:
    dataframe = pd.read_csv(uploaded_file, header=None, names=['target', 'id', 'date', 'flag', 'user', 'text'])
    nb_observations = dataframe.shape[0]
    st.write(f'Vos données contiennent {nb_observations} observations')
    st.write('Apperçu:')
    st.write(dataframe.head(5))
    ss['dataframe'] = dataframe

    if st.button('Analyser les données'):
        switch_page("Analyse des données")
