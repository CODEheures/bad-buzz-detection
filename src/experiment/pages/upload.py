import streamlit as st
from streamlit import session_state as ss
from st_pages import add_page_title
from streamlit_extras.switch_page_button import switch_page
import pandas as pd
import home

add_page_title()
st.subheader("Importez vos données pour lancer l'analyse et acceder à la modélisation", divider="rainbow")

st.markdown("""
Le fichier doit être au format csv zippé

Le séparateur doit être la ','

Les colonnes suivantes sont obligatoires pour être exploitables par les modèles:
1. "target": La cible de sentiment
    - 0 = negative
    - 2 = neutre
    - 4 = positive
1. "text": Le text des tweets
""")

target_col = st.number_input("Numéro de la colonne target", min_value=0, max_value=10, value=0)
text_col = st.number_input("Numéro de la colonne text", min_value=0, max_value=10, value=5)

uploaded_file = st.file_uploader("Choisir un fichier csv", type=['zip'], key='uploaded_file')


if uploaded_file:
    dataframe = pd.read_csv(uploaded_file,
                            compression='zip',
                            header=None,
                            usecols=[target_col, text_col],
                            names=['target', 'text'],
                            encoding='ISO-8859-1'
                            )
    nb_observations = dataframe.shape[0]
    st.write(f'Il y a {nb_observations} tweets')
    st.write('Apperçu:')
    st.write(dataframe.sample(5))
    ss['dataframe'] = dataframe
    # dataframe.sample(100000).to_csv("tweets", index=False)

    home.update_pages()
    if st.button('Analyser les données'):
        switch_page("Analyse des données")
