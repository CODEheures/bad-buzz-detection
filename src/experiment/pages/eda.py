import streamlit as st
import pandas as pd
from streamlit import session_state as ss
from streamlit_extras.switch_page_button import switch_page
from st_pages import add_page_title
import home

add_page_title()

st.header("Analyse des données importées", divider='rainbow')

if 'dataframe' in ss:
    n_rows = st.number_input("Nombre de tweets à analyser", min_value=1000, max_value=100000, value=1000)
    if n_rows is not None:
        df: pd.DataFrame = ss['dataframe'].sample(n_rows)

        st.markdown('### Apperçu')
        observations = len(df)
        st.write(f'Il y a {observations} tweets dans ce jeu de données d\'entrainement')
        st.write(df.head(10))

        st.markdown('### Doublons')
        duplicated = len(df[df.duplicated('text')])
        st.write(f'Il y a {duplicated} tweet dupliqués')

        st.markdown('### Répartition de la Target en %')
        st.write("0 = Négatif | 4 = positif")
        st.bar_chart(df['target'].value_counts(normalize=True))

        ss['analyse_ok'] = True
        home.update_pages()
        if st.button("Entrainement d'un Model"):
            switch_page("Entrainement d'un Model")
