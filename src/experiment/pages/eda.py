import streamlit as st
import pandas as pd
from streamlit import session_state as ss
from st_pages import add_page_title

add_page_title()

st.header("Analyse des données uploadées", divider='rainbow')

if 'dataframe' in ss:
    df: pd.DataFrame = ss['dataframe'].sample(1000)

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