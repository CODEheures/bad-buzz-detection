import streamlit as st
from st_pages import add_page_title
import pandas as pd
from io import StringIO

add_page_title()
st.subheader("Uploadez vos données pour lancer l'analyse et acceder à la modélisation", divider="rainbow")

uploaded_file = st.file_uploader("Choisir un fichier csv", type=['csv'])
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
