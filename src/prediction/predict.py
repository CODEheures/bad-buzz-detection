import mlflow.pyfunc
from common import params, setup_mlflow
import streamlit as st
import numpy as np


def run():
    setup_mlflow.init_mlflow()
    production_version = mlflow.pyfunc.load_model(f"models:/{params.model_name}@{params.alias}")

    st.header("Predition de sentiment d'un tweet", divider='rainbow')

    st.write(production_version.metadata.get_model_info())
    tweet = st.text_input('Tweet à analyser:', placeholder='Entrez le tweet ici....')

    if tweet:
        production_version.predict(np.array[tweet])
