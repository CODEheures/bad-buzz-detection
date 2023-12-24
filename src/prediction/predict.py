import mlflow.pyfunc
from common import params, setup_mlflow
import streamlit as st
from streamlit import session_state as ss
import pandas as pd


def run():
    setup_mlflow.init_mlflow()

    if ('predictions' not in ss):
        ss['predictions'] = []

    production_version = mlflow.pyfunc.load_model(f"models:/{params.model_name}@{params.alias}")

    st.header("Prediction de sentiment d'un tweet", divider='rainbow')

    # st.write(production_version.metadata)
    tweet = st.text_input('Tweet à analyser:', placeholder='Entrez le tweet ici....')

    if tweet:
        predict = production_version.predict(pd.DataFrame([tweet]))

        ss['predictions'].append(
            {"tweet": tweet, "predict": predict}
        )

    table = ''
    for prediction in reversed(ss['predictions']):
        p_tweet = prediction['tweet']
        if prediction['predict'] == 0:
            positif = ''
            negagtif = ':-1:'
        else:
            positif = ':+1:'
            negagtif = ''
        table = table + f'\n|{p_tweet}|{positif}|{negagtif}|'

    st.markdown(
        f"""
|Tweet|Positif|Negatif
|---|:-:|:-:|{table}
        """
        )
