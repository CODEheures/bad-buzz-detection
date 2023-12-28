import streamlit as st
from streamlit import session_state as ss
from api import predict


def run():
    if ('predictions' not in ss):
        ss['predictions'] = []

    st.header("Prediction de sentiment d'un tweet", divider='rainbow')

    tweet = st.text_input('Tweet à analyser:', placeholder='Entrez le tweet ici....')

    if tweet:
        response = predict.run(tweet=tweet)
        ss['predictions'].append(response)

    table = ''
    for prediction in reversed(ss['predictions']):
        table = table + f'\n|{prediction.tweet}|{prediction.icon_positif}|{prediction.icon_negatif}|'

    st.markdown(
        f"""
|Tweet|Positif|Negatif
|---|:-:|:-:|{table}
        """
        )
