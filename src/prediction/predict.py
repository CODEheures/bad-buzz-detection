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

        ss['predictions'].append(
            {"tweet": tweet, "predict": response}
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
