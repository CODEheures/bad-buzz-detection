import streamlit as st
from st_pages import add_page_title
from streamlit import session_state as ss
from streamlit_extras.switch_page_button import switch_page
from common import pipelines, splitter
from experiment import pages_management
import plotly.graph_objects as go
import pandas as pd
import mlflow
import os


add_page_title()

train_ok = False
seed = 1234
st.header("Cette page permet d'entrainer un model et d'obtenir les scores et graphiques associés", divider='rainbow')

st.markdown("### 1. Split Train/Validation/Test")
cuts = st.slider("Proportion Entrainement/Validation/Test", min_value=10, max_value=90, value=(60, 80))
train_size = cuts[0]
validation_size = cuts[1] - cuts[0]
test_size = 100 - cuts[1]

colors = ['gold', 'darkorange', 'lightgreen']

fig = go.Figure(data=[go.Pie(labels=['Entrainement', 'Validation', 'Test'],
                             values=[train_size, validation_size, test_size])])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
st.plotly_chart(fig)

st.markdown("### 2. Choix du Model et Paramètres")
model_name = st.selectbox("Choix du model",
                          ["SVM", "Deep Neural"],
                          index=None,
                          placeholder="Selectionnez un model...")

if (model_name == "SVM"):
    min_df = st.slider("Ignorer les mots qui apparaisent moins de n fois", min_value=1, max_value=10, value=1)
    max_df = st.slider("ignorer les x% des mots les plus fréqents", min_value=0, max_value=10, value=0)
    n_gram_range = st.slider('Rang des N_grams', min_value=1, max_value=4, value=(1, 1))
    preprocessor = pipelines.svm(min_df=min_df,
                                 max_df=(1-max_df/100),
                                 ngram_range=n_gram_range)
    model_ready = True
else:
    model_ready = False


if (model_ready and st.button("Lancer l'entrainement")):
    with st.status("Preprocess data...", expanded=True) as status:
        with mlflow.start_run():
            st.markdown('1. Découpage en Train/Validation/Test')
            X_train, X_validation, X_test, y_train, y_validation, y_test = splitter.split(train_size=train_size,
                                                                                          validation_size=validation_size,
                                                                                          test_size=test_size)

            st.markdown(
                f"""
|Jeu|Longeur|
|---|---|
|Entrainement|{len(X_train)}|
|Validation|{len(X_validation)}|
|Test|{len(X_test)}|
                """
            )

            st.markdown('2. Préprocessing')
            X_transform = preprocessor.fit_transform(X_train, y_train)

            # Test MlFlow
            mlflow.log_param('test_params', preprocessor.get_params(deep=False))
            pd.DataFrame(X_transform).to_csv('test_artifact.csv')
            mlflow.log_artifact('test_artifact.csv')
            os.remove('test_artifact.csv')

            st.write(pd.DataFrame(X_transform).sample(10, random_state=seed))
            status.update(label="Fin du traitement", state="complete")

if (train_ok and st.button("Je valide ce model et passe à la demande de déploiment", disabled=~train_ok)):
    ss['deploy_ok'] = True
    pages_management.update_pages()
    switch_page("Deploiement d'un Model")
