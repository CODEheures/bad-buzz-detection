import streamlit as st
from st_pages import add_page_title
from streamlit import session_state as ss
from streamlit_extras.switch_page_button import switch_page
from common import text_processing, setup_mlflow
from experiment import pages_management
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
import pandas as pd
import mlflow
import os


add_page_title()
setup_mlflow.init_tracking('air-paradis')

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
    text_transformer = Pipeline(
        steps=[
            ('preprocess_text', FunctionTransformer(text_processing.preprocess_text))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, ['text'])
        ]
    )


if (st.button("Lancer l'entrainement")):
    with st.status("Preprocess data...", expanded=True) as status:
        with mlflow.start_run():
            params = {
                "solver": "lbfgs",
                "max_iter": 1000,
                "multi_class": "auto",
                "random_state": 8888,
            }

            mlflow.log_param('test_params', params)
            st.write('Découpage en Train/Validation/Test')

            df = ss['dataframe']
            X = df.drop('target', axis=1)
            y = df['target'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size/100, random_state=seed)
            X_validation, X_test, y_validation, y_test = train_test_split(X_test,
                                                                          y_test,
                                                                          test_size=test_size/(test_size + validation_size),
                                                                          random_state=seed)

            st.markdown(
                f"""
|Jeu|Longeur|
|---|---|
|Entrainement|{len(X_train)}|
|Validation|{len(X_validation)}|
|Test|{len(X_test)}|
                """
            )

            X_transform = preprocessor.fit_transform(X_train, y_train)

            # Test MlFlow
            pd.DataFrame(X_transform).to_csv('test_artifact.csv')
            mlflow.log_artifact('test_artifact.csv')
            os.remove('test_artifact.csv') 

            st.write(pd.DataFrame(X_transform).sample(10, random_state=seed))
            status.update(label="Fin du traitement", state="complete")

if (train_ok and st.button("Je valide ce model et passe à la demande de déploiment", disabled=~train_ok)):
    ss['deploy_ok'] = True
    pages_management.update_pages()
    switch_page("Deploiement d'un Model")
