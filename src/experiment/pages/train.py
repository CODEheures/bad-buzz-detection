import streamlit as st
from st_pages import add_page_title
from streamlit import session_state as ss
from common import pipelines, splitter, setup_mlflow
import plotly.graph_objects as go
import mlflow
import pandas as pd
from enum import Enum


add_page_title()
seed = 1234
model_enum = Enum('Model', ['SVM', 'Deep Neural'])
st.header("Cette page permet d'entrainer un model et d'obtenir les scores et graphiques associés", divider='rainbow')


def step_split():
    with st.container():
        st.markdown("### 1. Split Train/Validation/Test")

        df = ss['dataframe']
        sample_count = st.number_input("Nombre d'observation pour cet entrainement:",
                                       min_value=100,
                                       max_value=len(df),
                                       value=len(df))

        cuts = st.slider("Proportion Entrainement/Validation/Test", min_value=10, max_value=90, value=(60, 80))
        train_size = cuts[0]
        validation_size = cuts[1] - cuts[0]
        test_size = 100 - cuts[1]
        split = splitter.split(sample_count=sample_count,
                               train_size=train_size,
                               validation_size=validation_size,
                               test_size=test_size)

        ss['X_train'] = split[0]
        ss['X_validation'] = split[1]
        ss['X_test'] = split[2]
        ss['y_train'] = split[3]
        ss['y_validation'] = split[4]
        ss['y_test'] = split[5]

        col1, col2 = st.columns([3, 1])
        with col1:
            colors = ['gold', 'darkorange', 'lightgreen']

            fig = go.Figure(data=[go.Pie(labels=['Entrainement', 'Validation', 'Test'],
                            values=[train_size, validation_size, test_size])])
            fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                              marker=dict(colors=colors, line=dict(color='#000000', width=2)))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown(
                f"""
|Jeu|Longeur|
|---|---|
|Entrainement|{len(ss['X_train'])}|
|Validation|{len(ss['X_validation'])}|
|Test|{len(ss['X_test'])}|
                    """
                    )

        if (st.button("Choisir et entrainer un model")):
            ss['train_step'] = 2


def step_select_model():
    with st.container():
        st.markdown("### 2. Choix du Model et Paramètres")
        seleted_model = st.selectbox("Choix du model",
                                     [member.name for member in model_enum],
                                     index=None,
                                     placeholder="Selectionnez un model...")

        if (seleted_model == model_enum.SVM.name):
            min_df = st.slider("Ignorer les mots qui apparaisent moins de n fois", min_value=1, max_value=10, value=1)
            max_df = st.slider("ignorer les x% des mots les plus fréqents", min_value=0, max_value=10, value=0)
            n_gram_range = st.slider('Rang des N_grams', min_value=1, max_value=4, value=(1, 1))
            C = st.slider("Param C", min_value=0.0, max_value=10.0, value=1.0, step=0.1)  # noqa: N806
            degree = st.slider("Degree", min_value=0, max_value=10, value=3)
            kernel = st.selectbox("Kernel",
                                  ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                                  index=2,
                                  placeholder="Selectionnez un kernel...")
            gamma = st.selectbox("Gamma",
                                 ['scale', 'auto'],
                                 index=0,
                                 key='scale',
                                 placeholder="Selectionnez un gamma...")

            model = pipelines.svm(min_df=min_df,
                                  max_df=(1-max_df/100),
                                  ngram_range=n_gram_range,
                                  C=C,   # noqa= N803
                                  degree=degree,
                                  kernel=kernel,
                                  gamma=gamma,
                                  seed=seed)

        if seleted_model:
            ss['selected_model'] = seleted_model
            ss['model'] = model
            if (st.button("entrainer le model")):
                ss['train_step'] = 3


def step_train_model():
    with st.status("Preprocess data...", expanded=True) as status:
        mlflow.end_run()
        with mlflow.start_run() as run:
            mlflow.autolog()
            if (ss['selected_model'] == model_enum.SVM.name):
                st.markdown('1. Entrainement')
                ss['model'].fit(ss['X_train'], ss['y_train'])
                st.write([f"{key}: {value:.4f}"
                         for key, value
                         in mlflow.get_run(run_id=run.info.run_id).data.metrics.items()])
                st.markdown('2. Validation')
                score = ss['model'].score(ss['X_validation'], ss['y_validation'])
                st.write(f"Score du model: {score:.4f}")

                df_test = pd.DataFrame(ss['X_test'], columns=['text'])
                df_test['target'] = ss['y_test']
                dataset = mlflow.data.from_pandas(df_test)
                mlflow.log_input(dataset, context='test')

            status.update(label="Fin du traitement", state="complete", expanded=True)

            if (st.button("Publier le model")):
                ss['train_step'] = 4


def step_publish():
    setup_mlflow.publish_model()
    st.write('hello')

if ('train_step' not in ss):
    step_split()
elif (ss['train_step'] == 2):
    step_select_model()
elif (ss['train_step'] == 3):
    step_train_model()
elif (ss['train_step'] == 4):
    step_publish()
    