import streamlit as st
from st_pages import add_page_title
from streamlit import session_state as ss
import mlflow
import pandas as pd
from common import params
from streamlit_extras.switch_page_button import switch_page
from experiment import pages_management


add_page_title()
st.header("Cette page permet d'entrainer un model et d'obtenir les scores et graphiques associés", divider='rainbow')

with st.status("Preprocess data...", expanded=True) as status:
    if (('train_ok' not in ss) or (ss['train_ok'] is False)):
        mlflow.end_run()
        with mlflow.start_run() as run:
            mlflow.sklearn.autolog()
            if (ss['selected_model'] == params.model_enum.SVM.name):
                st.markdown('1. Entrainement')
                ss['model'].fit(ss['X_train'], ss['y_train'])
                st.write([f"{key}: {value:.4f}"
                         for key, value
                         in mlflow.get_run(run_id=run.info.run_id).data.metrics.items()])
                st.markdown('2. Validation')
                ss['score'] = ss['model'].score(ss['X_validation'], ss['y_validation'])
                st.write(f"Score du model: {ss['score']:.4f}")

                df_test = pd.DataFrame(ss['X_test'], columns=['text'])
                df_test['target'] = ss['y_test']
                dataset = mlflow.data.from_pandas(df_test)
                mlflow.log_input(dataset, context='test')

                ss['run'] = run

            status.update(label="Fin du traitement", state="complete", expanded=True)
            ss['train_ok'] = True
    else:
        st.write([f"{key}: {value:.4f}"
                  for key, value
                  in mlflow.get_run(run_id=ss['run'].info.run_id).data.metrics.items()])
        st.markdown('2. Validation')
        st.write(f"Score du model: {ss['score']:.4f}")
        status.update(label="Fin du traitement", state="complete", expanded=True)


if (ss['train_ok']):
    pages_management.update_pages()
    user_name = st.text_input('Votre nom', placeholder='Renseigner votre nom...')
    description = st.text_input('Description de l\'entrainement', placeholder='Donner une description de cet entrainement...')
    if user_name and description and st.button("Publier ce model"):
        ss['user_name'] = user_name
        ss['description'] = description
        switch_page("Publication model")
