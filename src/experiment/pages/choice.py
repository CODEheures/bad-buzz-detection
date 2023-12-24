import streamlit as st
from st_pages import add_page_title
from streamlit import session_state as ss
from common import pipelines, params
from streamlit_extras.switch_page_button import switch_page
from experiment import pages_management


add_page_title()

st.header("Cette page permet de choisir et de paramétrer un model", divider='rainbow')

st.markdown("### 2. Choix du Model et Paramètres")
seleted_model = st.selectbox("Choix du model",
                             [member.name for member in params.model_enum],
                             index=None,
                             placeholder="Selectionnez un model...")

if (seleted_model == params.model_enum.SVM.name):
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
                          seed=params.seed)

if seleted_model:
    ss['selected_model'] = seleted_model
    ss['model'] = model
    ss['choice_ok'] = True
    pages_management.update_pages()
    if st.button("Entrainer ce model"):
        switch_page("Entrainement model")
