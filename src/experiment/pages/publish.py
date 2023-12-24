import streamlit as st
from st_pages import add_page_title
from common import setup_mlflow


add_page_title()
st.header("Publication du model", divider='rainbow')

setup_mlflow.publish_model()
st.write('Model publié! Ce model va être évalué et mis en production si les scores conviennent')
