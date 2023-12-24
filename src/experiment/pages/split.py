import streamlit as st
from st_pages import add_page_title
from streamlit import session_state as ss
from common import splitter
import plotly.graph_objects as go
from streamlit_extras.switch_page_button import switch_page
from experiment import pages_management

add_page_title()
seed = 1234
st.header("Cette page permet d'entrainer un model et d'obtenir les scores et graphiques associés", divider='rainbow')


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

ss['split_ok'] = True
pages_management.update_pages()
if st.button("Choisir et entrainer un model"):
    switch_page("Choix et paramétrage model")
