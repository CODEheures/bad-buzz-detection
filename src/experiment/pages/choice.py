import streamlit as st
from st_pages import add_page_title
from streamlit import session_state as ss
from common import pipelines, params
from streamlit_extras.switch_page_button import switch_page
from experiment import pages_management
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from transformers import Trainer

add_page_title()

st.header("Cette page permet de choisir et de paramétrer un model", divider='rainbow')

st.markdown("### 2. Choix du Model et Paramètres")
seleted_model = st.selectbox("Choix du model",
                             [member for member in params.model_enum],
                             format_func=params.get_format_model,
                             index=None,
                             placeholder="Selectionnez un model...")
model = None
if (seleted_model == params.model_enum.SVM):  # noqa: C901
    min_df = st.slider("Ignorer les mots qui apparaisent moins de n fois", min_value=1, max_value=10, value=1)
    max_df = st.slider("ignorer les x% des mots les plus fréqents", min_value=0, max_value=10, value=0)
    n_gram_range = st.slider('Rang des N_grams', min_value=1, max_value=4, value=(1, 1))
    pretraitement_type = st.selectbox("Choix du prétraitement",
                                      [member for member in params.pretraitement_enum],
                                      placeholder="Selectionnez un prétraitement...")
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
                          pretraitement_type=pretraitement_type,
                          ngram_range=n_gram_range,
                          C=C,   # noqa= N803
                          degree=degree,
                          kernel=kernel,
                          gamma=gamma,
                          seed=params.seed)
elif (seleted_model == params.model_enum.Tensorflow_Keras_base_embedding) \
     or (seleted_model == params.model_enum.Tensorflow_Keras_base_LSTM_embedding):
    max_tokens = st.slider("Nombre de tokens maxi à garder", min_value=1000, max_value=50000, value=10000, step=1000)
    max_sequence_length = st.slider("Nombre de tokens maxi dans un tweet", min_value=10, max_value=100, value=50, step=1)
    embedding = st.selectbox("Choix de l'embedding",
                             [member for member in params.embedding_enum],
                             format_func=params.get_format_embedding,
                             placeholder="Selectionnez un embedding...")
    if embedding == params.embedding_enum.Trainable:
        embedding_dim = st.slider("Nombre de dimensions de l'embedding", min_value=10, max_value=300, value=50, step=1)
    else:
        dim = params.embedding_infos['models'][embedding.name]['parameters']['dimension']
        embedding_dim = st.slider("Nombre de dimensions de l'embedding", min_value=10, max_value=300, value=dim, disabled=True)

    layers_count = st.slider("Nombre de couches denses", min_value=1, max_value=3, value=2, step=1)
    if seleted_model == params.model_enum.Tensorflow_Keras_base_embedding:
        layers: pipelines.DenseLayer = []
    elif seleted_model == params.model_enum.Tensorflow_Keras_base_LSTM_embedding:
        layers: pipelines.LstmLayer = []
    for i in range(layers_count):
        st.markdown(f'#### Couche {i}')
        units = st.slider("Nombre de neuronnes", min_value=1, max_value=128, value=16, step=1, key=f'dense_units_{i}')
        dropout = st.slider("Dropout", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key=f'dense_dropout_{i}')
        if seleted_model == params.model_enum.Tensorflow_Keras_base_embedding:
            layers.append(pipelines.DenseLayer(units=units, dropout=dropout))
        elif seleted_model == params.model_enum.Tensorflow_Keras_base_LSTM_embedding:
            layers.append(pipelines.LstmLayer(units=units, dropout=dropout))

    ss['batch_size'] = st.slider("Taille des batchs", min_value=64, max_value=1024, value=128, step=64)
    ss['epochs'] = st.number_input("Nombre d'epochs", min_value=1, max_value=500, value=2)
    ss['embedding'] = f"{embedding.name} + (dim={embedding_dim})"

    if seleted_model == params.model_enum.Tensorflow_Keras_base_embedding:
        model = pipelines.keras_base(max_tokens=max_tokens,
                                     max_sequence_length=max_sequence_length,
                                     embedding=embedding,
                                     embedding_dim=embedding_dim,
                                     denses_layers=layers)
    elif seleted_model == params.model_enum.Tensorflow_Keras_base_LSTM_embedding:
        model = pipelines.keras_lstm(max_tokens=max_tokens,
                                     max_sequence_length=max_sequence_length,
                                     embedding=embedding,
                                     embedding_dim=embedding_dim,
                                     lstm_layers=layers)

    pipelines.print_model_summary(model)
elif (seleted_model == params.model_enum.BERT_Transfert_learning):
    max_sequence_length = st.slider("Nombre de tokens maxi dans un tweet", min_value=10, max_value=100, value=50, step=1)
    batch_size = st.slider("Taille des batchs", min_value=16, max_value=1024, value=16, step=16)
    epochs = st.number_input("Nombre d'epochs", min_value=1, max_value=500, value=2)
    model = pipelines.bert(max_sequence_length=max_sequence_length,
                           epochs=epochs,
                           batch_size=batch_size)


if model is not None and \
   (type(model) is Pipeline) or \
   (type(model) is Sequential) or \
   (type(model) is Trainer):
    ss['selected_model'] = seleted_model
    ss['choice_ok'] = True
    pages_management.update_pages()
    if st.button("Entrainer ce model"):
        if (seleted_model == params.model_enum.SVM):
            ss['model'] = model
        elif (seleted_model == params.model_enum.Tensorflow_Keras_base_embedding):
            ss['model'] = pipelines.keras_base(max_tokens=max_tokens,
                                               max_sequence_length=max_sequence_length,
                                               embedding=embedding,
                                               embedding_dim=embedding_dim,
                                               denses_layers=layers,
                                               adapt_vectorize_layer=True)
        elif (seleted_model == params.model_enum.Tensorflow_Keras_base_LSTM_embedding):
            ss['model'] = pipelines.keras_lstm(max_tokens=max_tokens,
                                               max_sequence_length=max_sequence_length,
                                               embedding=embedding,
                                               embedding_dim=embedding_dim,
                                               lstm_layers=layers,
                                               adapt_vectorize_layer=True)
        elif (seleted_model == params.model_enum.BERT_Transfert_learning):
            ss['model'] = pipelines.bert(max_sequence_length=max_sequence_length,
                                         epochs=epochs,
                                         batch_size=batch_size,
                                         adapt_vectorize_layer=True)

        switch_page("Entrainement model")
