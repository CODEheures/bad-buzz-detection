from streamlit import session_state as ss
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import TextVectorization
from keras.metrics import Precision
from common import text_processing
import streamlit as st


def svm(min_df: int,
        max_df: float,
        ngram_range: tuple,
        C: float,   # noqa= N803
        degree: int,
        kernel,
        gamma: float,
        seed=1234) -> Pipeline:
    """Create a personalized SVM pipeline

    Args:
        min_df (int): The word must be appear at least 'min_df' times to be keeped
        max_df (float): The word must be in the 'max_df'% of the dict to be keeped
        ngram_range (tuple): To create ngrams. Ex: (1,2) to create mono and bi-grams
        C (float): C param for SVM
        kernel ({'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}): The kernel to use on SVM
        gamma (float): Gamma param for SVM
        seed (int, optional): Seed to replay SVM. Defaults to 1234.

    Returns:
        Pipeline: The SVM pipeline
    """
    pipeline = Pipeline(
        steps=[
            ('vectorize_text', TfidfVectorizer(preprocessor=text_processing.preprocess_text,
                                               tokenizer=text_processing.tokenize,
                                               token_pattern=None,
                                               stop_words=text_processing.stop_words(),
                                               sublinear_tf=True,
                                               ngram_range=ngram_range,
                                               min_df=min_df,
                                               max_df=max_df)),
            ('modelize', SVC(C=C,
                             degree=degree,
                             kernel=kernel,
                             gamma=gamma,
                             random_state=seed,
                             ))
        ]
    )

    return pipeline


class DenseLayer:
    def __init__(self, units: int, dropout: float):
        self.units = units
        self.dropout = dropout


def keras_base(max_tokens=10000,
               max_sequence_length=50,
               embedding_dim=50,
               denses_layers: list[DenseLayer] = [DenseLayer(32, 0.5)],
               adapt_vectorize_layer=False) -> Sequential:
    """Define model keras

    Args:
        max_tokens (int, optional): max tokens to keep. Defaults to 10000.
        max_sequence_length (int, optional): max sequence length for truncate or padding. Defaults to 50.
        embedding_dim (int, optional): vector dim dor embedding. Defaults to 50.
        denses_layers (list[DenseLayer], optional): Denses layers on model. Defaults to [DenseLayer(32, 0.5)].
        adapt_vectorize_layer (bool, optional): To adapt vectorize layer (because it's a long task). Defaults is False
    Returns:
        Sequential: The keras model
    """
    vectorize_layer = TextVectorization(
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        max_tokens=max_tokens,
        output_mode='int',
        output_sequence_length=max_sequence_length)

    if adapt_vectorize_layer:
        with st.spinner("""Adaptation du layer de vectorisation sur les données du jeu d\'entrainement.
                         Cette opération peut être longue, patientez svp..."""):
            vectorize_layer.adapt(ss['X_train'])

    model = Sequential()
    model.add(Input(shape=(1,), dtype=tf.string))
    model.add(vectorize_layer)
    model.add(Embedding(max_tokens + 1, embedding_dim))
    model.add(Flatten())

    for dense in denses_layers:
        model.add(Dense(dense.units, activation='relu'))
        model.add(Dropout(dense.dropout))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[Precision()])
    return model


def keras_lstm() -> Pipeline:
    return None


def bert() -> Pipeline:
    return None


def format_model_summary_line(x: str, stringlist: list):
    """Format a model keras model summary line into a new string list

    Args:
        x (str): the line to format
        stringlist (list): the string list to append the formatted line
    """
    if len(x.strip()) > 0:
        stringlist.append(f'<div style="white-space:pre-wrap">{x}</div>')


def print_model_summary(model: Sequential):
    """Print a model summary in streamlit

    Args:
        model (Sequential): the model to print on streamlit
    """
    stringlist = []
    model.summary(line_length=110, print_fn=lambda x: format_model_summary_line(x, stringlist))
    short_model_summary = "\n".join(stringlist)
    st.write(short_model_summary, unsafe_allow_html=True)
