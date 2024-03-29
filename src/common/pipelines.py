from streamlit import session_state as ss
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import TextVectorization
from keras.metrics import AUC
from common import text_processing
import streamlit as st
import numpy as np
from common import params
from gensim.models.keyedvectors import KeyedVectors
from gensim import downloader
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from datasets import Dataset

bert_metric = evaluate.load("roc_auc")


def svm(min_df: int,
        max_df: float,
        pretraitement_type: params.pretraitement_enum,
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
    if pretraitement_type == params.pretraitement_enum.Lemmatization:
        tokenizer = text_processing.tokenize_lemmatize
    else:
        tokenizer = text_processing.tokenize_stemming
    pipeline = Pipeline(
        steps=[
            ('vectorize_text', TfidfVectorizer(preprocessor=text_processing.preprocess_text,
                                               tokenizer=tokenizer,
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


class LstmLayer:
    def __init__(self, units: int, dropout: float):
        self.units = units
        self.dropout = dropout


def keras_base(max_tokens=10000,
               max_sequence_length=50,
               embedding=params.embedding_enum.Trainable,
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
    vectorize_layer = get_vectorized_layer(max_tokens=max_tokens,
                                           max_sequence_len=max_sequence_length,
                                           adapt=adapt_vectorize_layer)

    embedding_layer = get_embedding_layer(embedding=embedding,
                                          num_tokens=(max_tokens+2),
                                          vocabulary=vectorize_layer.get_vocabulary(),
                                          output_dim=embedding_dim)

    model = Sequential()
    model.add(Input(shape=(1,), dtype=tf.string))
    model.add(vectorize_layer)
    model.add(embedding_layer)
    model.add(Flatten())

    for dense in denses_layers:
        model.add(Dense(dense.units, activation='relu'))
        model.add(Dropout(dense.dropout))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
    return model


def keras_lstm(max_tokens=10000,
               max_sequence_length=50,
               embedding=params.embedding_enum.Trainable,
               embedding_dim=50,
               lstm_layers: list[LstmLayer] = [LstmLayer(32, 0.5)],
               adapt_vectorize_layer=False) -> Sequential:
    """Define model keras

    Args:
        max_tokens (int, optional): max tokens to keep. Defaults to 10000.
        max_sequence_length (int, optional): max sequence length for truncate or padding. Defaults to 50.
        embedding_dim (int, optional): vector dim dor embedding. Defaults to 50.
        lstm_layers (list[LstmLayer], optional): Lstm layers on model. Defaults to [LstmLayer(32, 0.5)].
        adapt_vectorize_layer (bool, optional): To adapt vectorize layer (because it's a long task). Defaults is False
    Returns:
        Sequential: The keras model
    """
    vectorize_layer = get_vectorized_layer(max_tokens=max_tokens,
                                           max_sequence_len=max_sequence_length,
                                           adapt=adapt_vectorize_layer)

    embedding_layer = get_embedding_layer(embedding=embedding,
                                          num_tokens=(max_tokens+2),
                                          vocabulary=vectorize_layer.get_vocabulary(),
                                          output_dim=embedding_dim)

    model = Sequential()
    model.add(Input(shape=(1,), dtype=tf.string))
    model.add(vectorize_layer)
    model.add(embedding_layer)

    for index, lstm_layer in enumerate(lstm_layers):
        return_sequences = True if (index + 1) < len(lstm_layers) else False
        model.add(LSTM(lstm_layer.units, return_sequences=return_sequences, dropout=lstm_layer.dropout))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
    return model


def bert(max_sequence_length=50, epochs=4, batch_size=16, adapt_vectorize_layer=False) -> Trainer:
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        label2id={0: 0, 1: 1},
        id2label={0: 0, 1: 1}
    )

    text_processing.tokenize_bert_max_length = max_sequence_length
    training_args = TrainingArguments(
        output_dir=".tmp/bert_trainer",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=0.25,
        save_steps=0.25,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=8,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        metric_for_best_model='eval_roc_auc',
        greater_is_better=True
    )

    train_dataset = None
    validation_dataset = None

    if adapt_vectorize_layer:
        train_df = Dataset.from_dict({"tweets": ss['X_train'], "labels": ss['y_train']})
        validation_df = Dataset.from_dict({"tweets": ss['X_validation'], "labels": ss['y_validation']})

        train_dataset = train_df.map(text_processing.tokenize_bert)
        train_dataset = train_dataset.remove_columns(["tweets"])

        validation_dataset = validation_df.map(text_processing.tokenize_bert)
        validation_dataset = validation_dataset.remove_columns(["tweets"])

    # Instantiate a `Trainer` instance that will be used to initiate a training run.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )

    return trainer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return bert_metric.compute(prediction_scores=predictions, references=labels)


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


def get_vectorized_layer(max_tokens: int, max_sequence_len: int, adapt=False) -> TextVectorization:
    """Construction du layer de vectorization

    Args:
        max_tokens (int): Nombre de tokens max à garder
        max_sequence_len (int): Dimenssion maximale de la séquence de sortie
        adapt (bool, optional): Adaptation au corpus (False pour une pré-construction rapide du model). Defaults to False.

    Returns:
        TextVectorization: Layer de sortie
    """
    vectorize_layer = TextVectorization(standardize="lower_and_strip_punctuation",
                                        split="whitespace",
                                        max_tokens=max_tokens,
                                        output_mode='int',
                                        output_sequence_length=max_sequence_len)

    if adapt:
        with st.spinner("""Adaptation du layer de vectorisation sur les données du jeu d\'entrainement.
                         Cette opération peut être longue, patientez svp..."""):
            vectorize_layer.adapt(ss['X_train'])

    return vectorize_layer


def get_embedding_matrix(num_tokens: int,
                         embedding_dim: int,
                         vocabulary: list[str],
                         vectors: KeyedVectors) -> np.ndarray[float]:
    """Conversion du vocabulaire en matrice de vecteurs pré-entrainés

    Args:
        num_tokens (int): Nombre de tokens de la matrice
        embedding_dim (int): Dimension de sortie de l'embedding
        vocabulary (list[str]): vocabulaire
        dict_pretrained (dict[str, np.ndarray[any]]): Dictionnaire des mots associés aux vecteurs pré-entrainés

    Returns:
        NDArray[float]: Matrice des vecteurs de l'embedding
    """
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    hits = 0
    misses = 0
    for index, word in enumerate(vocabulary):
        if vectors.has_index_for(word):
            embedding_matrix[index] = vectors.get_vector(word)
            hits += 1
        else:
            misses += 1
    st.write(f"{hits} mots convertis ({misses} non reconnus)")

    return embedding_matrix


def get_embedding_layer(embedding: params.embedding_enum,
                        num_tokens: int,
                        vocabulary: list[str], output_dim: int) -> Embedding:
    """Construction du layer Embedding

    Args:
        embedding (params.embedding_enum): type d'embedding (Trainable, GloVe, ...)
        num_tokens (int): Nombre de tokens du vocabulaire
        vocabulary (list[str]): vocabulaire (peut etre vide pour une pré-construction rapide du modèle)
        output_dim (int): Dimension de l'embedding

    Returns:
        Embedding (Embedding): Layer d'embedding
    """
    if embedding == params.embedding_enum.Trainable:
        layer = Embedding(input_dim=num_tokens, output_dim=output_dim, trainable=True)
    else:
        layer = Embedding(input_dim=num_tokens, output_dim=output_dim, trainable=False)
        layer.build((1,))
        if len(vocabulary) > 2:
            with st.spinner(f"Téléchargement de l'embedding pré-entrainé {embedding.name}"):
                vectors: KeyedVectors = downloader.load(embedding.name)

            with st.spinner("Adapdation de l'embedding au vocabulaire"):
                embedding_matrix = get_embedding_matrix(num_tokens=num_tokens,
                                                        embedding_dim=output_dim,
                                                        vocabulary=vocabulary,
                                                        vectors=vectors)
            layer.set_weights([embedding_matrix])

    return layer
