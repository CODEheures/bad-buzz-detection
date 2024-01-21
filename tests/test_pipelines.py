from src.common import pipelines
from src.common import params
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from transformers import Trainer


def test_pipeline_smv():
    svm = pipelines.svm(min_df=1,
                        max_df=5,
                        pretraitement_type=params.pretraitement_enum.Lemmatization,
                        ngram_range=(1, 1),
                        C=1,   # noqa= N803
                        degree=3,
                        kernel='rbf',
                        gamma='auto')

    assert type(svm) is Pipeline


def test_pipeline_keras_base():
    keras_base = pipelines.keras_base(max_tokens=10000,
                                      max_sequence_length=50,
                                      embedding=params.embedding_enum.Trainable,
                                      embedding_dim=50,
                                      denses_layers=[pipelines.DenseLayer(32, 0.5),
                                                     pipelines.DenseLayer(32, 0.5)],
                                      adapt_vectorize_layer=False)

    assert type(keras_base) is Sequential


def test_pipeline_lstm():
    keras_lstm = pipelines.keras_lstm(max_tokens=10000,
                                      max_sequence_length=50,
                                      embedding=params.embedding_enum['glove-twitter-100'],
                                      lstm_layers=[pipelines.LstmLayer(32, 0.5),
                                                   pipelines.LstmLayer(32, 0.5)],
                                      adapt_vectorize_layer=False)

    assert type(keras_lstm) is Sequential


def test_pipeline_bert():
    bert = pipelines.bert(max_sequence_length=50, epochs=4, batch_size=16, adapt_vectorize_layer=False)

    assert type(bert) is Trainer
