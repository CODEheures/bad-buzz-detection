from enum import Enum
from gensim import downloader

seed = 1234
model_enum = Enum('Model', ['SVM',
                            'Tensorflow_Keras_base_embedding',
                            'Tensorflow_Keras_base_LSTM_embedding',
                            'BERT_Transfert_learning'])
embedding_enum = Enum('Embedding', ['Trainable',
                                    'conceptnet-numberbatch-17-06-300',
                                    'fasttext-wiki-news-subwords-300',
                                    'glove-twitter-100',
                                    'glove-twitter-200',
                                    'glove-twitter-25',
                                    'glove-twitter-50',
                                    'glove-wiki-gigaword-100',
                                    'glove-wiki-gigaword-200',
                                    'glove-wiki-gigaword-300',
                                    'glove-wiki-gigaword-50',
                                    'word2vec-google-news-300'])
pretraitement_enum = Enum('Pretraitement', ['Lemmatization',
                                            'Stemming'])
tracking_uri = "https://mlflow.air-paradis.codeheures.fr"
s3_bucket = 'codeheures'
s3_uri = f"s3://{s3_bucket}/mlflow-air-paradis/"
model_name = "air-paradis"
alias = "production"
embedding_infos = downloader.info()


def get_format_model(model: model_enum):
    if (model == model_enum.SVM):
        return 'SVM (Support Vector Machine)'
    elif (model == model_enum.Tensorflow_Keras_base_embedding):
        return 'Tensorflow Keras base + embedding'
    elif (model == model_enum.Tensorflow_Keras_base_LSTM_embedding):
        return 'Tensorflow Keras base + LSTM + embedding'
    elif (model == model_enum.BERT_Transfert_learning):
        return 'BERT avec transfert learning'


def get_format_embedding(embedding: embedding_enum):
    if (embedding == embedding_enum.Trainable):
        return 'Nouvel Embedding entrainable'
    else:
        name = embedding.name
        description = embedding_infos['models'][embedding.name]['description']
        return f"{name} ({description})"
