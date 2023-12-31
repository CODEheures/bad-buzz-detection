from enum import Enum

seed = 1234
model_enum = Enum('Model', ['SVM',
                            'Tensorflow_Keras_base_embedding',
                            'Tensorflow_Keras_base_LSTM_embedding',
                            'BERT_Transfert_learning'])
tracking_uri = "https://mlflow.air-paradis.codeheures.fr"
s3_bucket = 'codeheures'
s3_uri = f"s3://{s3_bucket}/mlflow-air-paradis/"
model_name = "air-paradis"
alias = "production"


def get_format_model(model: model_enum):
    if (model == model_enum.SVM):
        return 'SVM (Support Vector Machine)'
    elif (model == model_enum.Tensorflow_Keras_base_embedding):
        return 'Tensorflow Keras base + embedding'
    elif (model == model_enum.Tensorflow_Keras_base_LSTM_embedding):
        return 'Tensorflow Keras base + LSTM + embedding'
    elif (model == model_enum.BERT_Transfert_learning):
        return 'BERT avec transfert learning'