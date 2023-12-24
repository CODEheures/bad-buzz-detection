from enum import Enum

seed = 1234
model_enum = Enum('Model', ['SVM', 'Deep Neural'])
tracking_uri = "https://mlflow.air-paradis.codeheures.fr"
s3_uri = "s3://codeheures/mlflow-air-paradis/"
model_name = "air-paradis"
alias = "production"
