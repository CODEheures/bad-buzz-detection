from enum import Enum

seed = 1234
model_enum = Enum('Model', ['SVM', 'Deep Neural'])
tracking_uri = "https://mlflow.air-paradis.codeheures.fr"
s3_bucket = 'codeheures'
s3_uri = f"s3://{s3_bucket}/mlflow-air-paradis/"
model_name = "air-paradis"
alias = "production"
