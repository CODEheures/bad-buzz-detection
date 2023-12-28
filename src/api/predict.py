import mlflow.pyfunc
from common import params, setup_mlflow
import pandas as pd

setup_mlflow.init_mlflow()
production_version = mlflow.pyfunc.load_model(f"models:/{params.model_name}@{params.alias}")


def run(tweet: str):
    predict = production_version.predict(pd.DataFrame([tweet]))
    return predict[0].item()
