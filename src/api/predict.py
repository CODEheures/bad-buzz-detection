import mlflow.pyfunc
from mlflow.pyfunc import PyFuncModel
from common import params, setup_mlflow
import pandas as pd
import streamlit as st

production_version: PyFuncModel = None


class PredictResponse:
    """Give a synthetic predict response
    """
    def __init__(self, tweet: str, predict: int):
        self.tweet = tweet
        self.predict = 0 if predict == 0 else 1
        self.human_predict = "negatif" if predict == 0 else "positif"
        self.icon_positif = ":+1:" if predict > 0 else ""
        self.icon_negatif = ":-1:" if predict == 0 else ""

    def asdict(self):
        return {
            "tweet": self.tweet,
            "predict": self.predict,
            "human_predict": self.human_predict,
        }


def run(tweet: str) -> PredictResponse:
    """Get a sentiment for a tweet by mlflow registered model production inference

    Args:
        tweet (str): Tweet you want to test. Ex: "This is a good day"

    Returns:
        PredictResponse: The redict response objetc. Contain "tweet", "predict" and "human_predict" attributes
    """
    global production_version
    setup_mlflow.init_mlflow()
    load = False
    if production_version is None:
        load = True
    else:
        client = mlflow.MlflowClient()
        models = client.search_model_versions("name='air-paradis'")
        st.write(production_version.metadata.get_model_info().run_id)
        for model in models:
            if (model.run_id == production_version.metadata.get_model_info().run_id) and ('production' not in model.aliases):
                st.write('koko')
                load = True
    if load:
        production_version = mlflow.pyfunc.load_model(f"models:/{params.model_name}@{params.alias}")
    response = production_version.predict(pd.DataFrame([tweet]))
    predict = response[0].item()
    predict_response = PredictResponse(tweet=tweet, predict=predict)
    return predict_response
