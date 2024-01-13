import mlflow.pyfunc
from mlflow.pyfunc import PyFuncModel
from mlflow.entities.model_registry import ModelVersion
from common import params, setup_mlflow
import pandas as pd

production_version: PyFuncModel = None


class PredictResponse:
    """Give a synthetic predict response
    """
    def __init__(self, tweet: str, predict: int, model_version: ModelVersion):
        model_version = dict(model_version)
        model_version['aliases'] = [{"value": value} for value in model_version['aliases']]
        self.tweet = tweet
        self.predict = predict
        self.model_version = model_version
        self.human_predict = "negatif" if predict < 0.5 else "positif"
        self.icon_positif = ":+1:" if predict >= 0.5 else ""
        self.icon_negatif = ":-1:" if predict < 0.5 else ""

    def asdict(self):
        print(self.model_version)
        return {
            "tweet": self.tweet,
            "predict": self.predict,
            "human_predict": self.human_predict,
            "model": self.model_version
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
    client = mlflow.MlflowClient()
    load = False
    model_version: ModelVersion = None
    if production_version is None:
        load = True
    else:
        model_versions = client.search_model_versions(f"run_id='{production_version.metadata.get_model_info().run_id}'")
        if len(model_versions) != 1 or 'production' not in model_versions[0].aliases:
            load = True
        else:
            model_version = model_versions[0]

    if load:
        production_version = mlflow.pyfunc.load_model(f"models:/{params.model_name}@{params.alias}")
        model_version = client.search_model_versions(f"run_id='{production_version.metadata.get_model_info().run_id}'")[0]

    response = production_version.predict(pd.DataFrame([tweet]))
    if type(response) is pd.DataFrame:
        predict = response.loc[0, 'label']
    else:
        predict = response[0]
    predict_response = PredictResponse(tweet=tweet, predict=predict.item(), model_version=model_version)
    return predict_response
