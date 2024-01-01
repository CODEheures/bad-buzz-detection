import mlflow.pyfunc
from common import params, setup_mlflow
import pandas as pd


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
    setup_mlflow.init_mlflow()
    production_version = mlflow.pyfunc.load_model(f"models:/{params.model_name}@{params.alias}")
    response = production_version.predict(pd.DataFrame([tweet]))
    predict = response[0].item()
    predict_response = PredictResponse(tweet=tweet, predict=predict)
    return predict_response
