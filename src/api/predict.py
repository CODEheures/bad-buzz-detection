import mlflow.pyfunc
from common import params, setup_mlflow
import pandas as pd

setup_mlflow.init_mlflow()
production_version = mlflow.pyfunc.load_model(f"models:/{params.model_name}@{params.alias}")


class PredictResponse:
    def __init__(self, tweet: str, predict: int):
        self.tweet = tweet
        self.predict = predict
        self.human_predict = "negatif" if predict == 0 else "positif"
        self.icon_positif = ":+1:" if predict == 4 else ""
        self.icon_negatif = ":-1:" if predict == 0 else ""

    def asdict(self):
        return {
            "tweet": self.tweet,
            "predict": self.predict,
            "human_predict": self.human_predict,
        }


def run(tweet: str) -> PredictResponse:
    response = production_version.predict(pd.DataFrame([tweet]))
    predict = response[0].item()
    predict_response = PredictResponse(tweet=tweet, predict=predict)
    return predict_response
