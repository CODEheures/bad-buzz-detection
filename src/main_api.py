from fastapi import FastAPI
from api import predict
app = FastAPI()


@app.get("/predict")
def prediction(tweet: str):
    """Api route to get a sentiment response for a tweet

    Args:
        tweet (str): The tweet to predit

    Returns:
        dict[str, Any]: A json response with tweet, predict, human_predict keys
    """
    return predict.run(tweet=tweet).asdict()
