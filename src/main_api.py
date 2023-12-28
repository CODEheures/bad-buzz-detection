from fastapi import FastAPI
from api import predict
app = FastAPI()


@app.get("/predict")
def prediction(tweet: str):
    return predict.run(tweet=tweet)
