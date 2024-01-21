from src.api import predict


def test_function():
    assert True
    prediction: predict.PredictResponse = predict.run('this is good day')
    assert type(prediction) is predict.PredictResponse
    assert prediction.predict > 0
