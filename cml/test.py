# import plotly.express as px
from src.api import predict


def run():
    """Write some commentaries in Pull request.
    We can improve this function to give some decision elements to user who merge the pull request
    """
    good_tweet = 'this is good day'
    bad_tweet = 'this is bad day'
    prediction_good: predict.PredictResponse = predict.run(good_tweet)
    prediction_bad: predict.PredictResponse = predict.run(bad_tweet)

    with open("./cml/metrics.txt", "w") as outfile:
        outfile.writelines(f"Tweet: {good_tweet}\nresponse:{prediction_good.asdict()}")
        outfile.writelines(f"Tweet: {bad_tweet}\nresponse:{prediction_bad.asdict()}")


def main():
    run()


if __name__ == "__main__":
    main()
