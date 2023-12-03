# import plotly.express as px


def run():
    with open("./cml/metrics.txt", "w") as outfile:
        outfile.writelines("Train score: 0.0\n")
        outfile.writelines("Test score: 0.0\n")

    # df = px.data.iris()
    # fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
    # fig.write_image("cml/metrics.jpg")


def main():
    run()


if __name__ == "__main__":
    main()
