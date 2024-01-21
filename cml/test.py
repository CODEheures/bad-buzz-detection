# import plotly.express as px

def run():
    """Write some commentaries in Pull request.
    We can improve this function to give some decision elements to user who merge the pull request
    """
    with open("./cml/metrics.txt", "w") as outfile:
        outfile.writelines("Train score: 0.0\n")
        outfile.writelines("Test score: 42\n")


def main():
    run()


if __name__ == "__main__":
    main()
