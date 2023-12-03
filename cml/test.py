

def run():
    with open("./cml/metrics.txt", "w") as outfile:
        outfile.writelines("Train score: 0\n")
        outfile.writelines("Test score: 0\n")


def main():
    run()


if __name__ == "__main__":
    main()
