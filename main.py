import pandas as pd

def main():
    print("This is the main file for our project")

    # a potential data set
    # marijuana arrests in columbia
    df = pd.read_csv("./data/Marijuana_Arrests.csv")
    print(df)


if __name__ == "__main__":
    main()
