import pandas as pd

def readin(filepath):
    df = pd.read_csv(filepath)
    df.drop(axis=1, inplace=True)


def main():
    filepath_weight = "data/a_w.csv"
    filepath_bias = "data/a_b.csv"
    weight = readin(filepath_weight)
    bias = readin(filepath_bias)




if __name__ == "__main__":
    main()