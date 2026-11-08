import pandas as pd

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load_fd001(data_dir="data", rul_cap=125):
    train = pd.read_csv(f"{data_dir}/train_FD001.txt", sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(f"{data_dir}/test_FD001.txt", sep=r"\s+", header=None, names=COLS)
    test_rul = pd.read_csv(f"{data_dir}/RUL_FD001.txt", header=None)[0].values

    train["RUL"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    train["RUL"] = train["RUL"].clip(upper=rul_cap)


    features_cols = [c for c in COLS[2:] if train[c].std() > 1e-6]

    return train, test, test_rul, features_cols