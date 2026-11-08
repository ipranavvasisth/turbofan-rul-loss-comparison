import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load_fd001(data_dir=ROOT / "data", rul_cap=125):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train_FD001.txt", sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(data_dir / "test_FD001.txt", sep=r"\s+", header=None, names=COLS)
    test_rul = pd.read_csv(data_dir / "RUL_FD001.txt", header=None)[0].values
    train["RUL"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    train["RUL"] = train["RUL"].clip(upper=rul_cap)


    features_cols = [c for c in COLS[2:] if train[c].std() > 1e-6]

    return train, test, test_rul, features_cols


def scale_features(train, test, features_cols):
    scaler = MinMaxScaler()
    train, test = train.copy(), test.copy()
    train[features_cols] = scaler.fit_transform(train[features_cols])
    test[features_cols] = scaler.transform(test[features_cols])
    return train, test, scaler


def last_rows(test, features_cols):
    return test.groupby("unit").tail(1)[features_cols].values

def make_train_windows(df, features_cols, window=30):
    X, y, units = [], [], []
    for unit, g in df.groupby("unit"):
        feats = g[features_cols].values
        rul = g["RUL"].values
        for end in range(window, len(g) + 1):
            X.append(feats[end - window:end])
            y.append(rul[end - 1])
            units.append(unit)
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(units))


def make_test_windows(df, features_cols, window=30):
    X = []
    for unit, g in df.groupby("unit"):
        feats = g[features_cols].values
        if len(feats) < window:
            pad = np.repeat(feats[:1], window - len(feats), axis=0)
            feats = np.vstack([pad, feats])
        X.append(feats[-window:])
    return np.array(X, dtype=np.float32)

def prepare_data(window=30, n_val_units=20, split_seed=42):
    train, test, test_rul, feats = load_fd001()
    train, test, _ = scale_features(train, test, feats)
    X_all, y_all, units = make_train_windows(train, feats, window)
    X_test = make_test_windows(test, feats, window)
    y_test = np.minimum(test_rul, 125)

    rng = np.random.default_rng(split_seed)
    val_units = rng.choice(np.unique(units), size=n_val_units, replace=False)
    val_mask = np.isin(units, val_units)
    return dict(X_tr=X_all[~val_mask], y_tr=y_all[~val_mask],
                X_val=X_all[val_mask], y_val=y_all[val_mask],
                X_test=X_test, y_test=y_test, feats=feats)