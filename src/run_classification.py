import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data import ROOT, prepare_data
from src.models import LSTMRegressor
from src.train import predict, train_model

THRESH = 30
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 60

RESULTS = ROOT / "results"
PRED_DIR = RESULTS / "preds"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce_metric(y, logits):
    """Validation metric for epoch selection (lower is better)."""
    p = np.clip(sigmoid(logits), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def scores(y_true, risk, pred_label):
    return dict(accuracy=accuracy_score(y_true, pred_label),
                F1=f1_score(y_true, pred_label, zero_division=0),
                AUC=roc_auc_score(y_true, risk))


def main():
    d = prepare_data()
    y_tr_c = (d["y_tr"] <= THRESH).astype(np.float32)
    y_val_c = (d["y_val"] <= THRESH).astype(np.float32)
    y_test_c = (d["y_test"] <= THRESH).astype(int)
    print(f"positive rate: train {y_tr_c.mean():.2f}, test {y_test_c.mean():.2f} "
          f"({y_test_c.sum()} of {len(y_test_c)} test engines)")

    rows = []

    # 1) LSTM trained directly with binary cross-entropy (output = logit)
    for seed in SEEDS:
        model = LSTMRegressor(n_features=len(d["feats"]), out_scale=1.0, out_bias=0.0)
        model, _ = train_model(model, d["X_tr"], y_tr_c, d["X_val"], y_val_c,
                               loss_fn=nn.BCEWithLogitsLoss(), epochs=EPOCHS,
                               seed=seed, verbose=False, val_metric=bce_metric)
        logits = predict(model, d["X_test"])
        prob = sigmoid(logits)
        rows.append(dict(model="BCE classifier", seed=seed,
                         **scores(y_test_c, prob, (prob >= 0.5).astype(int))))
        print(rows[-1], flush=True)

    # 2) Regression models used as classifiers: predicted RUL <= THRESH
    for f in sorted(PRED_DIR.glob("*_seed*.npy")):
        loss, seed = f.stem.rsplit("_seed", 1)
        pred = np.load(f)
        rows.append(dict(model=f"regression: {loss}", seed=int(seed),
                         **scores(y_test_c, -pred, (pred <= THRESH).astype(int))))

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "classification.csv", index=False)
    summary = out.groupby("model")[["accuracy", "F1", "AUC"]].agg(["mean", "std"]).round(3)
    print(summary)


if __name__ == "__main__":
    main()