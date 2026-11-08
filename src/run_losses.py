import time

import numpy as np
import pandas as pd
import torch.nn as nn

from src.data import ROOT, prepare_data
from src.losses import mape, msle, nasa_loss, tweedie
from src.models import LSTMRegressor
from src.train import nasa_score, predict, rmse, train_model

LOSSES = {"MSE": nn.MSELoss(), "MAE": nn.L1Loss(), "MSLE": msle,
          "MAPE": mape, "Tweedie": tweedie, "NASA": nasa_loss}
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 60

RESULTS = ROOT / "results"
PRED_DIR = RESULTS / "preds"
RUNS_CSV = RESULTS / "runs.csv"


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    d = prepare_data()
    y_test = d["y_test"]

    rows = pd.read_csv(RUNS_CSV).to_dict("records") if RUNS_CSV.exists() else []
    finished = {(r["loss"], int(r["seed"])) for r in rows}
    total = len(LOSSES) * len(SEEDS)

    for name, loss_fn in LOSSES.items():
        for seed in SEEDS:
            if (name, seed) in finished:
                continue
            t0 = time.time()
            model = LSTMRegressor(n_features=len(d["feats"]),
                                  out_bias=float(d["y_tr"].mean()))
            model, _ = train_model(model, d["X_tr"], d["y_tr"], d["X_val"], d["y_val"],
                                   loss_fn=loss_fn, epochs=EPOCHS, seed=seed,
                                   verbose=False)
            pred = np.clip(predict(model, d["X_test"]), 0, 125)   # same for every loss
            np.save(PRED_DIR / f"{name}_seed{seed}.npy", pred)

            row = dict(loss=name, seed=seed,
                       RMSE=rmse(y_test, pred),
                       MAE=float(np.mean(np.abs(y_test - pred))),
                       NASA=nasa_score(y_test, pred),
                       bias=float(np.mean(pred - y_test)))   # mean signed error
            rows.append(row)
            pd.DataFrame(rows).to_csv(RUNS_CSV, index=False)
            print(f"[{len(rows)}/{total}] {row}  ({time.time() - t0:.0f}s)", flush=True)

    print("done")


if __name__ == "__main__":
    main()