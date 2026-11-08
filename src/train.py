import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(model, X, batch_size=1024):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size]).to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def nasa_score(y_true, y_pred):
    """C-MAPSS asymmetric score: late predictions (pred > true) are penalised more."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def train_model(model, X_tr, y_tr, X_val, y_val, loss_fn,
                epochs=30, lr=1e-3, batch_size=256, seed=42, verbose=True,
                val_metric=rmse):
    set_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(torch.as_tensor(X_tr), torch.as_tensor(y_tr)),
                        batch_size=batch_size, shuffle=True)

    best_val, best_state, history = float("inf"), None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        val = val_metric(y_val, predict(model, X_val))
        history.append((total / len(X_tr), val))
        if val < best_val:
            best_val, best_state = val, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"epoch {epoch:2d}  train loss {total / len(X_tr):8.3f}  val RMSE {val:6.2f}")

    model.load_state_dict(best_state)   # keep the best epoch
    return model, history