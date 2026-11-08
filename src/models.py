import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2,
                 out_scale=50.0, out_bias=80.0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)
        self.out_scale = out_scale   # raw output of ~1 means 50 cycles
        self.out_bias = out_bias     # start near the typical RUL

    def forward(self, x):
        out, _ = self.lstm(x)
        raw = self.head(out[:, -1]).squeeze(-1)
        return self.out_bias + self.out_scale * raw