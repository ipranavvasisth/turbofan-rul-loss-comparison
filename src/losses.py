import torch
import torch.nn.functional as F


def msle(pred, y):
    mu = F.softplus(pred)                       # keeps predictions >= 0
    return torch.mean((torch.log1p(mu) - torch.log1p(y)) ** 2)


def mape(pred, y):
    return torch.mean(torch.abs(pred - y) / y.clamp(min=1.0))   # clamp: RUL can be 0


def tweedie(pred, y, power=1.5):
    mu = F.softplus(pred) + 1e-6
    return torch.mean(-y * mu ** (1 - power) / (1 - power)
                      + mu ** (2 - power) / (2 - power))


def nasa_loss(pred, y):
    """Asymmetric: late predictions (pred > true) cost more than early ones."""
    d = (pred - y).clamp(-50, 50)               # clamp keeps exp() from exploding
    return torch.mean(torch.where(d < 0, torch.exp(-d / 13) - 1,
                                  torch.exp(d / 10) - 1))