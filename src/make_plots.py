import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import ROOT, prepare_data

RESULTS = ROOT / "results"
PRED_DIR = RESULTS / "preds"
FIG_DIR = RESULTS / "figures"
ORDER = ["MSE", "MAE", "MSLE", "MAPE", "Tweedie", "NASA"]


def load_preds(loss, seeds):
    return [np.load(PRED_DIR / f"{loss}_seed{s}.npy") for s in seeds]


def summary_table(runs):
    n_seeds = runs.groupby("loss")["seed"].nunique().min()
    lines = [f"Mean ± std over {n_seeds} seeds (test set: last window of each of the 100 engines, RUL clipped at 125).",
             "",
             "| Loss | RMSE | MAE | NASA score | Mean signed error |",
             "|---|---|---|---|---|"]
    for loss in ORDER:
        g = runs[runs["loss"] == loss]
        cell = lambda c, g=g: f"{g[c].mean():.2f} ± {g[c].std():.2f}"
        lines.append(f"| {loss} | {cell('RMSE')} | {cell('MAE')} | "
                     f"{g['NASA'].mean():.0f} ± {g['NASA'].std():.0f} | {cell('bias')} |")
    return "\n".join(lines)


def plot_metrics(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in zip(axes, ["RMSE", "NASA"],
                              ["Test RMSE (lower is better)", "NASA score (lower is better)"]):
        for i, loss in enumerate(ORDER):
            vals = runs.loc[runs["loss"] == loss, col].values
            ax.scatter(np.full(len(vals), i), vals, alpha=0.5, s=18)
            ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1), fmt="_", color="k",
                        capsize=6, markersize=16)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Dots: individual seeds. Bar: mean ± std")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metrics_by_loss.png", dpi=150)
    plt.close(fig)


def plot_pred_vs_true(runs, y_test):
    seed = int(runs["seed"].min())
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for ax, loss in zip(axes.ravel(), ORDER):
        pred = load_preds(loss, [seed])[0]
        ax.scatter(y_test, pred, s=14, alpha=0.7)
        ax.plot([0, 125], [0, 125], "k--", lw=1)
        ax.set_title(f"{loss} (seed {seed})")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("true RUL (clipped at 125)")
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted RUL")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pred_vs_true.png", dpi=150)
    plt.close(fig)


def plot_signed_error(runs, y_test):
    bins = np.linspace(-60, 60, 41)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for ax, loss in zip(axes.ravel(), ORDER):
        seeds = runs.loc[runs["loss"] == loss, "seed"]
        err = np.concatenate([p - y_test for p in load_preds(loss, seeds)])
        ax.hist(err, bins=bins, alpha=0.8)
        ax.axvline(0, color="k", lw=1)
        ax.set_title(f"{loss}: mean {err.mean():+.1f}")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("prediction - truth (cycles)\n<0 early, >0 late")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "signed_error_hist.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(RESULTS / "runs.csv")
    y_test = prepare_data()["y_test"]

    table = summary_table(runs)
    (RESULTS / "summary_table.md").write_text(table + "\n")
    print(table)

    plot_metrics(runs)
    plot_pred_vs_true(runs, y_test)
    plot_signed_error(runs, y_test)
    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()