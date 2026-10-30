"""
Shared evaluation: the same metrics and plots for every model.

If two experiments are measured by different code, their numbers can't be
compared. That's why everything goes through this file.
"""

import matplotlib
matplotlib.use("Agg")  # save figures to files, don't open windows
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"]


def evaluate(y_true, attack_probability, threshold=0.5):
    """All the numbers we report for one model on one dataset."""
    y_pred = (attack_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),  # share of attacks caught
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, attack_probability),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,  # normal traffic wrongly flagged
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ------------------------------------------------------------------
# Choosing the decision threshold
# ------------------------------------------------------------------
#
# A model outputs a probability; something has to turn that into a yes or no.
# Using 0.5 is a habit, not a decision, and on this dataset it is a bad one:
# the training file is 68% attacks while the test file is 55%, so a model
# trained to 0.5 flags far too much normal traffic.
#
# The threshold is chosen on VALIDATION data only. In federated training no
# client may send its validation rows anywhere, so each client reports only
# four counts per candidate threshold and the server adds them up. Integers,
# never records.

DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
MIN_RECALL = 0.95          # a missed attack costs more than a false alarm
THRESHOLD_RULES = ["half", "min_fpr_at_recall", "youden", "max_f1"]


def confusion_counts_at_thresholds(y_true, attack_probability, grid=THRESHOLD_GRID):
    """tp, fp, fn, tn at every candidate threshold. The only thing a client sends."""
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(attack_probability)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)

    flagged = probability[:, None] >= grid[None, :]
    tp = (flagged & (y_true[:, None] == 1)).sum(axis=0)
    fp = (flagged & (y_true[:, None] == 0)).sum(axis=0)
    return pd.DataFrame({
        "threshold": grid,
        "tp": tp, "fp": fp,
        "fn": positives - tp,
        "tn": negatives - fp,
    })


def add_counts(tables):
    """Sum the per-threshold counts reported by several clients."""
    total = tables[0].copy()
    for table in tables[1:]:
        for column in ("tp", "fp", "fn", "tn"):
            total[column] = total[column].to_numpy() + table[column].to_numpy()
    return total


def choose_threshold(counts, rule="min_fpr_at_recall", min_recall=MIN_RECALL):
    """Pick one threshold from summed validation counts.

    half               keep 0.5, the untuned baseline
    min_fpr_at_recall  fewest false alarms while still catching min_recall of attacks
    youden             best recall minus false positive rate (the usual ROC point)
    max_f1             best balance of precision and recall
    """
    if rule == "half":
        return DEFAULT_THRESHOLD
    if rule not in THRESHOLD_RULES:
        raise ValueError(f"Unknown rule '{rule}'. Choose from {THRESHOLD_RULES}.")

    table = counts.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        table["recall"] = table["tp"] / (table["tp"] + table["fn"])
        table["fpr"] = table["fp"] / (table["fp"] + table["tn"])
        precision = table["tp"] / (table["tp"] + table["fp"])
        table["f1"] = 2 * precision * table["recall"] / (precision + table["recall"])
    table = table.fillna(0.0)

    if rule == "youden":
        return float(table.loc[(table["recall"] - table["fpr"]).idxmax(), "threshold"])
    if rule == "max_f1":
        return float(table.loc[table["f1"].idxmax(), "threshold"])

    # min_fpr_at_recall: among thresholds that keep enough recall, take the
    # lowest false positive rate. If none qualifies, keep the most sensitive.
    eligible = table[table["recall"] >= min_recall]
    if eligible.empty:
        return float(table.loc[table["recall"].idxmax(), "threshold"])
    return float(eligible.loc[eligible["fpr"].idxmin(), "threshold"])


def recall_per_attack_family(y_true, attack_probability, attack_cat, threshold=0.5):
    """For each attack family: what share of its rows did the model catch?
    For Normal it shows the share correctly left alone."""
    y_pred = (attack_probability >= threshold).astype(int)
    frame = pd.DataFrame({"family": attack_cat, "correct": y_pred == y_true})
    return frame.groupby("family")["correct"].mean().sort_values()


def summarize_seeds(rows):
    """rows: one dict per seed. Returns a table with mean and std of each metric."""
    table = pd.DataFrame(rows)
    numeric = table.select_dtypes("number").drop(columns=["seed"], errors="ignore")
    return pd.DataFrame({"mean": numeric.mean(), "std": numeric.std(ddof=1)})


def format_mean_std(summary, names=METRIC_NAMES):
    # "+/-" instead of the ± sign, because the Windows console can't print ±.
    return {name: f"{summary.loc[name, 'mean']:.4f} +/- {summary.loc[name, 'std']:.4f}" for name in names}


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def plot_confusion_matrix(y_true, attack_probability, title, path, threshold=0.5):
    y_pred = (attack_probability >= threshold).astype(int)
    display = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=[0, 1]),
                                     display_labels=["Normal", "Attack"])
    display.plot(values_format="d", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_roc_curve(y_true, attack_probability, title, path):
    fpr, tpr, _ = roc_curve(y_true, attack_probability)
    auc = roc_auc_score(y_true, attack_probability)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (recall)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_curves(history, x_column, y_columns, title, ylabel, path):
    plt.figure()
    for column in y_columns:
        plt.plot(history[x_column], history[column], marker="o", markersize=3, label=column)
    plt.xlabel(x_column)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
