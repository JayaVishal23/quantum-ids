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
