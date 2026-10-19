"""
All Baseline results in one place, for the slides and the report.

Reads the saved results of E1 (centralised MLPs), E2 (federated MLP) and
E3 (centralised QNN), and produces:
    results/baseline_models/baseline_results.csv   numbers (mean and std)
    results/baseline_models/baseline_results.md    the same as a readable table
    results/baseline_models/baseline_results.png   one figure comparing the models

Every number comes straight from the result files, so the slides can never
disagree with what the code actually produced.

Run from the project root (after the experiments):
    python -m experiments.baseline_models_summary
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import config

REVIEW_DIR = config.RESULTS_DIR / "baseline_models"

# (model, setting, results folder, file prefix, column holding the training time)
EXPERIMENTS = [
    ("MLP", "centralised", "e1_centralized", "mlp_", "training_seconds"),
    ("Tiny MLP", "centralised", "e1_centralized", "tiny_mlp_", "training_seconds"),
    ("QNN", "centralised", "e3_qnn_centralized", "qnn_", "training_seconds"),
    ("MLP", "FedAvg, 3 clients", "fl_mlp_fedavg", "", "total_seconds"),
]

METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("roc_auc", "ROC-AUC"),
    ("false_positive_rate", "False-positive rate"),
]

# Colours and ink from the chart palette (light mode).
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"


def load_results():
    rows = []
    for model, setting, folder, prefix, time_column in EXPERIMENTS:
        summary_file = REVIEW_DIR / folder / f"{prefix}summary.csv"
        per_seed_file = REVIEW_DIR / folder / f"{prefix}per_seed.csv"
        if not summary_file.exists():
            print(f"Skipping {model} ({setting}): {summary_file} not found yet")
            continue

        summary = pd.read_csv(summary_file, index_col=0)
        per_seed = pd.read_csv(per_seed_file)

        row = {
            "model": model,
            "setting": setting,
            "parameters": int(per_seed["parameters"].iloc[0]),
            "seeds": len(per_seed),
            "time_seconds": summary.loc[time_column, "mean"],
        }
        for metric, _ in METRICS:
            row[f"{metric}_mean"] = summary.loc[metric, "mean"]
            row[f"{metric}_std"] = summary.loc[metric, "std"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(table, path):
    def cell(row, metric):
        mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
        if metric == "roc_auc":
            return f"{mean:.3f} ± {std:.3f}"
        return f"{mean:.1%} ± {std:.1%}"

    headers = ["Model", "Setting", "Params"] + [label for _, label in METRICS] + ["Time (s)"]
    lines = [
        "# Baseline results",
        "",
        "Test sample of 5,000 rows. Mean ± standard deviation over "
        f"{int(table['seeds'].max())} seeds. Same 8 features and same rows for every model.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * 3 + "---:|" * (len(METRICS) + 1),
    ]
    for _, row in table.iterrows():
        cells = [row["model"], row["setting"], str(row["parameters"])]
        cells += [cell(row, metric) for metric, _ in METRICS]
        cells.append(f"{row['time_seconds']:.0f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "Notes:",
        "- The tiny MLP and the QNN both have 81 trainable parameters, so they compare like for like.",
        "- The federated model's best round is chosen by the clients' validation loss, not the test set.",
        "- The high false-positive rate affects every model the same way. It comes from a distribution",
        "  shift between the official UNSW-NB15 training and test files (see notes_false_positive_rate.md).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_figure(table, path):
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    panels = [("accuracy", "Accuracy"), ("recall", "Recall (attacks caught)"),
              ("false_positive_rate", "False-positive rate (lower is better)"), ("roc_auc", "ROC-AUC")]

    labels = [f"{row.model}\n{row.setting}" for row in table.itertuples()]
    positions = list(range(len(table)))[::-1]  # first model at the top

    figure, axes = plt.subplots(2, 2, figsize=(11, 6.5), facecolor=SURFACE)
    for ax, (metric, title) in zip(axes.ravel(), panels):
        means = table[f"{metric}_mean"].to_numpy()
        stds = table[f"{metric}_std"].fillna(0).to_numpy()

        # Keep every axis at least 10 points wide, so small differences
        # don't look bigger than they are, and never go past 100%.
        low, high = (means - stds).min(), (means + stds).max()
        span = max(high - low, 0.10)
        middle = (low + high) / 2
        left = max(middle - span * 0.65, 0.0)
        right = min(middle + span * 0.65, 1.0)
        ax.set_xlim(left, right)

        ax.hlines(positions, means - stds, means + stds, color=SERIES, linewidth=2)
        ax.plot(means, positions, "o", markersize=9, color=SERIES,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)

        gap = (right - left) * 0.03
        label_width = (right - left) * 0.12
        for y, mean, std in zip(positions, means, stds):
            text = f"{mean:.3f}" if metric == "roc_auc" else f"{mean:.1%}"
            if mean + std + gap + label_width <= right:
                ax.text(mean + std + gap, y, text, va="center", ha="left", fontsize=9, color=INK)
            else:  # no room on the right: put the label on the left of the dot
                ax.text(mean - std - gap, y, text, va="center", ha="right", fontsize=9, color=INK)

        if metric != "roc_auc":
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

        ax.set_title(title, loc="left", fontsize=11, color=INK_SECONDARY)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=9, color=INK)
        ax.set_ylim(-0.6, len(table) - 0.4)
        ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8, length=0)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)

    figure.suptitle("Baseline: detection results on the test sample (dot = mean, line = ± std over seeds)",
                    x=0.01, ha="left", fontsize=12, color=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main():
    table = load_results()
    if table.empty:
        raise SystemExit("No results found. Run the Baseline experiments first.")

    table.to_csv(REVIEW_DIR / "baseline_results.csv", index=False)
    write_markdown(table, REVIEW_DIR / "baseline_results.md")
    draw_figure(table, REVIEW_DIR / "baseline_results.png")

    printable = table[["model", "setting", "parameters"]].copy()
    for metric, label in METRICS:
        printable[label] = [f"{m:.3f} +/- {s:.3f}" for m, s in zip(table[f"{metric}_mean"], table[f"{metric}_std"])]
    print(printable.to_string(index=False))
    print(f"\nSaved baseline_results.csv, .md and .png in {REVIEW_DIR}")


if __name__ == "__main__":
    main()
