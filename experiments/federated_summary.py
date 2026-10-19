"""
Federated results: does the quantum model work inside federated learning?

Reads the saved results of:
    E1  centralised MLP / tiny MLP      (Review 1)
    E2  classical FL (MLP + FedAvg)     (Review 1)
    E3  centralised QNN                 (Review 1)
    E4  QML-FL (QNN + FedAvg)           (Federated)

and produces, in results/federated_models/:
    federated_results.csv / .md     the comparison table
    accuracy_per_round.png  classical FL vs QML-FL, round by round
    recall_per_family.png   which attack families each model catches

Run from the project root (after the experiments):
    python -m experiments.federated_models_summary
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import config

REVIEW1 = config.RESULTS_DIR / "baseline_models"
REVIEW2 = config.RESULTS_DIR / "federated_models"

# (label, setting, folder, file prefix, time column)
EXPERIMENTS = [
    ("MLP", "centralised", REVIEW1 / "e1_centralized", "mlp_", "training_seconds"),
    ("Tiny MLP", "centralised", REVIEW1 / "e1_centralized", "tiny_mlp_", "training_seconds"),
    ("QNN", "centralised", REVIEW1 / "e3_qnn_centralized", "qnn_", "training_seconds"),
    ("MLP", "FedAvg, 60 rounds", REVIEW1 / "fl_mlp_fedavg", "", "total_seconds"),
    ("MLP", "FedAvg, 30 rounds", REVIEW2 / "fl_mlp_fedavg", "", "total_seconds"),
    ("QNN", "FedAvg, 30 rounds", REVIEW2 / "fl_qnn_fedavg", "", "total_seconds"),
]

# The two federated runs compared round by round (same budget: 30 rounds, 1 local epoch).
CURVES = [
    ("Classical FL (MLP)", REVIEW2 / "fl_mlp_fedavg", "#2a78d6"),
    ("QML-FL (QNN)", REVIEW2 / "fl_qnn_fedavg", "#eb6834"),
]

METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("roc_auc", "ROC-AUC"),
    ("false_positive_rate", "False-positive rate"),
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"


def load_table():
    rows = []
    for label, setting, folder, prefix, time_column in EXPERIMENTS:
        summary_file = folder / f"{prefix}summary.csv"
        per_seed_file = folder / f"{prefix}per_seed.csv"
        if not summary_file.exists():
            print(f"Skipping {label} ({setting}): no results yet in {folder}")
            continue

        summary = pd.read_csv(summary_file, index_col=0)
        per_seed = pd.read_csv(per_seed_file)

        row = {
            "model": label,
            "setting": setting,
            "parameters": int(per_seed["parameters"].iloc[0]),
            "seeds": len(per_seed),
            "time_seconds": summary.loc[time_column, "mean"],
        }
        if "update_bytes_per_client" in per_seed:
            row["update_bytes"] = int(per_seed["update_bytes_per_client"].iloc[0])
        if "best_round" in per_seed:
            row["best_rounds"] = ", ".join(str(r) for r in per_seed["best_round"])
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

    headers = ["Model", "Setting", "Params", "Update bytes"] + [label for _, label in METRICS] + ["Time (s)"]
    lines = [
        "# Federated results: the quantum model inside federated learning",
        "",
        "Test sample of 5,000 rows. Mean ± standard deviation over 3 seeds.",
        "Same 8 features, same rows and same federated setup for every model.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * 2 + "---:|" * (len(METRICS) + 3),
    ]
    for _, row in table.iterrows():
        update = f"{int(row['update_bytes']):,}" if pd.notna(row.get("update_bytes")) else "-"
        cells = [row["model"], row["setting"], str(row["parameters"]), update]
        cells += [cell(row, metric) for metric, _ in METRICS]
        cells.append(f"{row['time_seconds']:.0f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "Notes:",
        "- The QNN and the tiny MLP both have 81 parameters, so model size is not the difference.",
        "- 'Update bytes' is what one client sends the server each round (parameters x 4 bytes).",
        "- The two 30-round federated runs use exactly the same budget and settings",
        "  (30 rounds, 1 local epoch, learning rate 0.01), so classical FL and QML-FL",
        "  can be compared directly.",
        "- The best round is chosen from the clients' validation loss, never the test set.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK_SECONDARY)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_accuracy_per_round(path):
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    figure, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ends = []

    for label, folder, colour in CURVES:
        history_file = folder / "history_seed42.csv"
        if not history_file.exists():
            print(f"Skipping curve for {label}: {history_file} not found")
            continue
        history = pd.read_csv(history_file)
        ax.plot(history["round"], history["test_accuracy"], linewidth=2, color=colour, label=label)
        last_round = history["round"].iloc[-1]
        last_value = history["test_accuracy"].iloc[-1]
        ax.plot([last_round], [last_value], "o", markersize=8, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ends.append((last_value, last_round))

    if not ends:
        plt.close(figure)
        return

    # When both curves finish at nearly the same accuracy, their end labels would
    # print on top of each other, so nudge them apart vertically.
    ends.sort()
    too_close = len(ends) == 2 and abs(ends[1][0] - ends[0][0]) < 0.005
    for index, (value, round_number) in enumerate(ends):
        dy = 0
        if too_close:
            dy = -9 if index == 0 else 9
        ax.annotate(f"{value:.1%}", (round_number, value), xytext=(8, dy),
                    textcoords="offset points", va="center", fontsize=9, color=INK)

    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_xlim(left=0)
    ax.margins(x=0.08)
    style(ax, "Test accuracy per federated round (seed 42)", "round", "accuracy")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def draw_recall_per_family(path):
    frames = {}
    for label, folder, colour in CURVES:
        recall_file = folder / "recall_per_family.csv"
        if recall_file.exists():
            frames[label] = (pd.read_csv(recall_file, index_col=0)["share_correct"], colour)

    if not frames:
        print("Skipping per-family figure: no recall files yet")
        return

    families = next(iter(frames.values()))[0].sort_values().index
    positions = range(len(families))
    height = 0.38

    figure, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    for offset, (label, (values, colour)) in zip((height / 2, -height / 2), frames.items()):
        y = [p + offset for p in positions]
        ax.barh(y, values.reindex(families), height=height * 0.92, color=colour, label=label)
        for yy, value in zip(y, values.reindex(families)):
            ax.annotate(f"{value:.0%}", (value, yy), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)

    ax.set_yticks(list(positions))
    ax.set_yticklabels(families, fontsize=9, color=INK)
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    style(ax, "Share of rows classified correctly, per attack family (seed 42)", "", "")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main():
    REVIEW2.mkdir(parents=True, exist_ok=True)
    table = load_table()
    if table.empty:
        raise SystemExit("No results found. Run the experiments first.")

    table.to_csv(REVIEW2 / "federated_results.csv", index=False)
    write_markdown(table, REVIEW2 / "federated_results.md")
    draw_accuracy_per_round(REVIEW2 / "accuracy_per_round.png")
    draw_recall_per_family(REVIEW2 / "recall_per_family.png")

    printable = table[["model", "setting", "parameters"]].copy()
    for metric, label in METRICS:
        printable[label] = [f"{m:.3f} +/- {s:.3f}" for m, s in zip(table[f"{metric}_mean"], table[f"{metric}_std"])]
    print(printable.to_string(index=False))
    print(f"\nSaved federated_results.csv, .md and figures in {REVIEW2}")


if __name__ == "__main__":
    main()
