"""
Angle encoding against amplitude encoding: does more features fix the alarms?

Puts the two quantum models side by side on the things the comparison is
actually about: how many features each one sees, how many false alarms it
raises, and what it costs to send its parameters every round.

Reads the federated runs and writes:
    results/encoding/encoding_results.csv
    results/encoding/encoding_comparison.png

Run from the project root:
    python -m experiments.encoding_summary
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import config

OUTPUT_DIR = config.RESULTS_DIR / "encoding"

RUNS = [
    ("Angle encoding\n8 features", config.RESULTS_DIR / "review2" / "fl_qnn_fedavg", 8, "#2a78d6"),
    ("Amplitude encoding\n39 features", config.RESULTS_DIR / "encoding" / "fl_qnn_amplitude_fedavg", 39, "#eb6834"),
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"


def load():
    rows = []
    for label, folder, features, colour in RUNS:
        summary_file = folder / "summary.csv"
        if not summary_file.exists():
            print(f"missing: {summary_file}")
            continue
        summary = pd.read_csv(summary_file, index_col=0)
        per_seed = pd.read_csv(folder / "per_seed.csv")

        row = {
            "encoding": label.replace("\n", ", "),
            "label": label,
            "colour": colour,
            "features_seen": features,
            "parameters": int(per_seed["parameters"].iloc[0]),
            "bytes_per_round": int(per_seed["update_bytes_per_client"].iloc[0]),
            "seeds": len(per_seed),
        }
        for metric in ("accuracy", "recall", "false_positive_rate", "f1", "roc_auc"):
            row[metric] = summary.loc[metric, "mean"]
            row[f"{metric}_std"] = summary.loc[metric, "std"]
        for metric in ("tuned_accuracy", "tuned_recall", "tuned_false_positive_rate"):
            if metric in summary.index:
                row[metric] = summary.loc[metric, "mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def draw(table, path):
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor=SURFACE)
    positions = range(len(table))
    labels = table["label"].tolist()

    panels = [
        ("false_positive_rate", "False alarms on normal traffic", True),
        ("accuracy", "Detection accuracy", True),
        ("bytes_per_round", "Parameters sent per round", False),
    ]

    for ax, (column, title, is_share) in zip(axes, panels):
        values = table[column]
        ax.bar(positions, values, width=0.5, color=table["colour"])
        for x, value in zip(positions, values):
            text = f"{value:.1%}" if is_share else f"{value:,.0f} B"
            ax.text(x, value, text, ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.set_title(title, loc="left", fontsize=11, color=INK_SECONDARY)
        ax.set_ylim(0, values.max() * 1.3)
        if is_share:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, fontsize=9, color=INK)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)

    figure.suptitle("More features per qubit: fewer false alarms, bigger updates",
                    x=0.01, ha="left", fontsize=12.5, color=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main():
    table = load()
    if table.empty:
        raise SystemExit("No encoding runs found yet.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["encoding", "features_seen", "parameters", "bytes_per_round",
               "accuracy", "recall", "false_positive_rate", "roc_auc"]
    print(table[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if "tuned_false_positive_rate" in table:
        print("\nWith the tuned decision threshold:")
        tuned = ["encoding", "tuned_accuracy", "tuned_recall", "tuned_false_positive_rate"]
        print(table[[c for c in tuned if c in table]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))

    table.drop(columns=["colour", "label"]).to_csv(OUTPUT_DIR / "encoding_results.csv", index=False)
    if len(table) > 1:
        draw(table, OUTPUT_DIR / "encoding_comparison.png")
    print(f"\nSaved encoding_results.csv in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
