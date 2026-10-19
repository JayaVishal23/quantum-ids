"""
Turns the poisoning runs into one table and one figure.

Reads results/poisoning/ and produces:
    poisoning_results.csv      every number
    poisoning_comparison.png   FedAvg against the median defence

Run from the project root (after experiments/run_poisoning.py):
    python -m experiments.poisoning_summary
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from attacks.poisoning import LABEL_FLIP, MODEL_POISON, NO_ATTACK
from common import config

POISONING = config.RESULTS_DIR / "poisoning"

SCENARIOS = [
    (NO_ATTACK, "No attack"),
    (LABEL_FLIP, "Label flipping"),
    (MODEL_POISON, "Model poisoning"),
]
AGGREGATORS = [
    ("fedavg", "FedAvg (no defence)", "#2a78d6"),
    ("median", "Median (defence)", "#eb6834"),
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"


def folder_for(model, aggregator, attack):
    name = f"fl_{model}_{aggregator}"
    if attack != NO_ATTACK:
        name += f"_{attack}"
    return POISONING / name


def load(model):
    rows = []
    for attack, attack_label in SCENARIOS:
        for aggregator, aggregator_label, _ in AGGREGATORS:
            folder = folder_for(model, aggregator, attack)
            if not (folder / "summary.csv").exists():
                print(f"missing: {folder}")
                continue
            summary = pd.read_csv(folder / "summary.csv", index_col=0)
            per_seed = pd.read_csv(folder / "per_seed.csv")
            rows.append({
                "scenario": attack_label,
                "aggregation": aggregator_label,
                "accuracy_mean": summary.loc["accuracy", "mean"],
                "accuracy_std": summary.loc["accuracy", "std"],
                "recall_mean": summary.loc["recall", "mean"],
                "recall_std": summary.loc["recall", "std"],
                "f1_mean": summary.loc["f1", "mean"],
                "roc_auc_mean": summary.loc["roc_auc", "mean"],
                "missed_attacks": per_seed["fn"].mean(),
                "seeds": len(per_seed),
            })
    return pd.DataFrame(rows)


def draw(table, path):
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5), facecolor=SURFACE)
    labels = [label for _, label in SCENARIOS]
    width = 0.36

    panels = [
        ("accuracy_mean", "accuracy_std", "Detection accuracy", axes[0]),
        ("recall_mean", "recall_std", "Attacks caught (recall)", axes[1]),
    ]

    for mean_column, std_column, title, ax in panels:
        for index, (aggregator, aggregator_label, colour) in enumerate(AGGREGATORS):
            part = table[table["aggregation"] == aggregator_label].set_index("scenario")
            if part.empty:
                continue
            values = [part.loc[label, mean_column] if label in part.index else 0 for label in labels]
            errors = [part.loc[label, std_column] if label in part.index else 0 for label in labels]
            positions = [i + (index - 0.5) * width for i in range(len(labels))]
            ax.bar(positions, values, width=width * 0.92, color=colour, label=aggregator_label,
                   yerr=errors, capsize=4, ecolor=MUTED)
            for x, value in zip(positions, values):
                ax.text(x, value + 0.015, f"{value:.0%}", ha="center", va="bottom", fontsize=9, color=INK)

        ax.set_title(title, loc="left", fontsize=11.5, color=INK_SECONDARY)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9.5, color=INK)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    figure.suptitle("One malicious client out of three: what it does, and what the median defence stops",
                    x=0.01, ha="left", fontsize=12.5, color=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["qnn", "mlp"], default="qnn")
    args = parser.parse_args()

    table = load(args.model)
    if table.empty:
        raise SystemExit(f"No poisoning runs found for {args.model}. Run experiments/run_poisoning.py first.")

    table.insert(0, "model", args.model)
    table.to_csv(POISONING / f"poisoning_results_{args.model}.csv", index=False)
    draw(table, POISONING / f"poisoning_comparison_{args.model}.png")

    print("\nAccuracy (mean +/- std over seeds)\n")
    pivot = table.pivot(index="scenario", columns="aggregation", values="accuracy_mean")
    pivot = pivot.reindex([label for _, label in SCENARIOS])
    print(pivot.to_string(float_format=lambda v: f"{v:.1%}"))

    print("\nAttacks caught (recall)\n")
    recall = table.pivot(index="scenario", columns="aggregation", values="recall_mean")
    recall = recall.reindex([label for _, label in SCENARIOS])
    print(recall.to_string(float_format=lambda v: f"{v:.1%}"))

    print("\nAttacks missed (average count out of about 2,750)\n")
    missed = table.pivot(index="scenario", columns="aggregation", values="missed_attacks")
    missed = missed.reindex([label for _, label in SCENARIOS])
    print(missed.to_string(float_format=lambda v: f"{v:,.0f}"))

    print(f"\nSaved poisoning_results_{args.model}.csv and poisoning_comparison_{args.model}.png in {POISONING}")


if __name__ == "__main__":
    main()
