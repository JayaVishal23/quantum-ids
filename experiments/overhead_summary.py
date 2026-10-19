"""
The security-versus-overhead result (Objective 3).

Compares the same quantum federated training run three ways:
    no encryption        the plain simulation
    X25519               today's classical key exchange
    ML-KEM-768           the post-quantum replacement

and produces, in results/secured/:
    overhead_results.csv        the numbers
    overhead_comparison.png     two panels: what it costs, and what it changes

Run from the project root (after the three runs exist):
    python -m experiments.overhead_summary
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import config

SECURED = config.RESULTS_DIR / "secure_channel"

RUNS = [
    ("No encryption", config.RESULTS_DIR / "federated_models" / "fl_qnn_fedavg", "#898781"),
    ("X25519\n(classical)", SECURED / "fl_qnn_fedavg_x25519_mldsa65", "#2a78d6"),
    ("ML-KEM-768\n(post-quantum)", SECURED / "fl_qnn_fedavg_mlkem768_mldsa65", "#eb6834"),
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"


def load():
    rows = []
    for label, folder, colour in RUNS:
        per_seed_file = folder / "per_seed.csv"
        if not per_seed_file.exists():
            print(f"missing: {per_seed_file}")
            continue
        per_seed = pd.read_csv(per_seed_file)
        summary = pd.read_csv(folder / "summary.csv", index_col=0)

        rows.append({
            "setting": label.replace("\n", " "),
            "colour": colour,
            "seeds": len(per_seed),
            "accuracy_mean": summary.loc["accuracy", "mean"],
            "accuracy_std": summary.loc["accuracy", "std"],
            "f1_mean": summary.loc["f1", "mean"],
            "roc_auc_mean": summary.loc["roc_auc", "mean"],
            "bytes_per_round": summary.loc["total_bytes_per_round", "mean"] if "total_bytes_per_round" in summary.index else 0.0,
            "handshake_bytes_per_round": summary.loc["handshake_bytes_per_round", "mean"] if "handshake_bytes_per_round" in summary.index else 0.0,
            "crypto_ms_per_round": (summary.loc["crypto_seconds_per_round", "mean"] * 1000)
            if "crypto_seconds_per_round" in summary.index else 0.0,
            "round_seconds": summary.loc["total_seconds", "mean"] / per_seed["rounds"].iloc[0],
        })
    return pd.DataFrame(rows)


def draw(table, path):
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    labels = [label for label, _, _ in RUNS][:len(table)]
    positions = range(len(table))

    # What security costs.
    kilobytes = table["bytes_per_round"] / 1000
    left.bar(positions, kilobytes, width=0.55, color=table["colour"])
    for x, value in zip(positions, kilobytes):
        left.text(x, value, f"{value:.1f} KB" if value else "0", ha="center", va="bottom",
                  fontsize=9.5, color=INK)
    left.set_title("What it costs: data sent per round (3 clients)", loc="left",
                   fontsize=11, color=INK_SECONDARY)
    left.set_ylabel("kilobytes per round", fontsize=9, color=MUTED)
    left.set_ylim(0, max(kilobytes.max() * 1.25, 1))

    # What security changes: nothing.
    accuracy = table["accuracy_mean"]
    right.bar(positions, accuracy, width=0.55, color=table["colour"],
              yerr=table["accuracy_std"], capsize=5, ecolor=MUTED)
    for x, value in zip(positions, accuracy):
        right.text(x, value, f"{value:.1%}", ha="center", va="bottom", fontsize=9.5, color=INK)
    right.set_title("What it changes: detection accuracy", loc="left",
                    fontsize=11, color=INK_SECONDARY)
    right.set_ylabel("accuracy", fontsize=9, color=MUTED)
    right.set_ylim(0, 1.0)
    right.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    for ax in (left, right):
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, fontsize=9, color=INK)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)

    figure.suptitle("Post-quantum security: the cost is communication, not detection quality",
                    x=0.01, ha="left", fontsize=12.5, color=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main():
    table = load()
    if table.empty:
        raise SystemExit("No runs found yet.")

    SECURED.mkdir(parents=True, exist_ok=True)
    table.drop(columns=["colour"]).to_csv(SECURED / "overhead_results.csv", index=False)
    draw(table, SECURED / "overhead_comparison.png")

    show = table[["setting", "seeds", "accuracy_mean", "f1_mean", "roc_auc_mean",
                  "handshake_bytes_per_round", "bytes_per_round", "crypto_ms_per_round", "round_seconds"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    if len(table) == 3:
        extra_bytes = table["bytes_per_round"].iloc[2] - table["bytes_per_round"].iloc[1]
        print(f"\nPost-quantum vs classical: +{extra_bytes:,.0f} bytes per round (exact, not timing-dependent)")
        print(f"Accuracy spread across the three settings: "
              f"{table['accuracy_mean'].max() - table['accuracy_mean'].min():.4f}")

        # Timings taken during training vary with whatever else the computer is
        # doing, so the trustworthy numbers come from benchmarks/pqc_cost.py.
        benchmark = SECURED / "crypto_cost_per_round.csv"
        if benchmark.exists():
            measured = pd.read_csv(benchmark)
            column = [c for c in measured.columns if c.startswith("per round")][0]
            classical, post_quantum = measured[column].iloc[0], measured[column].iloc[1]
            print(f"\nCryptography time per round, measured on its own (benchmarks/pqc_cost.py):")
            print(f"  {measured['suite'].iloc[0]:<24}{classical:6.2f} ms")
            print(f"  {measured['suite'].iloc[1]:<24}{post_quantum:6.2f} ms  "
                  f"({post_quantum - classical:+.2f} ms)")
            print(f"  a training round takes about {table['round_seconds'].iloc[0]:.0f} s, "
                  f"so cryptography is roughly {post_quantum / (table['round_seconds'].iloc[0] * 1000):.2%} of it")
        else:
            print("\nRun benchmarks/pqc_cost.py for reliable timing numbers.")

    print(f"\nSaved overhead_results.csv and overhead_comparison.png in {SECURED}")


if __name__ == "__main__":
    main()
