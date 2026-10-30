"""
One figure for the three hardware-realism experiments.

Puts the noise sweep, the finite-shot runs and the readout error mitigation
side by side, because the point only becomes clear when they are together:
the device noise costs nothing, the shot budget costs something, and the
mitigation fixes the part that was not the problem.

Reads the CSVs written by run_noise.py and run_rem.py and writes:
    results/noise/hardware_realism.png

Run from the project root:
    python -m experiments.hardware_summary
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import config

NOISE_DIR = config.RESULTS_DIR / "noise"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK_SECONDARY)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    sweep = pd.read_csv(NOISE_DIR / "noise_results_qnn.csv")
    shots = pd.concat([pd.read_csv(NOISE_DIR / f"noise_results_qnn_{n}shots.csv")
                       for n in (1024, 4096)])
    rem = pd.read_csv(NOISE_DIR / "rem_results_qnn.csv")

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor=SURFACE)

    # ---- 1. accuracy against noise strength -------------------------------
    ax = axes[0]
    by_scale = sweep.groupby("noise_scale")["accuracy"].mean()
    ax.plot(by_scale.index, by_scale.values, marker="o", color=BLUE, linewidth=2)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("noise strength (1 = Eagle-class device)", fontsize=9, color=INK)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for x, y in zip(by_scale.index, by_scale.values):
        ax.annotate(f"{y:.1%}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=INK)
    style(ax, "Accuracy does not move with device noise")

    # ---- 2. the spread caused by finite shots -----------------------------
    ax = axes[1]
    grouped = shots.groupby(["shots", "noise_scale"])["accuracy"].agg(["mean", "std"])
    budgets = sorted(shots["shots"].unique())
    width = 0.34
    for offset, (scale, colour, label) in enumerate(
            [(0.0, BLUE, "noiseless"), (1.0, ORANGE, "Eagle-class noise")]):
        means = [grouped.loc[(n, scale), "mean"] for n in budgets]
        errors = [grouped.loc[(n, scale), "std"] for n in budgets]
        positions = [i + (offset - 0.5) * width for i in range(len(budgets))]
        ax.bar(positions, means, width=width, color=colour, label=label,
               yerr=errors, capsize=5, ecolor=INK_SECONDARY)
        for x, mean, error in zip(positions, means, errors):
            ax.text(x, mean + error, f"±{error * 100:.1f}", ha="center", va="bottom",
                    fontsize=9, color=INK)
    ax.set_ylim(0.75, 0.95)
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([f"{n} shots" for n in budgets], fontsize=9, color=INK)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK_SECONDARY)
    style(ax, "Shot noise is what actually moves the accuracy")

    # ---- 3. what readout error mitigation recovers ------------------------
    ax = axes[2]
    scales = sorted(rem["noise_scale"].unique())
    for offset, (kind, colour, label) in enumerate(
            [("raw", BLUE, "measured"), ("mitigated", ORANGE, "after REM")]):
        values = [rem[(rem["noise_scale"] == s) & (rem["mitigation"] == kind)]
                  ["mean_drift"].iloc[0] for s in scales]
        positions = [i + (offset - 0.5) * width for i in range(len(scales))]
        ax.bar(positions, values, width=width, color=colour, label=label)
    for i, scale in enumerate(scales):
        raw = rem[(rem["noise_scale"] == scale) & (rem["mitigation"] == "raw")]["mean_drift"].iloc[0]
        fixed = rem[(rem["noise_scale"] == scale) & (rem["mitigation"] == "mitigated")]["mean_drift"].iloc[0]
        ax.text(i, max(raw, fixed), f"-{(raw - fixed) / raw:.1%}", ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f"{s:g}x noise" for s in scales], fontsize=9, color=INK)
    ax.set_ylabel("drift of <Z> from the ideal value", fontsize=9, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK_SECONDARY)
    style(ax, "Mitigation removes a shrinking share of the drift")

    figure.suptitle("The detector on noisy hardware: device error is free, sampling error is not",
                    x=0.01, ha="left", fontsize=12.5, color=INK)
    figure.tight_layout()
    path = NOISE_DIR / "hardware_realism.png"
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
