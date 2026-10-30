"""
How much of the detector survives on noisy hardware?

Takes a model that was TRAINED on the ideal simulator and re-evaluates it on a
density-matrix simulator carrying realistic gate, decoherence and readout
errors. Training is left alone on purpose: this answers "would the trained
detector still work on a real machine?", which is the question that matters
for deployment.

The noise strength is swept rather than fixed at one value, because a single
number cannot tell you whether accuracy degrades gently or collapses.

    scale 0     ideal, and a check that the noisy path reproduces it exactly
    scale 0.5   better than today's hardware
    scale 1     an IBM Eagle-class device (see models/quantum/noise.py)
    scale 2     worse than today's hardware

Density-matrix simulation is far slower than state-vector, so the evaluation
uses a stratified subsample of the test set. The size is printed and saved
with the results.

Run from the project root (after the model has been trained):
    python -m experiments.run_noise --model qnn
    python -m experiments.run_noise --model qnn --rows 500 --scales 0 1
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch

from common import config, data, metrics
from models.quantum.noise import build_noisy_model, describe
from training.loops import predict_probability

OUTPUT_DIR = config.RESULTS_DIR / "noise"

TRAINED = {
    "qnn": [
        config.PROJECT_ROOT / "models" / "trained" / "global_qnn.pt",
        config.RESULTS_DIR / "review2" / "fl_qnn_fedavg" / "global_model_seed42.pt",
    ],
    "qnn_amplitude": [
        config.RESULTS_DIR / "encoding" / "fl_qnn_amplitude_fedavg" / "global_model_seed42.pt",
    ],
}

DEFAULT_SCALES = [0.0, 0.5, 1.0, 2.0]


def load_weights(model_name):
    for path in TRAINED[model_name]:
        if path.exists():
            return torch.load(path), path
    raise SystemExit(f"No trained {model_name} found. Looked in:\n  " +
                     "\n  ".join(str(p) for p in TRAINED[model_name]))


def subsample(arrays, rows, seed=config.SPLIT_SEED):
    """A stratified slice of the test set, so the class mix is preserved."""
    if rows >= len(arrays["y"]):
        return arrays
    from sklearn.model_selection import train_test_split
    positions, _ = train_test_split(
        np.arange(len(arrays["y"])), train_size=rows,
        stratify=arrays["attack_cat"], random_state=seed)
    positions = np.sort(positions)
    return {name: values[positions] for name, values in arrays.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["qnn", "qnn_amplitude"], default="qnn")
    parser.add_argument("--rows", type=int, default=1000,
                        help="test rows to evaluate (density-matrix simulation is slow)")
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    parser.add_argument("--shots", type=int, default=None,
                        help="finite-shot sampling instead of exact expectation values")
    parser.add_argument("--repeats", type=int, default=1,
                        help="re-evaluate this many times; with shots it measures the spread")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    weights, path = load_weights(args.model)
    wide = args.model == "qnn_amplitude"
    test = subsample(data.load_test(wide=wide), args.rows)

    print(f"Model   : {args.model}, trained weights from {path.name}")
    print(f"Test    : {len(test['y']):,} rows (stratified from the {config.TEST_ROWS:,}-row test sample)")
    print(f"Noise   : {describe(1.0)}")
    if args.shots:
        print(f"Shots   : {args.shots} per circuit")
    print()

    rows = []
    for scale in args.scales:
        model = build_noisy_model(args.model, scale=scale, shots=args.shots)
        model.load_state_dict(weights)
        model.eval()

        for repeat in range(args.repeats):
            start = time.perf_counter()
            probability = predict_probability(model, test["X"], batch_size=args.batch_size)
            seconds = time.perf_counter() - start

            scores = metrics.evaluate(test["y"], probability)
            rows.append({
                "model": args.model, "noise_scale": scale, "shots": args.shots or 0,
                "repeat": repeat, "rows": len(test["y"]), "seconds": seconds,
                **{name: scores[name] for name in metrics.METRIC_NAMES},
            })
            print(f"  scale {scale:<4} | repeat {repeat + 1}/{args.repeats} | "
                  f"accuracy {scores['accuracy']:.4f} | recall {scores['recall']:.4f} | "
                  f"FPR {scores['false_positive_rate']:.4f} | {seconds:.0f}s")

    table = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.shots}shots" if args.shots else ""
    table.to_csv(OUTPUT_DIR / f"noise_results_{args.model}{suffix}.csv", index=False)

    if args.repeats > 1:
        print()
        print("Spread across repeats (this is the shot noise):")
        spread = table.groupby("noise_scale")[list(metrics.METRIC_NAMES)].agg(["mean", "std"])
        print(spread[["accuracy", "recall", "false_positive_rate"]].to_string(
            float_format=lambda v: f"{v:.4f}"))

    # Compare the MEANS over repeats. Comparing single runs would mostly show
    # shot noise, not the effect of the device.
    ideal = table[table["noise_scale"] == 0]["accuracy"]
    device = table[table["noise_scale"] == 1]["accuracy"]
    if not ideal.empty and not device.empty:
        drop = (ideal.mean() - device.mean()) * 100
        spread = table.groupby("noise_scale")["accuracy"].std().max()
        print(f"\nOn an Eagle-class device the model loses {drop:.2f} accuracy points "
              f"({ideal.mean():.2%} -> {device.mean():.2%}).")
        if args.repeats > 1 and abs(drop) < spread * 100:
            print(f"That difference is smaller than the spread between repeats "
                  f"({spread * 100:.2f} points), so it is not distinguishable from shot noise.")
    print(f"Saved noise_results_{args.model}{suffix}.csv in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
