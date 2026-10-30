"""
Does readout error mitigation buy the detector anything?

run_noise.py showed that an Eagle-class device costs the detector no accuracy
at all. This script asks the follow-up question a reviewer would ask: if we
apply the standard correction for readout error, does anything improve?

It measures the answer at two levels, which is the point of the experiment:

  the measured numbers   how far the eight <Z> values drift from their ideal
                         values, before and after correction. This is what REM
                         is designed to fix, so this is where it should show.
  the decisions          accuracy, recall and false alarm rate. This is what
                         the project is actually judged on.

The noise strength is swept well past today's hardware, because a correction
that does nothing at a 1.3% readout error may still matter at 13%.

The circuits are simulated once per noise level and the correction is then
applied to the saved expectation values, since REM is post-processing and
needs no further circuit runs.

Run from the project root (after the model has been trained):
    python -m experiments.run_rem --rows 300
    python -m experiments.run_rem --rows 300 --scales 1 5 10
"""

import argparse
import time

import numpy as np
import pandas as pd

from common import config, data, metrics
from experiments.run_noise import load_weights, subsample
from models.quantum import mitigation
from models.quantum.noise import build_noisy_model, READOUT_ERROR

OUTPUT_DIR = config.RESULTS_DIR / "noise"

DEFAULT_SCALES = [1.0, 5.0, 10.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["qnn", "qnn_amplitude"], default="qnn")
    parser.add_argument("--rows", type=int, default=300,
                        help="test rows to evaluate (density-matrix simulation is slow)")
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    parser.add_argument("--shots", type=int, default=None,
                        help="finite-shot sampling instead of exact expectation values")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    weights, path = load_weights(args.model)
    wide = args.model == "qnn_amplitude"
    test = subsample(data.load_test(wide=wide), args.rows)

    print(f"Model   : {args.model}, trained weights from {path.name}")
    print(f"Test    : {len(test['y']):,} rows (stratified from the {config.TEST_ROWS:,}-row test sample)")
    if args.shots:
        print(f"Shots   : {args.shots} per circuit")
    print()

    # The ideal circuit output is the reference everything else is measured
    # against. Built on the same simulator so only the noise differs.
    ideal_model = build_noisy_model(args.model, scale=0.0, shots=args.shots)
    ideal_model.load_state_dict(weights)
    ideal_features = mitigation.quantum_features(ideal_model, test["X"], args.batch_size)
    ideal_scores = metrics.evaluate(test["y"],
                                    mitigation.head_probability(ideal_model, ideal_features))
    print(f"Ideal simulator: accuracy {ideal_scores['accuracy']:.4f} | "
          f"recall {ideal_scores['recall']:.4f} | FPR {ideal_scores['false_positive_rate']:.4f}")
    print()

    n_qubits = ideal_features.shape[1]
    rows = []

    for scale in args.scales:
        model = build_noisy_model(args.model, scale=scale, shots=args.shots)
        model.load_state_dict(weights)

        gain, offset = mitigation.calibration_matrix(n_qubits, scale=scale, shots=args.shots)

        start = time.perf_counter()
        measured = mitigation.quantum_features(model, test["X"], args.batch_size)
        seconds = time.perf_counter() - start
        corrected = mitigation.correct(measured, gain, offset)

        print(f"noise scale {scale:g}  (readout error {READOUT_ERROR * scale:.1%}, "
              f"calibration gain {gain.mean():.4f})  {seconds:.0f}s")

        for label, features in (("raw", measured), ("mitigated", corrected)):
            drift = np.abs(features - ideal_features)
            scores = metrics.evaluate(test["y"],
                                      mitigation.head_probability(model, features))
            rows.append({
                "model": args.model, "noise_scale": scale, "shots": args.shots or 0,
                "rows": len(test["y"]), "readout_error": READOUT_ERROR * scale,
                "calibration_gain": gain.mean(), "mitigation": label,
                "mean_drift": drift.mean(), "max_drift": drift.max(),
                **{name: scores[name] for name in metrics.METRIC_NAMES},
            })
            print(f"    {label:<10} | drift from ideal {drift.mean():.5f} mean, "
                  f"{drift.max():.5f} worst | accuracy {scores['accuracy']:.4f} | "
                  f"recall {scores['recall']:.4f} | FPR {scores['false_positive_rate']:.4f}")

        recovered = rows[-2]["mean_drift"] - rows[-1]["mean_drift"]
        share = recovered / rows[-2]["mean_drift"] if rows[-2]["mean_drift"] else 0.0
        print(f"    REM removes {share:.1%} of the drift in the measured values, "
              f"and changes accuracy by "
              f"{(rows[-1]['accuracy'] - rows[-2]['accuracy']) * 100:+.2f} points.")
        print()

    table = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.shots}shots" if args.shots else ""
    table.to_csv(OUTPUT_DIR / f"rem_results_{args.model}{suffix}.csv", index=False)
    print(f"Saved rem_results_{args.model}{suffix}.csv in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
