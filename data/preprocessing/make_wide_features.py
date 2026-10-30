"""
Scales ALL 39 numerical features, federated, for the amplitude-encoded model.

The angle-encoded model uses 8 features because it spends one qubit per
feature. Amplitude encoding puts a whole vector into the amplitudes of the
same 8 qubits, so the feature budget is no longer tied to the qubit count and
the model can see far more of the record.

The procedure is exactly the one used for the 8 features, so no new privacy
claim is needed: each client sends only count, min and max per feature, the
server takes the extremes, and every client then scales its own rows.

Everything is written beside the existing files with a "_wide" suffix, so the
8-feature arrays the baseline uses are left exactly as they are.

Run from the project root (after federated/scaling_fed.py):
    python -m data.preprocessing.make_wide_features
"""

import json

import numpy as np
import pandas as pd

from common import config
from data.preprocessing.make_subsample import load_npz, stratified_sample
from federated.scaling_fed import (
    client_summary_statistics,
    server_combine_statistics,
    to_arrays,
)

WIDE_PARAMS_FILE = "scaling_params_wide.json"


def main():
    features = list(config.NUMERIC_FEATURES)
    print(f"Scaling all {len(features)} numerical features, federated.\n")

    # --- clients report only count, min and max, exactly as before ---
    messages = []
    for client_id in config.CLIENT_ATTACKS:
        messages.append(client_summary_statistics(client_id, features))
        print(f"Client {client_id}: sent count/min/max for {len(features)} features")

    params = server_combine_statistics(messages, features)
    with open(config.SHARED_DIR / WIDE_PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)

    # --- each client scales its own rows ---
    for client_id in config.CLIENT_ATTACKS:
        client_dir = config.CLIENTS_DIR / f"client_{client_id}"
        for part in ("train", "val"):
            rows = pd.read_csv(client_dir / f"{part}.csv")
            np.savez(client_dir / f"{part}_wide.npz", **to_arrays(rows, params, features))
        print(f"Client {client_id}: scaled its own train and val data")

    # --- the evaluator scales the official test set the same way ---
    test_rows = pd.read_csv(config.TEST_CSV)
    if len(test_rows) != config.EXPECTED_TEST_ROWS:
        raise ValueError(f"Expected {config.EXPECTED_TEST_ROWS} test rows, found {len(test_rows)}.")
    test = to_arrays(test_rows, params, features)
    np.savez(config.TEST_DIR / "test_wide.npz", **test)

    # --- the same stratified sample sizes the other models use ---
    for client_id in config.CLIENT_ATTACKS:
        client_dir = config.CLIENTS_DIR / f"client_{client_id}"
        for part, size in (("train", config.TRAIN_ROWS_PER_CLIENT), ("val", config.VAL_ROWS_PER_CLIENT)):
            full = load_npz(client_dir / f"{part}_wide.npz")
            np.savez(client_dir / f"{part}_wide_small.npz", **stratified_sample(full, size))
    np.savez(config.TEST_DIR / "test_wide_small.npz",
             **stratified_sample(load_npz(config.TEST_DIR / "test_wide.npz"), config.TEST_ROWS))

    low, high = config.ANGLE_RANGE
    sample = load_npz(config.CLIENTS_DIR / "client_1" / "train_wide_small.npz")
    print(f"\nClient 1 sample: {sample['X'].shape}, "
          f"values in [{sample['X'].min():.3f}, {sample['X'].max():.3f}] "
          f"(allowed [{low:.3f}, {high:.3f}])")
    print(f"Test sample: {load_npz(config.TEST_DIR / 'test_wide_small.npz')['X'].shape}")
    print(f"Saved {config.SHARED_DIR / WIDE_PARAMS_FILE} and the *_wide.npz files")


if __name__ == "__main__":
    main()
