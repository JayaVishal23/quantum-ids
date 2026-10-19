"""
Step A3: federated scaling.

Every client must scale features the same way. If client 1 used its own
min/max and client 2 used different ones, the same raw value would turn into
different numbers (and different qubit angles), and averaging their models
would mix incompatible inputs.

So, again, the work is split:

  CLIENT
    - keeps only the 8 agreed features
    - applies log1p to heavy-tailed features
    - sends the server count, min and max of each feature (24 numbers + count)

  SERVER
    - global min = smallest of the clients' mins
    - global max = largest of the clients' maxes
    - sends these back

  CLIENT again
    - scales its own train and validation rows into [0, pi]

The official test set is scaled with exactly the same numbers, so it looks
like data from any other client.

Run from the project root:
    python -m federated.scaling_fed
"""

import json

import numpy as np
import pandas as pd

from common import config


def load_selected_features():
    with open(config.SHARED_DIR / "selected_features.json") as f:
        return json.load(f)["features"]


def apply_log(frame, features):
    """log1p on heavy-tailed columns. Values are never negative in UNSW-NB15,
    but clip at 0 anyway so log1p can't fail."""
    frame = frame[features].astype("float64").copy()
    for column in features:
        if column in config.HEAVY_TAILED_FEATURES:
            frame[column] = np.log1p(frame[column].clip(lower=0))
    return frame


# ------------------------------------------------------------------
# Client side
# ------------------------------------------------------------------

def client_summary_statistics(client_id, features):
    """Runs inside one client. Returns count, min and max per feature."""
    train_rows = pd.read_csv(config.CLIENTS_DIR / f"client_{client_id}" / "train.csv")
    values = apply_log(train_rows, features)

    return {
        "client_id": client_id,
        "count": len(values),
        "min": values.min().to_dict(),
        "max": values.max().to_dict(),
    }


def scale(values, params, features):
    """Map each feature from [global_min, global_max] to [0, pi]."""
    low, high = config.ANGLE_RANGE
    scaled = np.empty(values.shape, dtype="float32")

    for i, column in enumerate(features):
        lo = params["min"][column]
        hi = params["max"][column]
        span = hi - lo if hi > lo else 1.0  # a constant column would divide by zero
        unit = (values[column].to_numpy() - lo) / span
        scaled[:, i] = low + np.clip(unit, 0.0, 1.0) * (high - low)

    return scaled


def to_arrays(rows, params, features):
    return {
        "X": scale(apply_log(rows, features), params, features),
        "y": rows[config.LABEL_COLUMN].to_numpy().astype("int64"),
        "attack_cat": rows[config.CATEGORY_COLUMN].to_numpy().astype("U20"),
    }


def client_scale_own_data(client_id, params, features):
    client_dir = config.CLIENTS_DIR / f"client_{client_id}"
    for part in ("train", "val"):
        rows = pd.read_csv(client_dir / f"{part}.csv")
        np.savez(client_dir / f"{part}.npz", **to_arrays(rows, params, features))


# ------------------------------------------------------------------
# Server side
# ------------------------------------------------------------------

def server_combine_statistics(messages, features):
    return {
        "features": features,
        "log1p_applied_to": [f for f in features if f in config.HEAVY_TAILED_FEATURES],
        "angle_range": list(config.ANGLE_RANGE),
        "min": {f: min(m["min"][f] for m in messages) for f in features},
        "max": {f: max(m["max"][f] for m in messages) for f in features},
        "total_rows": sum(m["count"] for m in messages),
    }


def main():
    features = load_selected_features()
    print(f"Agreed features: {features}\n")

    messages = []
    for client_id in config.CLIENT_ATTACKS:
        messages.append(client_summary_statistics(client_id, features))
        print(f"Client {client_id}: sent count/min/max for {len(features)} features")

    params = server_combine_statistics(messages, features)

    with open(config.SHARED_DIR / "scaling_params.json", "w") as f:
        json.dump(params, f, indent=2)

    print("\nGlobal scaling parameters (after log1p where applied):\n")
    table = pd.DataFrame({
        "log1p": [f in params["log1p_applied_to"] for f in features],
        "global_min": [params["min"][f] for f in features],
        "global_max": [params["max"][f] for f in features],
    }, index=features)
    print(table.round(3).to_string())

    for client_id in config.CLIENT_ATTACKS:
        client_scale_own_data(client_id, params, features)
        print(f"Client {client_id}: scaled its own train and val data")

    # The evaluator scales the official test set with the same parameters.
    test_rows = pd.read_csv(config.TEST_CSV)
    if len(test_rows) != config.EXPECTED_TEST_ROWS:
        raise ValueError(f"Expected {config.EXPECTED_TEST_ROWS} test rows, found {len(test_rows)}.")

    config.TEST_DIR.mkdir(parents=True, exist_ok=True)
    test = to_arrays(test_rows, params, features)
    np.savez(config.TEST_DIR / "test.npz", **test)

    low, high = config.ANGLE_RANGE
    print(f"\nTest set scaled: {test['X'].shape}, "
          f"values in [{test['X'].min():.3f}, {test['X'].max():.3f}] "
          f"(allowed [{low:.3f}, {high:.3f}])")
    print(f"Saved {config.SHARED_DIR / 'scaling_params.json'}")


if __name__ == "__main__":
    main()
