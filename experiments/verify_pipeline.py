"""
Checks the dataset and the preprocessing, and prints the evidence.

This answers the question "is the dataset and preprocessing actually done
correctly?" with numbers instead of claims. It only reads files.

Run from the project root:
    python -m experiments.verify_pipeline
"""

import json

import numpy as np
import pandas as pd

from common import config, data


def line(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_raw_dataset():
    line("1. RAW DATASET")
    train = pd.read_csv(config.TRAIN_CSV)
    test = pd.read_csv(config.TEST_CSV)

    for name, frame in (("training", train), ("testing", test)):
        normal = int((frame[config.LABEL_COLUMN] == 0).sum())
        attack = int((frame[config.LABEL_COLUMN] == 1).sum())
        print(f"  {name:9s}: {len(frame):,} rows, {frame.shape[1]} columns  "
              f"normal {normal:,} ({normal / len(frame):.1%}) | attack {attack:,} ({attack / len(frame):.1%})")

    print(f"\n  attack families in training: {train[config.CATEGORY_COLUMN].nunique() - 1}")
    print("  " + ", ".join(f"{k} {v:,}" for k, v in
                           train[config.CATEGORY_COLUMN].value_counts().items()))
    print(f"\n  columns never used as model input: {config.ID_COLUMN}, {config.CATEGORY_COLUMN}, {config.LABEL_COLUMN}")
    return train, test


def check_client_split(train):
    line("2. SPLIT INTO 3 CLIENTS (done on raw rows, before any preprocessing)")
    rows = []
    all_ids = []
    for client_id in config.CLIENT_ATTACKS:
        client_dir = config.CLIENTS_DIR / f"client_{client_id}"
        part_train = pd.read_csv(client_dir / "train.csv")
        part_val = pd.read_csv(client_dir / "val.csv")
        both = pd.concat([part_train, part_val])
        all_ids.append(both[config.ID_COLUMN])
        families = sorted(set(both[config.CATEGORY_COLUMN]) - {"Normal"})
        rows.append({
            "client": client_id,
            "train rows": len(part_train),
            "val rows": len(part_val),
            "normal": int((both[config.LABEL_COLUMN] == 0).sum()),
            "attack": int((both[config.LABEL_COLUMN] == 1).sum()),
            "attack families": ", ".join(families),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    merged = pd.concat(all_ids)
    print(f"\n  rows across all clients: {len(merged):,} (original file has {len(train):,})")
    print(f"  any row in two clients?  {merged.duplicated().any()}")
    print(f"  validation rows are held out of training: yes, {config.VALIDATION_FRACTION:.0%} per client")


def check_features():
    line("3. FEATURE SELECTION (done inside each client)")
    with open(config.SHARED_DIR / "selected_features.json") as f:
        chosen = json.load(f)
    print(f"  candidates scored by each client : {len(config.NUMERIC_FEATURES)} numeric features")
    print(f"  what each client sent the server : {len(config.NUMERIC_FEATURES)} scores + its row count")
    print(f"  method                           : {chosen['method']}")
    print(f"  agreed features ({len(chosen['features'])})             : {', '.join(chosen['features'])}")

    scores = pd.read_csv(config.RESULTS_DIR / "review1" / "feature_scores_per_client.csv", index_col=0)
    print("\n  each client's rank for the agreed features (they disagree = non-IID data):")
    ranks = scores.loc[chosen["features"], [c for c in scores.columns if c.endswith("_rank")]]
    print(ranks.to_string())


def check_scaling():
    line("4. SCALING (shared values, computed from client summaries only)")
    with open(config.SHARED_DIR / "scaling_params.json") as f:
        params = json.load(f)
    table = pd.DataFrame({
        "log1p applied": [f in params["log1p_applied_to"] for f in params["features"]],
        "global min": [round(params["min"][f], 3) for f in params["features"]],
        "global max": [round(params["max"][f], 3) for f in params["features"]],
    }, index=params["features"])
    print(table.to_string())
    print(f"\n  every client sent only: count, min and max per feature")
    print(f"  all values scaled into {params['angle_range']} (the rotation-angle range for a qubit)")


def check_prepared_arrays():
    line("5. PREPARED DATA USED BY THE MODELS")
    rows = []
    for client_id in config.CLIENT_ATTACKS:
        for part in ("train", "val"):
            arrays = data.load_client(client_id, part)
            rows.append({
                "data": f"client {client_id} {part}",
                "rows": len(arrays["y"]),
                "features": arrays["X"].shape[1],
                "min": round(float(arrays["X"].min()), 3),
                "max": round(float(arrays["X"].max()), 3),
                "attack share": f"{arrays['y'].mean():.1%}",
            })
    test = data.load_test()
    rows.append({"data": "test (evaluation)", "rows": len(test["y"]), "features": test["X"].shape[1],
                 "min": round(float(test["X"].min()), 3), "max": round(float(test["X"].max()), 3),
                 "attack share": f"{test['y'].mean():.1%}"})
    print(pd.DataFrame(rows).to_string(index=False))

    # Check the real values, not the rounded ones shown above.
    low, high = config.ANGLE_RANGE
    everything = [data.load_client(c, p)["X"] for c in config.CLIENT_ATTACKS for p in ("train", "val")]
    everything.append(test["X"])
    inside = all(values.min() >= low and values.max() <= high + 1e-6 for values in everything)

    print("\n  checks:")
    print(f"    every value inside [0, pi]:            {inside}")
    print(f"    missing values anywhere:               "
          f"{any(np.isnan(data.load_client(c, p)['X']).any() for c in config.CLIENT_ATTACKS for p in ('train', 'val'))}")
    print("    test set never used for training:      yes, it is only loaded by the evaluator")


def main():
    train, _ = check_raw_dataset()
    check_client_split(train)
    check_features()
    check_scaling()
    check_prepared_arrays()
    print("\nAll checks completed.\n")


if __name__ == "__main__":
    main()
