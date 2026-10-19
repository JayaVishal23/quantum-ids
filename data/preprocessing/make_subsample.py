"""
Step A4: make the smaller dataset used by every experiment.

Simulating an 8-qubit circuit is slow: training on ~140,000 rows for many
federated rounds would take far too long on a laptop. So every model,
classical and quantum, is trained and tested on the same smaller sample.
Using the same sample for all models is what keeps the comparison fair.

The sample is stratified by attack family. That means each family keeps
roughly the same share it had in the full data. Rare families like Worms
don't disappear.

Each client samples its own data. The test sample is drawn from the
already-scaled official test set.

Run from the project root (after federated/scaling_fed.py):
    python -m data.preprocessing.make_subsample
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import config


def stratified_sample(arrays, size):
    """Pick `size` rows, keeping the attack-family mix of the original."""
    if size >= len(arrays["y"]):
        return arrays

    positions = np.arange(len(arrays["y"]))
    chosen, _ = train_test_split(
        positions,
        train_size=size,
        stratify=arrays["attack_cat"],
        random_state=config.SPLIT_SEED,
    )
    chosen = np.sort(chosen)
    return {name: values[chosen] for name, values in arrays.items()}


def load_npz(path):
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def attack_share(arrays):
    return arrays["y"].mean()


def main():
    rows = []

    for client_id in config.CLIENT_ATTACKS:
        client_dir = config.CLIENTS_DIR / f"client_{client_id}"
        for part, size in (("train", config.TRAIN_ROWS_PER_CLIENT), ("val", config.VAL_ROWS_PER_CLIENT)):
            full = load_npz(client_dir / f"{part}.npz")
            small = stratified_sample(full, size)
            np.savez(client_dir / f"{part}_small.npz", **small)
            rows.append((f"client_{client_id}", part, len(full["y"]), len(small["y"]),
                         attack_share(full), attack_share(small)))

    full_test = load_npz(config.TEST_DIR / "test.npz")
    small_test = stratified_sample(full_test, config.TEST_ROWS)
    np.savez(config.TEST_DIR / "test_small.npz", **small_test)
    rows.append(("test", "test", len(full_test["y"]), len(small_test["y"]),
                 attack_share(full_test), attack_share(small_test)))

    table = pd.DataFrame(rows, columns=["owner", "part", "full_rows", "sample_rows",
                                        "attack_share_full", "attack_share_sample"])
    table["difference"] = (table["attack_share_sample"] - table["attack_share_full"]).abs()

    print(table.to_string(index=False, formatters={
        "attack_share_full": "{:.2%}".format,
        "attack_share_sample": "{:.2%}".format,
        "difference": "{:.2%}".format,
    }))

    worst = table["difference"].max()
    if worst > 0.01:
        raise ValueError(f"A sample's attack share is off by {worst:.2%} (limit 1%).")

    print("\nTest sample, rows per attack family:")
    print(pd.Series(small_test["attack_cat"]).value_counts().to_string())

    output_dir = config.RESULTS_DIR / "baseline_models"
    table.to_csv(output_dir / "subsample_summary.csv", index=False)
    print("\nAll samples within 1% of the original attack share.")


if __name__ == "__main__":
    main()
