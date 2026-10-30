"""
Step A1: split the raw UNSW-NB15 training rows into 3 federated clients.

In a real deployment every organisation already has its own traffic, so this
step wouldn't exist. Here we only *simulate* that situation. The script does
no learning at all: no scaling, no feature scoring, no encoding. It only
decides which raw row belongs to which client. Everything that learns from
data happens later, inside each client.

How rows are shared out:
  - Normal traffic is shuffled and divided equally between the 3 clients.
  - Each attack family goes entirely to one client (see common/config.py).
  - Inside each client, 20% of rows are kept aside as a validation set,
    stratified by attack family so every family appears in both parts.

Run from the project root:
    python -m data.preprocessing.split_raw_clients
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import config


def load_raw_training_rows():
    rows = pd.read_csv(config.TRAIN_CSV)

    if len(rows) != config.EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"Expected {config.EXPECTED_TRAIN_ROWS} rows in {config.TRAIN_CSV.name}, "
            f"found {len(rows)}. Are the train/test files swapped?"
        )
    return rows


def assign_rows_to_clients(rows):
    """Return {client_id: DataFrame of that client's raw rows}."""
    rng = np.random.default_rng(config.SPLIT_SEED)

    # Normal traffic: shuffle once, then cut into equal parts.
    normal_positions = np.flatnonzero(rows[config.CATEGORY_COLUMN] == "Normal")
    rng.shuffle(normal_positions)
    normal_parts = np.array_split(normal_positions, config.NUM_CLIENTS)

    clients = {}
    for client_id, attack_families in config.CLIENT_ATTACKS.items():
        attack_positions = np.flatnonzero(
            rows[config.CATEGORY_COLUMN].isin(attack_families)
        )
        positions = np.concatenate([normal_parts[client_id - 1], attack_positions])
        clients[client_id] = rows.iloc[positions]

    return clients


def check_split(rows, clients):
    """Make sure every row went to exactly one client."""
    all_ids = pd.concat([client[config.ID_COLUMN] for client in clients.values()])

    if len(all_ids) != len(rows):
        raise ValueError(f"Clients hold {len(all_ids)} rows, the file has {len(rows)}.")

    if all_ids.duplicated().any():
        raise ValueError("Some rows were given to more than one client.")

    total_attacks = sum(int(client[config.LABEL_COLUMN].sum()) for client in clients.values())
    if total_attacks != int(rows[config.LABEL_COLUMN].sum()):
        raise ValueError("Attack count across clients doesn't match the original file.")


def split_train_val(client_rows):
    return train_test_split(
        client_rows,
        test_size=config.VALIDATION_FRACTION,
        stratify=client_rows[config.CATEGORY_COLUMN],
        random_state=config.SPLIT_SEED,
    )


def build_count_table(clients, parts):
    """Rows = attack family, columns = client (and train/val), values = counts."""
    columns = {}
    for client_id, client_rows in clients.items():
        train_rows, val_rows = parts[client_id]
        columns[f"client_{client_id}"] = client_rows[config.CATEGORY_COLUMN].value_counts()
        columns[f"client_{client_id}_train"] = train_rows[config.CATEGORY_COLUMN].value_counts()
        columns[f"client_{client_id}_val"] = val_rows[config.CATEGORY_COLUMN].value_counts()

    table = pd.DataFrame(columns).fillna(0).astype(int)

    # Put Normal first, then attack families from largest to smallest.
    order = ["Normal"] + [c for c in table.sum(axis=1).sort_values(ascending=False).index if c != "Normal"]
    table = table.loc[order]

    table.loc["Attack (label 1)"] = table.drop(index="Normal").sum()
    table.loc["Total"] = table.loc["Normal"] + table.loc["Attack (label 1)"]
    return table


def main():
    rows = load_raw_training_rows()
    print(f"Loaded {len(rows):,} raw training rows from {config.TRAIN_CSV.name}")

    clients = assign_rows_to_clients(rows)
    check_split(rows, clients)

    parts = {}
    for client_id, client_rows in clients.items():
        train_rows, val_rows = split_train_val(client_rows)
        parts[client_id] = (train_rows, val_rows)

        client_dir = config.CLIENTS_DIR / f"client_{client_id}"
        client_dir.mkdir(parents=True, exist_ok=True)
        train_rows.to_csv(client_dir / "train.csv", index=False)
        val_rows.to_csv(client_dir / "val.csv", index=False)

    table = build_count_table(clients, parts)

    summary_columns = [f"client_{k}" for k in clients]
    print("\nRows per client (before the train/val split):\n")
    print(table[summary_columns].to_string())

    print("\nShare of attacks per client:")
    for client_id, client_rows in clients.items():
        share = client_rows[config.LABEL_COLUMN].mean()
        print(f"  client_{client_id}: {share:.1%}")

    print("\nTrain / validation sizes:")
    for client_id, (train_rows, val_rows) in parts.items():
        print(f"  client_{client_id}: {len(train_rows):,} train, {len(val_rows):,} val")

    output_dir = config.RESULTS_DIR / "review1"
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "client_split_counts.csv")

    print(f"\nAll {len(rows):,} rows assigned exactly once.")
    print(f"Client files saved in {config.CLIENTS_DIR}")
    print(f"Count table saved to {output_dir / 'client_split_counts.csv'}")


if __name__ == "__main__":
    main()
