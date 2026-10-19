"""
Step A2: federated feature selection.

Goal: all 3 clients must use the same 8 features, but no one is allowed to
see all the data. So we split the work:

  CLIENT (runs on its own data only)
    - takes a random sample of its own training rows
    - scores each of the 39 numeric features with mutual information,
      i.e. "how much does this feature tell me about attack vs normal?"
    - sends the server a small message: 39 scores + its row count

  SERVER (never sees a single row)
    - averages the scores, giving bigger clients more weight
    - keeps the 8 highest-scoring features
    - sends that list back to everyone

Run from the project root:
    python -m federated.feature_selection_fed
"""

import json

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from common import config


# ------------------------------------------------------------------
# Client side
# ------------------------------------------------------------------

def client_score_features(client_id):
    """Runs inside one client. Returns the only thing that client shares."""
    train_rows = pd.read_csv(config.CLIENTS_DIR / f"client_{client_id}" / "train.csv")

    # Mutual information is slow on big tables, so score a random sample.
    sample_size = min(config.MI_ROWS_PER_CLIENT, len(train_rows))
    sample = train_rows.sample(n=sample_size, random_state=config.SPLIT_SEED)

    scores = mutual_info_classif(
        sample[config.NUMERIC_FEATURES],
        sample[config.LABEL_COLUMN],
        random_state=config.SPLIT_SEED,
    )

    # This dictionary is the whole message sent to the server.
    return {
        "client_id": client_id,
        "num_rows": len(train_rows),
        "scores": dict(zip(config.NUMERIC_FEATURES, scores.tolist())),
    }


# ------------------------------------------------------------------
# Server side
# ------------------------------------------------------------------

def server_choose_features(messages):
    """Runs on the server. Only sees the clients' messages, never their rows."""
    total_rows = sum(message["num_rows"] for message in messages)

    combined = pd.Series(0.0, index=config.NUMERIC_FEATURES)
    for message in messages:
        weight = message["num_rows"] / total_rows
        combined += weight * pd.Series(message["scores"])

    combined = combined.sort_values(ascending=False)
    selected = combined.head(config.NUM_SELECTED_FEATURES).index.tolist()
    return selected, combined


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

def build_score_table(messages, combined):
    table = pd.DataFrame({"combined": combined})
    for message in messages:
        name = f"client_{message['client_id']}"
        table[name] = pd.Series(message["scores"])
        table[f"{name}_rank"] = table[name].rank(ascending=False).astype(int)
    table.insert(0, "rank", range(1, len(table) + 1))
    table.index.name = "feature"
    return table


def build_top_k_table(messages, selected):
    """Side by side: each client's own top 8 vs the agreed top 8."""
    columns = {}
    for message in messages:
        own_top = pd.Series(message["scores"]).sort_values(ascending=False)
        columns[f"client_{message['client_id']}"] = own_top.head(config.NUM_SELECTED_FEATURES).index.tolist()
    columns["agreed"] = selected
    table = pd.DataFrame(columns, index=range(1, config.NUM_SELECTED_FEATURES + 1))
    table.index.name = "position"
    return table


def main():
    messages = []
    for client_id in config.CLIENT_ATTACKS:
        print(f"Client {client_id}: scoring features on its own data...")
        messages.append(client_score_features(client_id))

    selected, combined = server_choose_features(messages)

    score_table = build_score_table(messages, combined)
    top_k_table = build_top_k_table(messages, selected)

    print("\nEach client's own top 8 vs the agreed top 8:\n")
    print(top_k_table.to_string())

    print("\nScores of the agreed features (higher = more useful):\n")
    print(score_table.head(config.NUM_SELECTED_FEATURES).round(3).to_string())

    # What the server keeps and sends back to clients.
    config.SHARED_DIR.mkdir(parents=True, exist_ok=True)
    shared_file = config.SHARED_DIR / "selected_features.json"
    with open(shared_file, "w") as f:
        json.dump(
            {
                "features": selected,
                "method": "mutual information, averaged across clients weighted by row count",
                "client_rows": {f"client_{m['client_id']}": m["num_rows"] for m in messages},
            },
            f,
            indent=2,
        )

    # For the report and slides.
    output_dir = config.RESULTS_DIR / "baseline_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_table.to_csv(output_dir / "feature_scores_per_client.csv")
    top_k_table.to_csv(output_dir / "top8_per_client_vs_agreed.csv")

    print(f"\nAgreed features: {selected}")
    print(f"Saved to {shared_file}")


if __name__ == "__main__":
    main()
