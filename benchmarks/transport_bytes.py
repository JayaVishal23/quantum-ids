"""
What each transport setting puts on the wire in one federated round.

Runs one complete round with 3 clients for every key-exchange setting and
reports the real message sizes, so the comparison does not depend on a training
run finishing. Nothing is trained here: the weights are a stand-in of exactly
the size the quantum model sends (81 floats).

Run from the project root:
    python -m benchmarks.transport_bytes
"""

import pandas as pd
import torch

from common import config
from crypto.secure_fl import NO_ENCRYPTION, SecureTransport
from models.factory import build_model
from training.loops import copy_weights

TRANSPORTS = [
    (NO_ENCRYPTION, "No encryption"),
    ("x25519", "X25519 (classical)"),
    ("mlkem768", "ML-KEM-768 (post-quantum)"),
    ("hybrid", "Hybrid X25519 + ML-KEM-768"),
]

OUTPUT_DIR = config.RESULTS_DIR / "secured"


def quantum_sized_weights():
    """The real quantum model's weights.

    Built from the actual model rather than stand-in tensors, because the
    serialised payload carries the parameter names, so invented names would
    give a payload size that disagrees with the real training runs.
    """
    torch.manual_seed(0)
    return copy_weights(build_model("qnn"))


def one_round(key_exchange, weights):
    client_ids = sorted(config.CLIENT_ATTACKS)
    transport = SecureTransport(client_ids, key_exchange=key_exchange, signature="mldsa65")
    transport.start_round(1)
    for client_id in client_ids:
        transport.send_global_model(client_id, weights)   # handshake happens here
        transport.send_update(client_id, weights)
    return transport.round_cost()


def main():
    weights = quantum_sized_weights()
    parameters = sum(w.numel() for w in weights.values())
    clients = len(config.CLIENT_ATTACKS)
    print(f"One round, {clients} clients, {parameters} parameters per update\n")

    rows = []
    for key_exchange, label in TRANSPORTS:
        cost = one_round(key_exchange, weights)
        rows.append({
            "transport": label,
            "handshake bytes": cost["handshake_bytes"],
            "upload bytes": cost["upload_bytes"],
            "download bytes": cost["download_bytes"],
            "total bytes": cost["total_bytes"],
            "handshake per client": cost["handshake_bytes"] // clients if cost["handshake_bytes"] else 0,
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    classical = table.loc[table["transport"].str.startswith("X25519"), "total bytes"].iloc[0]
    print()
    for _, row in table.iterrows():
        if not row["transport"].startswith("X25519") and row["total bytes"]:
            extra = row["total bytes"] - classical
            print(f"{row['transport']}: {extra:+,} bytes per round against the classical baseline")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "transport_bytes_per_round.csv", index=False)
    print(f"\nSaved transport_bytes_per_round.csv in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
