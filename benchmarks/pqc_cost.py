"""
How long each cryptographic operation takes, measured properly.

Timings taken from a federated run can be distorted by whatever else the
computer is doing. This script measures each operation on its own, many times,
and reports mean +/- standard deviation. It then adds them up into the cost of
one federated round.

Method: 50 warm-up calls, then 1000 timed calls per operation, using
time.perf_counter_ns. Nothing else should be running.

Run from the project root:
    python -m benchmarks.pqc_cost
"""

import os
import platform
import statistics
import time

import pandas as pd
from cryptography.hazmat.primitives.asymmetric import ed25519, mldsa, mlkem, x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from common import config
from crypto.channel import derive_session_keys

REPEATS = 1000
WARMUP = 50

# Payload sizes we actually send.
QNN_UPDATE = os.urandom(324 + 150)      # 81 float32 values plus the small header
MLP_UPDATE = os.urandom(1156 + 150)     # 289 float32 values plus the header


def measure(name, call):
    for _ in range(WARMUP):
        call()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) / 1e6)  # milliseconds
    return {
        "operation": name,
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.stdev(samples),
        "repeats": REPEATS,
    }


def benchmark_operations():
    rows = []

    # --- ML-KEM-768 (post-quantum key exchange) ---
    kem_private = mlkem.MLKEM768PrivateKey.generate()
    kem_public_bytes = kem_private.public_key().public_bytes_raw()
    kem_public = mlkem.MLKEM768PublicKey.from_public_bytes(kem_public_bytes)
    _, kem_ciphertext = kem_public.encapsulate()

    rows.append(measure("ML-KEM-768 key generation", mlkem.MLKEM768PrivateKey.generate))
    rows.append(measure("ML-KEM-768 encapsulate", kem_public.encapsulate))
    rows.append(measure("ML-KEM-768 decapsulate", lambda: kem_private.decapsulate(kem_ciphertext)))

    # --- X25519 (classical key exchange) ---
    x_private = x25519.X25519PrivateKey.generate()
    x_public = x_private.public_key()
    other_private = x25519.X25519PrivateKey.generate()
    other_public_bytes = other_private.public_key().public_bytes_raw()

    rows.append(measure("X25519 key generation", x25519.X25519PrivateKey.generate))
    rows.append(measure("X25519 client side (keygen + exchange)",
                        lambda: x25519.X25519PrivateKey.generate().exchange(x_public)))
    rows.append(measure("X25519 server side (exchange)",
                        lambda: x_private.exchange(x25519.X25519PublicKey.from_public_bytes(other_public_bytes))))

    # --- Signatures ---
    message = os.urandom(5000)  # about the size of a handshake transcript
    mldsa_private = mldsa.MLDSA65PrivateKey.generate()
    mldsa_public = mldsa_private.public_key()
    mldsa_signature = mldsa_private.sign(message)
    ed_private = ed25519.Ed25519PrivateKey.generate()
    ed_public = ed_private.public_key()
    ed_signature = ed_private.sign(message)

    rows.append(measure("ML-DSA-65 sign", lambda: mldsa_private.sign(message)))
    rows.append(measure("ML-DSA-65 verify", lambda: mldsa_public.verify(mldsa_signature, message)))
    rows.append(measure("Ed25519 sign", lambda: ed_private.sign(message)))
    rows.append(measure("Ed25519 verify", lambda: ed_public.verify(ed_signature, message)))

    # --- Key derivation and encryption ---
    secret = os.urandom(32)
    transcript = os.urandom(9000)
    rows.append(measure("HKDF-SHA256 key derivation", lambda: derive_session_keys(secret, transcript)))

    aes = AESGCM(os.urandom(32))
    nonce = os.urandom(12)
    header = b"header"
    sealed_qnn = aes.encrypt(nonce, QNN_UPDATE, header)
    sealed_mlp = aes.encrypt(nonce, MLP_UPDATE, header)

    rows.append(measure("AES-256-GCM encrypt (quantum update)", lambda: aes.encrypt(nonce, QNN_UPDATE, header)))
    rows.append(measure("AES-256-GCM decrypt (quantum update)", lambda: aes.decrypt(nonce, sealed_qnn, header)))
    rows.append(measure("AES-256-GCM encrypt (classical update)", lambda: aes.encrypt(nonce, MLP_UPDATE, header)))
    rows.append(measure("AES-256-GCM decrypt (classical update)", lambda: aes.decrypt(nonce, sealed_mlp, header)))

    return pd.DataFrame(rows)


def cost_per_round(timings):
    """Add the operations up into one federated round with 3 clients.

    Per client: the server makes a key pair and signs it, the client verifies and
    encapsulates and signs, the server verifies and decapsulates, both derive keys,
    then two messages are encrypted and decrypted (update up, global model down).
    """
    ms = dict(zip(timings["operation"], timings["mean_ms"]))
    clients = len(config.CLIENT_ATTACKS)

    x25519_total = (ms["X25519 key generation"] + ms["X25519 client side (keygen + exchange)"]
                    + ms["X25519 server side (exchange)"])
    mlkem_total = (ms["ML-KEM-768 key generation"] + ms["ML-KEM-768 encapsulate"]
                   + ms["ML-KEM-768 decapsulate"])

    suites = {
        "X25519 + ML-DSA-65": x25519_total,
        "ML-KEM-768 + ML-DSA-65": mlkem_total,
        # The hybrid suite really does run both exchanges, so its cost is the
        # sum of the two. Everything else in the round is unchanged.
        "Hybrid X25519+ML-KEM-768 + ML-DSA-65": x25519_total + mlkem_total,
    }

    signatures = 2 * ms["ML-DSA-65 sign"] + 2 * ms["ML-DSA-65 verify"]
    derivation = 2 * ms["HKDF-SHA256 key derivation"]
    messages = 2 * (ms["AES-256-GCM encrypt (quantum update)"] + ms["AES-256-GCM decrypt (quantum update)"])

    rows = []
    for name, key_exchange in suites.items():
        per_client = key_exchange + signatures + derivation + messages
        rows.append({
            "suite": name,
            "key exchange (ms)": key_exchange,
            "signatures (ms)": signatures,
            "key derivation (ms)": derivation,
            "encryption (ms)": messages,
            "per client (ms)": per_client,
            f"per round, {clients} clients (ms)": per_client * clients,
        })
    return pd.DataFrame(rows)


def main():
    print("Hardware and software")
    print(f"  machine : {platform.processor() or platform.machine()}")
    print(f"  system  : {platform.system()} {platform.release()}")
    print(f"  python  : {platform.python_version()}")
    print(f"  method  : {WARMUP} warm-up calls, then {REPEATS} timed calls per operation\n")

    timings = benchmark_operations()
    print(timings.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    rounds = cost_per_round(timings)
    print("\nCost of one federated round (adding the operations up)\n")
    print(rounds.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    per_round = rounds.set_index("suite")["per round, 3 clients (ms)"]
    classical = per_round["X25519 + ML-DSA-65"]
    print()
    for suite in per_round.index:
        if suite != "X25519 + ML-DSA-65":
            print(f"{suite} costs {per_round[suite] - classical:+.2f} ms per round "
                  f"compared with the classical suite.")

    output_dir = config.RESULTS_DIR / "secured"
    output_dir.mkdir(parents=True, exist_ok=True)
    timings.to_csv(output_dir / "crypto_operation_timings.csv", index=False)
    rounds.to_csv(output_dir / "crypto_cost_per_round.csv", index=False)
    print(f"\nSaved crypto_operation_timings.csv and crypto_cost_per_round.csv in {output_dir}")


if __name__ == "__main__":
    main()
