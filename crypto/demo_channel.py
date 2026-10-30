"""
Live demo for Review 1: send a REAL trained model through the post-quantum channel.

It uses the FedAvg model trained in step A6, walks through the handshake
printing every message and its size, sends the model encrypted both ways,
checks nothing changed, and then runs the attack tests.

Run from the project root:
    python -m crypto.demo_channel                     (ML-DSA-65 signatures)
    python -m crypto.demo_channel --signature ed25519 (classical signatures)
"""

import argparse

import torch

from common import config
from crypto.channel import ClientChannel, ServerChannel, generate_signing_key, pack
from crypto.payload import bytes_to_weights, weights_to_bytes
from crypto.test_channel import CHECKS

MODEL_FILE = config.RESULTS_DIR / "review1" / "fl_mlp_fedavg" / "global_model_seed42.pt"


def show(direction, message):
    size = len(pack(message))
    print(f"  {direction:<44} {size:>7,} bytes")
    for name, value in message.items():
        if isinstance(value, bytes) and len(value) > 32:
            print(f"      {name:<24} {len(value):>7,} bytes")
    return size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signature", choices=["mldsa65", "ed25519"], default="mldsa65")
    parser.add_argument("--kem", choices=["mlkem768", "x25519", "hybrid"], default="mlkem768",
                        help="mlkem768 = post-quantum, x25519 = the classical baseline, "
                             "hybrid = both together")
    args = parser.parse_args()

    if not MODEL_FILE.exists():
        raise SystemExit(f"Model not found: {MODEL_FILE}\nRun first: python -m fl.run_fl --model mlp")

    print(f"Key exchange: {args.kem} | signatures: {args.signature} | encryption: AES-256-GCM\n")

    print("0. Setup, done once before training")
    server_signing_key = generate_signing_key(args.signature)
    client_signing_key = generate_signing_key(args.signature)
    server = ServerChannel(server_signing_key, registered_clients={1: client_signing_key.public_key()},
                           key_exchange=args.kem)
    client = ClientChannel(1, client_signing_key, pinned_server_public_key=server_signing_key.public_key(),
                           key_exchange=args.kem)
    print("  server knows client 1's public key (registration)")
    print("  client 1 knows the server's public key (pinning)\n")

    print("1. Handshake for round 1")
    server.start_round(1)
    handshake_bytes = 0
    hello = client.start_handshake(1)
    handshake_bytes += show("client -> server  hello", hello)
    reply = server.reply_to_hello(hello)
    handshake_bytes += show(f"server -> client  {args.kem} public key + signature", reply)
    finish = client.finish_handshake(reply)
    print("  client checked the server's signature with the pinned key: OK")
    handshake_bytes += show(f"client -> server  {args.kem} ciphertext + signature", finish)
    server.complete_handshake(finish)
    print("  server checked client 1's signature: OK")
    same_keys = client.send_key == server.sessions[1]["receive_key"]
    print(f"  both sides derived the same AES-256 key: {same_keys}\n")

    print(f"2. Client 1 sends its model update (FedAvg MLP from {MODEL_FILE.parent.name})")
    weights = torch.load(MODEL_FILE)
    payload = weights_to_bytes(weights)
    print(f"  model: {sum(v.numel() for v in weights.values())} parameters -> {len(payload):,} bytes before encryption")
    sealed = client.encrypt_update(payload)
    update_bytes = show("client -> server  encrypted update", sealed)
    received = bytes_to_weights(server.decrypt_update(sealed))
    identical = all(torch.equal(weights[name], received[name]) for name in weights)
    print(f"  server decrypted it; weights identical to the original: {identical}\n")

    print("3. Server sends the global model back")
    sealed_back = server.encrypt_global_model(1, payload)
    show("server -> client  encrypted global model", sealed_back)
    back = bytes_to_weights(client.decrypt_global_model(sealed_back))
    print(f"  client decrypted it; identical: {all(torch.equal(weights[n], back[n]) for n in weights)}\n")

    print("4. Attacks (each must be rejected)")
    for description, check in CHECKS[1:]:
        try:
            reason = check(args.signature, args.kem)
            print(f"  [REJECTED] {description}")
            print(f"             {reason}")
        except AssertionError:
            print(f"  [NOT REJECTED!] {description}")
    print()

    print("5. Size summary for one client, one round")
    print(f"  handshake (3 messages)            {handshake_bytes:>7,} bytes")
    print(f"  model update, plaintext           {len(payload):>7,} bytes")
    print(f"  model update, encrypted message   {update_bytes:>7,} bytes "
          f"(+{update_bytes - len(payload)} bytes: 16-byte tag + header fields)")


if __name__ == "__main__":
    main()
