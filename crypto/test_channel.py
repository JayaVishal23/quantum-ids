"""
Security tests for the secure channel.

Each "attack" function plays the attacker and returns the error message the
channel gave when it rejected the attack. If an attack is NOT rejected, it
raises AssertionError, which means the channel has a hole.

Every attack is tried with both key-exchange suites (ML-KEM-768 and X25519)
and both signature schemes (ML-DSA-65 and Ed25519).

Run from the project root:
    python -m crypto.test_channel
or with pytest:
    pytest crypto/test_channel.py
"""

import sys

import torch
from cryptography.hazmat.primitives.asymmetric import mlkem, x25519

from crypto.channel import (
    KEY_EXCHANGE_SUITES,
    SIGNATURE_SCHEMES,
    ChannelError,
    ClientChannel,
    ServerChannel,
    generate_signing_key,
    pack,
)
from crypto.payload import bytes_to_weights, weights_to_bytes


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_parties(scheme, key_exchange):
    """A server and one registered client (id 1), set up before training starts."""
    server_key = generate_signing_key(scheme)
    client_key = generate_signing_key(scheme)
    server = ServerChannel(server_key, registered_clients={1: client_key.public_key()},
                           key_exchange=key_exchange)
    client = ClientChannel(1, client_key, pinned_server_public_key=server_key.public_key(),
                           key_exchange=key_exchange)
    return server, client, server_key


def do_handshake(server, client, round_number=1):
    server.start_round(round_number)
    hello = client.start_handshake(round_number)
    reply = server.reply_to_hello(hello)
    finish = client.finish_handshake(reply)
    server.complete_handshake(finish)
    return hello, reply, finish


def must_be_rejected(action):
    try:
        action()
    except ChannelError as error:
        return str(error)
    raise AssertionError("The attack was NOT rejected.")


def attacker_public_key(key_exchange):
    """A public key the attacker generated, for the suite in use."""
    if key_exchange == "x25519":
        return x25519.X25519PrivateKey.generate().public_key().public_bytes_raw()
    return mlkem.MLKEM768PrivateKey.generate().public_key().public_bytes_raw()


def example_weights():
    torch.manual_seed(0)
    return {"layer.weight": torch.randn(16, 8), "layer.bias": torch.randn(16)}


def same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[name], b[name]) for name in a)


# ------------------------------------------------------------------
# Honest run
# ------------------------------------------------------------------

def honest_exchange(scheme, key_exchange):
    server, client, _ = make_parties(scheme, key_exchange)
    do_handshake(server, client)
    weights = example_weights()

    received_by_server = bytes_to_weights(server.decrypt_update(client.encrypt_update(weights_to_bytes(weights))))
    assert same_weights(weights, received_by_server), "Server got different weights."

    received_by_client = bytes_to_weights(
        client.decrypt_global_model(server.encrypt_global_model(1, weights_to_bytes(weights))))
    assert same_weights(weights, received_by_client), "Client got different weights."
    return "weights arrived bit-for-bit identical in both directions"


# ------------------------------------------------------------------
# Attacks
# ------------------------------------------------------------------

def attack_swap_kem_public_key(scheme, key_exchange):
    """Man-in-the-middle replaces the server's public key with its own."""
    server, client, _ = make_parties(scheme, key_exchange)
    server.start_round(1)
    reply = server.reply_to_hello(client.start_handshake(1))

    forged = {**reply, "kem_public_key": attacker_public_key(key_exchange)}
    return must_be_rejected(lambda: client.finish_handshake(forged))


def attack_swap_key_and_resign(scheme, key_exchange):
    """Smarter man-in-the-middle: swaps the key AND signs the reply with its own signing key."""
    server, client, _ = make_parties(scheme, key_exchange)
    server.start_round(1)
    hello = client.start_handshake(1)
    reply = server.reply_to_hello(hello)

    attacker_signing_key = generate_signing_key(scheme)
    forged = {**reply, "kem_public_key": attacker_public_key(key_exchange)}
    signed_part = {name: forged[name] for name in
                   ("type", "client_id", "round", "suite", "kem_public_key", "server_nonce")}
    forged["signature"] = attacker_signing_key.sign(pack(hello) + pack(signed_part))
    return must_be_rejected(lambda: client.finish_handshake(forged))


def attack_downgrade_suite(scheme, key_exchange):
    """Attacker tries to make the client accept a weaker key exchange than it asked for."""
    server, client, _ = make_parties(scheme, key_exchange)
    other = "x25519" if key_exchange != "x25519" else "mlkem768"
    server.start_round(1)
    reply = server.reply_to_hello(client.start_handshake(1))
    return must_be_rejected(lambda: client.finish_handshake({**reply, "suite": other.encode()}))


def attack_tamper_with_update(scheme, key_exchange):
    """Attacker flips one bit of the encrypted model update."""
    server, client, _ = make_parties(scheme, key_exchange)
    do_handshake(server, client)
    sealed = client.encrypt_update(weights_to_bytes(example_weights()))

    changed = bytearray(sealed["ciphertext"])
    changed[100] ^= 0x01
    return must_be_rejected(lambda: server.decrypt_update({**sealed, "ciphertext": bytes(changed)}))


def attack_change_header(scheme, key_exchange):
    """Attacker leaves the ciphertext alone but changes the (unencrypted) header."""
    server, client, _ = make_parties(scheme, key_exchange)
    do_handshake(server, client)
    sealed = client.encrypt_update(weights_to_bytes(example_weights()))
    return must_be_rejected(lambda: server.decrypt_update({**sealed, "seq": 7}))


def attack_replay_update(scheme, key_exchange):
    """Attacker records a valid encrypted update and sends it again."""
    server, client, _ = make_parties(scheme, key_exchange)
    do_handshake(server, client)
    sealed = client.encrypt_update(weights_to_bytes(example_weights()))
    server.decrypt_update(sealed)  # the real one is accepted
    return must_be_rejected(lambda: server.decrypt_update(sealed))


def attack_replay_old_round(scheme, key_exchange):
    """Attacker replays a whole handshake recorded in round 1 during round 2."""
    server, client, _ = make_parties(scheme, key_exchange)
    hello, _, finish = do_handshake(server, client, round_number=1)
    server.start_round(2)

    # Plain replay: the round number gives it away.
    must_be_rejected(lambda: server.reply_to_hello(hello))

    # Replay with the round number edited to 2: the old signature no longer matches.
    server.reply_to_hello({**hello, "round": 2})
    return must_be_rejected(lambda: server.complete_handshake({**finish, "round": 2}))


def attack_unregistered_client(scheme, key_exchange):
    """A stranger who was never registered tries to join."""
    server, _, server_key = make_parties(scheme, key_exchange)
    stranger = ClientChannel(99, generate_signing_key(scheme), server_key.public_key(),
                             key_exchange=key_exchange)
    server.start_round(1)
    return must_be_rejected(lambda: server.reply_to_hello(stranger.start_handshake(1)))


def attack_impersonate_client(scheme, key_exchange):
    """An impostor claims to be client 1 but doesn't have client 1's signing key."""
    server, _, server_key = make_parties(scheme, key_exchange)
    impostor = ClientChannel(1, generate_signing_key(scheme), server_key.public_key(),
                             key_exchange=key_exchange)
    server.start_round(1)
    finish = impostor.finish_handshake(server.reply_to_hello(impostor.start_handshake(1)))
    return must_be_rejected(lambda: server.complete_handshake(finish))


CHECKS = [
    ("Honest client: weights decrypt identically", honest_exchange),
    ("MITM swaps the public key", attack_swap_kem_public_key),
    ("MITM swaps the key and re-signs with its own key", attack_swap_key_and_resign),
    ("MITM downgrades the key-exchange suite", attack_downgrade_suite),
    ("One bit of the encrypted update flipped", attack_tamper_with_update),
    ("Unencrypted header changed", attack_change_header),
    ("Valid update sent a second time", attack_replay_update),
    ("Round-1 handshake replayed in round 2", attack_replay_old_round),
    ("Unregistered client tries to join", attack_unregistered_client),
    ("Impostor claims to be client 1", attack_impersonate_client),
]


# ------------------------------------------------------------------
# pytest entry point
# ------------------------------------------------------------------

def test_all_checks():
    for key_exchange in KEY_EXCHANGE_SUITES:
        for scheme in SIGNATURE_SCHEMES:
            for _, check in CHECKS:
                check(scheme, key_exchange)


def run_all():
    failures = 0
    for key_exchange in KEY_EXCHANGE_SUITES:
        for scheme in SIGNATURE_SCHEMES:
            print(f"\nKey exchange: {key_exchange} | signatures: {scheme}")
            for description, check in CHECKS:
                try:
                    outcome = check(scheme, key_exchange)
                    print(f"  [PASS] {description}")
                    if description.startswith("Honest"):
                        print(f"         -> {outcome}")
                except AssertionError as error:
                    failures += 1
                    print(f"  [FAIL] {description}: {error}")
    print()
    print("All checks passed." if failures == 0 else f"{failures} check(s) FAILED.")
    return failures == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
